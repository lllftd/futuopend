"""Train / infer PyTorch L2 (replaces LGBM when L2_LEGACY_LGBM is unset)."""

from __future__ import annotations

import os
import pickle
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from tqdm.auto import tqdm

from core.training.common.constants import (
    L1A_REGIME_COLS,
    L2_META_FILE,
    L2_OUTPUT_CACHE_FILE,
    L2_SCHEMA_VERSION,
    L2_UNIFIED_MODEL_FILE,
    L2_UNIFIED_TRAIN_CHECKPOINT_FILE,
    MODEL_DIR,
    NUM_REGIME_CLASSES,
)
from core.training.unified.config.env import env_bool, env_float, env_int
from core.training.unified.model import UnifiedL2L3Config, UnifiedL2L3Net, split_feature_indices
from core.training.common.stack_v2_common import save_output_cache
from core.training.common.threshold_registry import attach_threshold_registry, threshold_entry


@dataclass
class L2PytorchContext:
    train_started_at: datetime
    train_started_perf: float
    df: pd.DataFrame
    l1a_outputs: pd.DataFrame
    l1b_outputs: pd.DataFrame
    frame: pd.DataFrame
    feature_cols: list[str]
    range_feature_cols: list[str]
    X: np.ndarray
    X_range: np.ndarray
    X_train_fit: np.ndarray
    X_range_train_fit: np.ndarray
    y_trade: np.ndarray
    y_range: np.ndarray
    y_mfe: np.ndarray
    y_mae: np.ndarray
    y_mfe_fit: np.ndarray
    y_mae_fit: np.ndarray
    y_range_fit: np.ndarray
    mfe_w_all: np.ndarray
    mfe_head_prep: dict[str, Any]
    mae_head_prep: dict[str, Any]
    range_head_prep: dict[str, Any]
    y_r10_fit: np.ndarray | None
    y_r20_fit: np.ndarray | None
    y_ttp_fit: np.ndarray | None
    r10_head_prep: dict[str, Any] | None
    r20_head_prep: dict[str, Any] | None
    ttp_head_prep: dict[str, Any] | None
    mh_on: bool
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    fit_train_mask: np.ndarray
    val_tune_mask: np.ndarray
    val_report_mask: np.ndarray
    straddle_label_stats: dict[str, Any]
    derived_feature_stats: dict[str, Any]
    l2_dropped_features: list[str]
    tune_frac: float
    min_std: float
    skip_hard: bool
    n_oof: int


def _unified_grad_clip_max() -> float:
    """``L2_UNIFIED_GRAD_CLIP_MAX`` (default 2.0) — global L2 for Phase1/Phase2 ``clip_grad_norm_(..., max_norm)``.

    ``clip_grad_norm_`` caps the **global** norm; per-submodule norms are not bounded by ``max_norm``.
    Tighten with e.g. ``L2_UNIFIED_GRAD_CLIP_MAX=1`` or ``1.5``.
    """
    return env_float("L2_UNIFIED_GRAD_CLIP_MAX", 2.0)


def _unified_checkpoint_path() -> str:
    name = (os.environ.get("L2_UNIFIED_CHECKPOINT_FILE") or L2_UNIFIED_TRAIN_CHECKPOINT_FILE).strip()
    return os.path.join(MODEL_DIR, name)


def _unified_checkpoint_writes_enabled() -> bool:
    return env_bool("L2_UNIFIED_CHECKPOINT", True)


def _unified_resume_enabled() -> bool:
    return env_bool("L2_UNIFIED_RESUME", False)


def _unified_checkpoint_keep_on_success() -> bool:
    return env_bool("L2_UNIFIED_CHECKPOINT_KEEP", False)


def _unified_checkpoint_every_n() -> int:
    """L2_UNIFIED_CHECKPOINT_EVERY: 0 = only last epoch of a phase, 1 = every epoch (default), N>=2 = every N + last."""
    raw = (os.environ.get("L2_UNIFIED_CHECKPOINT_EVERY") or "1").strip()
    if not raw:
        return 1
    return int(raw)


def _unified_should_write_phase_checkpoint(step_1based: int, total: int) -> bool:
    if not _unified_checkpoint_writes_enabled() or total < 1:
        return False
    every = _unified_checkpoint_every_n()
    if every < 0:
        every = 1
    if step_1based < 1 or step_1based > total:
        return False
    if every == 0:
        return step_1based == total
    if every == 1:
        return True
    return (step_1based % every == 0) or (step_1based == total)


def _unified_resume_ckpt_version() -> int:
    return 1


def _write_unified_training_checkpoint(
    path: str,
    *,
    net: nn.Module,
    optimizer_p1: torch.optim.Optimizer,
    optimizer_p2: torch.optim.Optimizer | None,
    rng: np.random.Generator,
    phase1_completed: int,
    phase2_completed: int,
) -> None:
    """Full training state for resume (Phase1 + optional Phase2). ``rng`` must be the training Generator."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: dict[str, Any] = {
        "version": _unified_resume_ckpt_version(),
        "model": net.state_dict(),
        "optimizer_phase1": optimizer_p1.state_dict(),
        "phase1_completed": int(phase1_completed),
        "phase2_completed": int(phase2_completed),
        "rng_state": pickle.dumps(rng, protocol=pickle.HIGHEST_PROTOCOL),
    }
    if optimizer_p2 is not None:
        payload["optimizer_phase2"] = optimizer_p2.state_dict()
    else:
        payload["optimizer_phase2"] = None
    torch.save(payload, path)


def _load_unified_training_checkpoint(path: str) -> dict[str, Any] | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        print(f"  [L2 unified][warn] could not load checkpoint {path!r}: {e}", flush=True)
        return None
    if int(d.get("version", 0)) != _unified_resume_ckpt_version():
        print(
            f"  [L2 unified][warn] checkpoint version mismatch; ignoring {path!r}.",
            flush=True,
        )
        return None
    return d


def _restore_rng_from_checkpoint(blob: bytes) -> np.random.Generator:
    return pickle.loads(blob)


def _tensor_device() -> torch.device:
    """Prefer CUDA, then MPS (Apple Silicon), else CPU. ``L2_UNIFIED_DEVICE`` overrides: cpu/0, cuda, mps."""
    raw = (os.environ.get("L2_UNIFIED_DEVICE", "") or "").strip().lower()
    if raw in {"cpu", "0"}:
        return torch.device("cpu")
    if raw in {"cuda", "gpu", "1"} and torch.cuda.is_available():
        return torch.device("cuda")
    if raw in {"mps", "metal"}:
        mps_b = getattr(torch.backends, "mps", None)
        if mps_b is not None and mps_b.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_b = getattr(torch.backends, "mps", None)
    if mps_b is not None and mps_b.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _unified_amp_enabled(device: torch.device) -> bool:
    if not env_bool("L2_UNIFIED_AMP", False):
        return False
    return device.type in ("cuda", "mps")


def _maybe_compile_unified_net(net: nn.Module) -> nn.Module:
    if not env_bool("L2_UNIFIED_TORCH_COMPILE", False):
        return net
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        print("  [L2 unified][warn] L2_UNIFIED_TORCH_COMPILE=1 but torch.compile unavailable; skip.", flush=True)
        return net
    try:
        return compile_fn(net)  # type: ignore[no-any-return, misc]
    except Exception as e:  # noqa: BLE001
        print(f"  [L2 unified][warn] torch.compile failed: {e}; using eager.", flush=True)
        return net


def _unified_autocast_cm(device: torch.device):
    if not _unified_amp_enabled(device):
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, enabled=True)


def _regression_pred_inverse(pred_raw: np.ndarray, prep: dict[str, Any] | None, *, clip_max: float) -> np.ndarray:
    cfg = dict(prep or {})
    transform = str(cfg.get("target_transform", "none")).strip().lower() or "none"
    cap = float(cfg.get("clip_max", clip_max))
    pred = pred_raw.astype(np.float64)
    if transform == "log1p":
        pred = np.expm1(pred)
    pred = np.clip(pred, 0.0, cap)
    return pred.astype(np.float32)


def build_column_tensors(
    X: np.ndarray,
    market_idx: list[int],
    regime_idx: list[int],
    n_pos: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = X.shape[0]
    xm = (X[:, market_idx] if market_idx else np.zeros((n, 0), np.float32)).astype(np.float32, copy=False)
    xr = (X[:, regime_idx] if regime_idx else np.zeros((n, 0), np.float32)).astype(np.float32, copy=False)
    xp = np.zeros((n, n_pos), dtype=np.float32)
    return (
        torch.as_tensor(xm, device=device, dtype=torch.float32),
        torch.as_tensor(xr, device=device, dtype=torch.float32),
        torch.as_tensor(xp, device=device, dtype=torch.float32),
    )


@torch.inference_mode()
def infer_l2_unified_raw(
    model: nn.Module,
    X: np.ndarray,
    meta: dict[str, Any],
    *,
    batch_size: int = 16384,
) -> dict[str, np.ndarray]:
    ucfg = meta.get("l2_unified_config") or {}
    market_idx = [int(i) for i in ucfg.get("market_idx", [])]
    regime_idx = [int(i) for i in ucfg.get("regime_idx", [])]
    n_pos = int(ucfg.get("n_position", 8))
    device = _tensor_device()
    model.eval()
    model.to(device)
    n = int(X.shape[0])
    acc: dict[str, list] = {k: [] for k in ("gate_logit", "range", "mfe", "mae", "range_10", "range_20", "ttp90")}
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        xb = X[start:end]
        bxm, bxr, bxp = build_column_tensors(xb, market_idx, regime_idx, n_pos, device)
        raw = model(bxm, bxr, bxp)
        for k in ("gate_logit", "range", "mfe", "mae"):
            acc[k].append(raw[k].detach().float().cpu().numpy())
        for k2 in ("range_10", "range_20", "ttp90"):
            if k2 in raw:
                acc[k2].append(raw[k2].detach().float().cpu().numpy())
            else:
                acc[k2].append(np.zeros(end - start, dtype=np.float32))
    return {k: np.concatenate(v) for k, v in acc.items()}


def _fit_gate_calibrator(
    val_tune_mask: np.ndarray,
    y_trade: np.ndarray,
    trade_logits: np.ndarray,
) -> Any:
    m = np.asarray(val_tune_mask, dtype=bool) & np.isfinite(trade_logits)
    if int(m.sum()) < 20:
        return None
    from torch.nn.functional import sigmoid
    p = sigmoid(torch.as_tensor(trade_logits[m])).numpy().astype(np.float64)
    y = y_trade[m].astype(np.int32)
    try:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        return iso.fit(p, y)
    except Exception:
        return None


def _pinball_loss_quantiles(
    pred: "torch.Tensor",
    target: "torch.Tensor",
    quantiles: tuple[float, ...],
) -> "torch.Tensor":
    """``pred`` [B, Q], ``target`` [B] — one scalar outcome per row, pinball for each output column."""
    import torch

    t = target.unsqueeze(1).expand_as(pred)
    qv = pred.new_tensor(quantiles, dtype=pred.dtype, device=pred.device).view(1, -1)
    err = t - pred
    return torch.maximum(qv * err, (qv - 1.0) * err).mean()


def _unified_head_weight_norms(net: torch.nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    for n, p in net.named_parameters():
        if p.requires_grad and "head" in n and "weight" in n:
            out[n] = round(float(p.detach().float().cpu().norm().item()), 4)
    return out


def _l2_unified_dim_check(
    feature_cols: list[str],
    market_idx: list[int],
    regime_idx: list[int],
    n_m: int,
    n_r: int,
    n_pos: int,
    position_col_names: list[str],
) -> dict[str, Any]:
    return {
        "n_feature_cols": int(len(feature_cols)),
        "n_market": int(n_m),
        "n_regime": int(n_r),
        "n_position": int(n_pos),
        "sum_check": int(n_m + n_r + n_pos),
        "market_cols": [feature_cols[i] for i in market_idx[:5]],
        "regime_cols": [feature_cols[i] for i in regime_idx[:5]],
        "position_cols": list(position_col_names),
    }


def _l2_unified_train_input_stats_numpy(
    X: np.ndarray,
    fit_mask: np.ndarray,
    market_idx: list[int],
    regime_idx: list[int],
    feature_cols: list[str],
    n_pos: int,
) -> dict[str, Any]:
    m = np.asarray(fit_mask, dtype=bool)
    if not m.any():
        return {
            "train_market_mean": [],
            "train_market_std": [],
            "train_regime_mean": [],
            "train_regime_std": [],
            "train_position_mean": [],
            "train_position_std": [],
        }
    from core.training.unified.position_features import build_position_matrix

    Xf = X[m]
    mm = (Xf[:, market_idx] if market_idx else np.zeros((Xf.shape[0], 0), np.float32)).astype(
        np.float32, copy=False
    )
    rm = (Xf[:, regime_idx] if regime_idx else np.zeros((Xf.shape[0], 0), np.float32)).astype(
        np.float32, copy=False
    )
    xpp, _ = build_position_matrix(Xf, list(feature_cols), n_pos)
    return {
        "train_market_mean": mm.mean(axis=0, dtype=np.float64).astype(np.float32).tolist() if mm.size else [],
        "train_market_std": mm.std(axis=0, dtype=np.float64).astype(np.float32).tolist() if mm.size else [],
        "train_regime_mean": rm.mean(axis=0, dtype=np.float64).astype(np.float32).tolist() if rm.size else [],
        "train_regime_std": rm.std(axis=0, dtype=np.float64).astype(np.float32).tolist() if rm.size else [],
        "train_position_mean": xpp.mean(axis=0, dtype=np.float64).astype(np.float32).tolist() if xpp.size else [],
        "train_position_std": xpp.std(axis=0, dtype=np.float64).astype(np.float32).tolist() if xpp.size else [],
    }


@torch.inference_mode()
def _l2_unified_calibration_summary(
    net: torch.nn.Module,
    ctx: L2PytorchContext,
    l2_outputs: pd.DataFrame,
    market_idx: list[int],
    regime_idx: list[int],
    n_pos: int,
    device: torch.device,
    tr_idx: np.ndarray,
    xm: torch.Tensor,
    xr: torch.Tensor,
    batch_sz: int,
    p2_wanted: bool,
    rng: np.random.Generator,
    value_target_norm: dict[str, Any] | None = None,
    policy_build_cache: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import mean_squared_error, roc_auc_score

    from core.training.common.val_metrics_extra import brier_binary, pearson_corr

    cal_max = int((os.environ.get("L2_UNIFIED_CAL_MAX_ROWS") or "0").strip() or 0)
    tix = tr_idx
    if cal_max > 0 and tix.size > cal_max:
        tix = rng.choice(tix, size=cal_max, replace=False)
    if tix.size < 1:
        return {"error": "empty_train_index"}

    net.eval()
    y_tr = np.asarray(ctx.y_trade, dtype=np.float64)
    y_range_f = np.asarray(ctx.y_range_fit, dtype=np.float64)
    y_mfe_f = np.asarray(ctx.y_mfe_fit, dtype=np.float64)
    y_mae_f = np.asarray(ctx.y_mae_fit, dtype=np.float64)

    g_pred: list[np.ndarray] = []
    g_true: list[np.ndarray] = []
    r_p: list[np.ndarray] = []
    r_t: list[np.ndarray] = []
    mf_p: list[np.ndarray] = []
    mf_t: list[np.ndarray] = []
    ma_p: list[np.ndarray] = []
    ma_t: list[np.ndarray] = []
    r10_p, r20_p, ttp_p = [], [], []
    r10_t, r20_t, ttp_t = [], [], []

    for s0 in range(0, tix.size, batch_sz):
        b = tix[s0 : s0 + batch_sz]
        bxm, bxr = xm[b], xr[b]
        bxp0 = torch.zeros((b.size, n_pos), device=device, dtype=torch.float32)
        o = net(bxm, bxr, bxp0)
        gl = o["gate_logit"].detach().float().cpu().numpy().ravel()
        pr = 1.0 / (1.0 + np.exp(-np.clip(gl, -20, 20)))
        g_pred.append(pr.astype(np.float64))
        g_true.append(y_tr[b])
        r_p.append(o["range"].detach().float().cpu().numpy().ravel())
        r_t.append(y_range_f[b])
        mfe_m = (o["mfe"].detach().float().cpu().numpy().ravel(),)
        mfe_t = (y_mfe_f[b],)
        mf_p.append(mfe_m[0])
        mf_t.append(mfe_t[0])
        ma_p.append(o["mae"].detach().float().cpu().numpy().ravel())
        ma_t.append(y_mae_f[b])
        if ctx.mh_on and "range_10" in o:
            r10_p.append(o["range_10"].detach().float().cpu().numpy().ravel())
            r20_p.append(o["range_20"].detach().float().cpu().numpy().ravel())
            ttp_p.append(o["ttp90"].detach().float().cpu().numpy().ravel())
            r10_t.append(np.asarray(ctx.y_r10_fit, dtype=np.float64)[b])
            r20_t.append(np.asarray(ctx.y_r20_fit, dtype=np.float64)[b])
            ttp_t.append(np.asarray(ctx.y_ttp_fit, dtype=np.float64)[b])

    g_predv = np.concatenate(g_pred) if g_pred else np.array([])
    g_truev = np.concatenate(g_true) if g_true else np.array([])

    def _auc_or_none(y: np.ndarray, p: np.ndarray) -> float | None:
        yb = y.astype(int)
        if len(yb) < 2 or yb.min() == yb.max():
            return None
        try:
            return float(roc_auc_score(yb, p))
        except Exception:  # noqa: BLE001
            return None

    def _reg_block(p_vec: np.ndarray, t_vec: np.ndarray) -> dict[str, Any]:
        p_vec = np.asarray(p_vec, np.float64).ravel()
        t_vec = np.asarray(t_vec, np.float64).ravel()
        m = np.isfinite(p_vec) & np.isfinite(t_vec)
        if int(m.sum()) < 1:
            return {"pred_mean": None, "label_mean": None, "rmse": None, "corr": None}
        p_vec, t_vec = p_vec[m], t_vec[m]
        cr = pearson_corr(t_vec, p_vec)
        return {
            "pred_mean": float(np.mean(p_vec)),
            "label_mean": float(np.mean(t_vec)),
            "rmse": float(np.sqrt(mean_squared_error(t_vec, p_vec))),
            "corr": float(cr) if cr == cr else None,
        }

    summary: dict[str, Any] = {
        "gate": {
            "pred_mean": float(np.mean(g_predv)) if g_predv.size else None,
            "label_mean": float(np.mean(g_truev)) if g_truev.size else None,
            "brier": float(brier_binary(g_truev, g_predv)) if g_truev.size else None,
            "auroc": _auc_or_none(g_truev, g_predv),
        },
        "range": _reg_block(np.concatenate(r_p), np.concatenate(r_t)),
        "mfe": _reg_block(np.concatenate(mf_p), np.concatenate(mf_t)),
        "mae": _reg_block(np.concatenate(ma_p), np.concatenate(ma_t)),
    }
    if ctx.mh_on and r10_p:
        summary["range_10"] = _reg_block(np.concatenate(r10_p), np.concatenate(r10_t))
        summary["range_20"] = _reg_block(np.concatenate(r20_p), np.concatenate(r20_t))
        summary["ttp90"] = _reg_block(np.concatenate(ttp_p), np.concatenate(ttp_t))

    if p2_wanted and getattr(net, "exit_head", None) is not None:
        from core.training.unified.config.defaults import max_hold_bars
        from core.training.unified.exit_labels import remaining_bars_to_episode_end, urgency_from_y_exit
        from core.training.unified.policy_data import _build_l3_policy_dataset
        from core.training.unified.position_features import build_position_matrix
        from core.training.unified.trajectory import L3TrajectoryConfig

        try:
            if (
                policy_build_cache is not None
                and isinstance(policy_build_cache, tuple)
                and len(policy_build_cache) == 10
            ):
                Xp, y_exit, y_value, _t_st, fcols, rows_entry, _, _, _rfm, pol = policy_build_cache
                print(
                    "  [L2 unified] calibration: reusing Phase2 L3 policy dataset (no rebuild).",
                    flush=True,
                )
            else:
                Xp, y_exit, y_value, _t_st, fcols, rows_entry, _, _, _rfm, pol = _build_l3_policy_dataset(
                    ctx.df,
                    ctx.l1a_outputs,
                    l2_outputs,
                    max_hold=max_hold_bars(),
                    build_traj=False,
                    traj_cfg=L3TrajectoryConfig(),
                )
        except Exception:  # noqa: BLE001
            return summary
        if len(Xp) < 8:
            return summary
        _midx2 = pol.get("policy_rows_merged_idx")
        merged = np.asarray([] if _midx2 is None else _midx2, dtype=np.int64)
        if merged.size != len(y_exit) or merged.size == 0:
            return summary
        n_all2 = int(ctx.X.shape[0])
        valid = (merged >= 0) & (merged < n_all2)
        fit_m = np.asarray(ctx.fit_train_mask, dtype=bool)
        ex_ok = valid & fit_m[merged]
        ex_i = np.flatnonzero(ex_ok)
        if ex_i.size < 8:
            return summary
        vn = value_target_norm
        use_vn = bool(p2_wanted and isinstance(vn, dict) and vn.get("enabled", True))
        pm = ps = bm = bs = 0.0
        if use_vn:
            pm = float(vn.get("remaining_pnl_mean", 0.0))
            ps = max(float(vn.get("remaining_pnl_std", 1.0)), 1e-8)
            bm = float(vn.get("remaining_bars_mean", 0.0))
            bs = max(float(vn.get("remaining_bars_std", 1.0)), 1e-8)
        pos_x, _ = build_position_matrix(Xp, list(fcols), n_pos)
        y_ex = y_exit[ex_i].astype(np.float32)
        y_va = y_value[ex_i].astype(np.float32)
        u_t = urgency_from_y_exit(y_exit[ex_i], rows_entry[ex_i])
        ih0 = int(fcols.index("l3_hold_bars"))
        y_ba_full = remaining_bars_to_episode_end(rows_entry, Xp[:, ih0])
        y_ba = y_ba_full[ex_i].astype(np.float32)
        m_idx = merged[ex_i]
        xpm = (ctx.X[m_idx][:, market_idx] if market_idx else np.zeros((ex_i.size, 0), np.float32)).astype(
            np.float32, copy=False
        )
        xpr = (ctx.X[m_idx][:, regime_idx] if regime_idx else np.zeros((ex_i.size, 0), np.float32)).astype(
            np.float32, copy=False
        )
        xpp = pos_x[ex_i].astype(np.float32, copy=False)
        ex_max = ex_i.size
        cal2 = ex_max if (cal_max <= 0) else min(cal_max, ex_max)
        if cal2 < ex_max:
            sub = np.sort(rng.choice(ex_i.size, size=cal2, replace=False))
        else:
            sub = np.arange(ex_i.size, dtype=np.int64)
        cl_p, cl_t = [], []
        u_p, u_ta = [], []
        vp, vt = [], []
        vb_p, vb_t = [], []
        for s0 in range(0, sub.size, batch_sz):
            j = sub[s0 : s0 + batch_sz]
            t_xpm = torch.as_tensor(xpm[j], device=device, dtype=torch.float32)
            t_xpr = torch.as_tensor(xpr[j], device=device, dtype=torch.float32)
            t_xpp = torch.as_tensor(xpp[j], device=device, dtype=torch.float32)
            o2 = net(t_xpm, t_xpr, t_xpp, return_all=True)
            close_log = o2["exit_close_logit"].detach().float().cpu().numpy().ravel()
            p_close = 1.0 / (1.0 + np.exp(-np.clip(close_log, -20, 20)))
            cl_p.append(p_close)
            cl_t.append(y_ex[j].astype(np.float64))
            u_p.append(torch.sigmoid(o2["exit_urgency_logit"]).detach().float().cpu().numpy().ravel())
            u_ta.append(u_t[j].astype(np.float64))
            vp.append(o2["value_remaining_pnl"].detach().float().cpu().numpy().ravel())
            yv = y_va[j].astype(np.float64)
            yb = y_ba[j].astype(np.float64)
            if use_vn:
                yv = (yv - pm) / ps
                yb = (yb - bm) / bs
            vt.append(yv)
            vb_p.append(o2["value_remaining_bars"].detach().float().cpu().numpy().ravel())
            vb_t.append(yb)
        c_pv = np.concatenate(cl_p) if cl_p else np.array([])
        c_tv = np.concatenate(cl_t) if cl_t else np.array([])
        _lpr = float(np.mean(c_tv)) if c_tv.size else None
        summary["exit_close"] = {
            "pred_mean": float(np.mean(c_pv)) if c_pv.size else None,
            "label_mean": float(np.mean(c_tv)) if c_tv.size else None,
            "label_pos_rate": _lpr,
            "brier": float(brier_binary(c_tv, c_pv)) if c_pv.size else None,
            "auroc": _auc_or_none(c_tv, c_pv),
        }
        summary["exit_urgency"] = _reg_block(np.concatenate(u_p), np.concatenate(u_ta))
        vpv = np.concatenate(vp) if vp else np.array([])
        vtv = np.concatenate(vt) if vt else np.array([])
        vpb = np.concatenate(vb_p) if vb_p else np.array([])
        vbb = np.concatenate(vb_t) if vb_t else np.array([])
        summary["value_target_calibration_space"] = "normalized" if use_vn else "native"
        summary["value_remaining_pnl"] = _reg_block(vpv, vtv)
        summary["value_remaining_bars"] = _reg_block(vpb, vbb)
        if use_vn and vpv.size and vtv.size:
            y_va_sub = y_va[sub].astype(np.float64)
            y_ba_sub = y_ba[sub].astype(np.float64)
            summary["value_remaining_pnl_native"] = _reg_block(vpv * ps + pm, y_va_sub)
            summary["value_remaining_bars_native"] = _reg_block(vpb * bs + bm, y_ba_sub)
    return summary


def _unified_l2_phase2_joint(
    net: nn.Module,
    ctx: L2PytorchContext,
    l2_outputs: pd.DataFrame,
    market_idx: list[int],
    regime_idx: list[int],
    n_pos: int,
    device: torch.device,
    xm: torch.Tensor,
    xr: torch.Tensor,
    tr_idx: np.ndarray,
    gate_y: torch.Tensor,
    y_range_fit_t: torch.Tensor,
    y_mfe_fit_t: torch.Tensor,
    y_mae_fit_t: torch.Tensor,
    mfe_w: torch.Tensor,
    mfe_m: torch.Tensor,
    y_r10_t: torch.Tensor | None,
    y_r20_t: torch.Tensor | None,
    y_ttp_t: torch.Tensor | None,
    mh_on: bool,
    batch_sz: int,
    rng: np.random.Generator,
    bce: nn.Module,
    huber: nn.Module,
    *,
    resume_phase2_completed: int = 0,
    optimizer_phase2_state: dict[str, Any] | None = None,
    checkpoint_path: str | None = None,
    p1_epochs_total: int = 0,
    opt_p1: torch.optim.Optimizer | None = None,
    rng_for_ckpt: np.random.Generator | None = None,
) -> dict[str, Any]:
    from torch.nn import functional as F

    from core.training.unified.config.defaults import max_hold_bars
    from core.training.unified.exit_labels import remaining_bars_to_episode_end, urgency_from_y_exit
    from core.training.unified.policy_data import _build_l3_policy_dataset, _l3_exit_scale_pos_weight_from_train
    from core.training.unified.position_features import build_position_matrix
    from core.training.unified.trajectory import L3TrajectoryConfig
    from core.training.common.lgbm_utils import _tqdm_stream

    p2_ep = env_int("L2_UNIFIED_PHASE2_EPOCHS", 8, lo=1)
    p2_lr = env_float("L2_UNIFIED_PHASE2_LR", 3e-4)
    w_e = env_float("L2_UNIFIED_P2_W_ENTRY", 1.0)
    w_ex = env_float("L2_UNIFIED_P2_W_EXIT", 0.5)
    w_v = env_float("L2_UNIFIED_P2_W_VAL", 0.3)
    p2_entry_ratio = env_float("L2_UNIFIED_P2_ENTRY_RATIO", 0.5)
    p2_w_bars = env_float("L2_UNIFIED_P2_W_BARS_MULT", 0.3)
    p2_w_q = env_float("L2_UNIFIED_P2_W_Q_MULT", 0.5)
    use_bars = (os.environ.get("L2_UNIFIED_P2_USE_BARS", "1") or "1").strip().lower() not in ("0", "false", "no")
    use_q = (os.environ.get("L2_UNIFIED_P2_USE_Q", "1") or "1").strip().lower() not in ("0", "false", "no")
    q_levels = (0.05, 0.25, 0.5, 0.75, 0.95)

    Xp, y_exit, y_value, _t_st, fcols, rows_entry, _traj_ph, _traj_len, _rfm, pol = _build_l3_policy_dataset(
        ctx.df,
        ctx.l1a_outputs,
        l2_outputs,
        max_hold=max_hold_bars(),
        build_traj=False,
        traj_cfg=L3TrajectoryConfig(),
    )
    policy_tup: tuple[Any, ...] = (
        Xp,
        y_exit,
        y_value,
        _t_st,
        fcols,
        rows_entry,
        _traj_ph,
        _traj_len,
        _rfm,
        pol,
    )
    if len(Xp) < 32:
        print("  [L2 unified] Phase2 skipped: policy dataset too small.", flush=True)
        return {"policy_build_cache": policy_tup}
    _midx = pol.get("policy_rows_merged_idx")
    # Do not use ``x or []``: policy_data may store a numpy ndarray; ``bool(ndarray)`` is ambiguous.
    merged_idx = np.asarray([] if _midx is None else _midx, dtype=np.int64)
    if merged_idx.size != len(y_exit) or merged_idx.size == 0:
        print("  [L2 unified] Phase2 skipped: missing policy_rows_merged_idx.", flush=True)
        return {"policy_build_cache": policy_tup}
    n_all2 = int(ctx.X.shape[0])
    valid = (merged_idx >= 0) & (merged_idx < n_all2)
    fit_m = np.asarray(ctx.fit_train_mask, dtype=bool)
    ex_ok = valid & fit_m[merged_idx]
    ex_i = np.flatnonzero(ex_ok)
    if ex_i.size < 32:
        print("  [L2 unified] Phase2 skipped: not enough hold rows on L2 fit mask.", flush=True)
        return {"policy_build_cache": policy_tup}

    pos_x, _ = build_position_matrix(Xp, list(fcols), n_pos)
    y_ex = y_exit[ex_i].astype(np.float32)
    exit_pos_rate = float(np.clip(np.mean(y_ex.astype(np.float64)), 0.0, 1.0))

    def _p2_exit_bce_pos_weight(p: float) -> float:
        """Default 1.0: mild exit-label imbalance (e.g. ~23% pos) needs no BCE reweight; tune OOS threshold instead.

        Set L2_UNIFIED_P2_EXIT_POS_WEIGHT=auto (or lgbm) to apply the same step rule as LGBM ``scale_pos_weight``
        (can jump e.g. 20% → 21%); use a positive float to force a value.
        """
        raw = (os.environ.get("L2_UNIFIED_P2_EXIT_POS_WEIGHT") or "").strip().lower()
        if raw in ("", "1", "1.0", "default", "off", "none"):
            return 1.0
        if raw in ("auto", "lgbm", "scale_pos"):
            pw = _l3_exit_scale_pos_weight_from_train(p)
            return 1.0 if pw is None else float(pw)
        return float(raw)

    p2_exit_bce_pw = _p2_exit_bce_pos_weight(exit_pos_rate)
    y_va = y_value[ex_i].astype(np.float32)
    u_t = urgency_from_y_exit(y_exit[ex_i], rows_entry[ex_i])
    ih0 = int(fcols.index("l3_hold_bars"))
    y_ba_full = remaining_bars_to_episode_end(rows_entry, Xp[:, ih0])
    y_ba = y_ba_full[ex_i].astype(np.float32)
    m_idx = merged_idx[ex_i]
    vnorm_on = env_bool("L2_UNIFIED_VALUE_NORM", True)
    if vnorm_on:
        pnl_m = float(np.mean(y_va))
        pnl_s = float(max(np.std(y_va), 1e-8))
        bars_m = float(np.mean(y_ba))
        bars_s = float(max(np.std(y_ba), 1e-8))
    else:
        pnl_m, pnl_s = 0.0, 1.0
        bars_m, bars_s = 0.0, 1.0
    l2_unified_value_target_norm: dict[str, Any] = {
        "enabled": bool(vnorm_on),
        "remaining_pnl_mean": pnl_m,
        "remaining_pnl_std": pnl_s,
        "remaining_bars_mean": bars_m,
        "remaining_bars_std": bars_s,
    }
    xpm = (ctx.X[m_idx][:, market_idx] if market_idx else np.zeros((ex_i.size, 0), np.float32)).astype(np.float32, copy=False)
    xpr = (ctx.X[m_idx][:, regime_idx] if regime_idx else np.zeros((ex_i.size, 0), np.float32)).astype(np.float32, copy=False)
    xpp = pos_x[ex_i].astype(np.float32, copy=False)

    opt2 = torch.optim.AdamW(net.parameters(), lr=p2_lr, weight_decay=env_float("L2_UNIFIED_WD", 1e-4))
    if optimizer_phase2_state is not None:
        try:
            opt2.load_state_dict(optimizer_phase2_state)
        except Exception as e:  # noqa: BLE001
            print(f"  [L2 unified][warn] could not load Phase2 optimizer state: {e}", flush=True)
    p2_start = max(0, int(resume_phase2_completed))
    if p2_start >= p2_ep:
        print(
            f"  [L2 unified] Phase2 skipped: checkpoint says phase2 already complete ({p2_start}>={p2_ep}).",
            flush=True,
        )
        return {
            "l2_unified_value_target_norm": l2_unified_value_target_norm,
            "policy_build_cache": policy_tup,
        }
    n_ex = ex_i.size
    n_en = tr_idx.size
    if n_en < 1:
        print("  [L2 unified] Phase2 skipped: empty L2 train index.", flush=True)
        return {
            "l2_unified_value_target_norm": l2_unified_value_target_norm,
            "policy_build_cache": policy_tup,
        }

    t_y_ex = torch.as_tensor(y_ex, device=device, dtype=torch.float32)
    t_y_va = torch.as_tensor(y_va, device=device, dtype=torch.float32)
    t_y_b = torch.as_tensor(y_ba, device=device, dtype=torch.float32)
    t_y_va_n = (t_y_va - pnl_m) / pnl_s
    t_y_b_n = (t_y_b - bars_m) / bars_s
    t_u = torch.as_tensor(u_t, device=device, dtype=torch.float32)
    t_xpm = torch.as_tensor(xpm, device=device, dtype=torch.float32)
    t_xpr = torch.as_tensor(xpr, device=device, dtype=torch.float32)
    t_xpp = torch.as_tensor(xpp, device=device, dtype=torch.float32)
    t_tr = tr_idx
    t_ex = np.arange(n_ex, dtype=np.int64)
    p2_scaler: torch.amp.GradScaler | None = (
        torch.amp.GradScaler("cuda", enabled=True)
        if (device.type == "cuda" and _unified_amp_enabled(device))
        else None
    )

    print(
        f"  [L2 unified] Phase2 joint: epochs={p2_ep}  policy_rows={n_ex}  L2_train_rows={n_en}  "
        f"lr={p2_lr}  w_entry/exit/val={w_e}/{w_ex}/{w_v}  value_norm={l2_unified_value_target_norm.get('enabled')}  "
        f"exit_label_pos_rate={exit_pos_rate:.4f}  exit_bce_pos_weight={p2_exit_bce_pw:.4f}  "
        f"(L2_UNIFIED_P2_EXIT_POS_WEIGHT: default/1=off; auto|lgbm=LGBM scale_pos_weight rule; else float)",
        flush=True,
    )
    ne_fix = max(1, min(n_en, int(batch_sz * p2_entry_ratio)))
    bf_gate = t_tr[:ne_fix]
    bf_gate_t = torch.as_tensor(bf_gate, device=device, dtype=torch.long)
    z_fix = torch.zeros((ne_fix, n_pos), device=device, dtype=torch.float32)
    net.eval()
    with torch.inference_mode():
        o_gate0 = net(xm[bf_gate_t], xr[bf_gate_t], z_fix, return_all=True)
        gate_p_before = float(torch.sigmoid(o_gate0["gate_logit"]).mean().item())
    net.train()
    p2bar = tqdm(range(p2_start, p2_ep), desc="[L2 unified] phase2", file=_tqdm_stream(), leave=True, initial=p2_start, total=p2_ep)
    for p2_i in p2bar:
        net.train()
        ne = max(1, int(batch_sz * p2_entry_ratio))
        nx = max(1, batch_sz - ne)
        b_en = t_tr[rng.integers(0, n_en, size=ne, endpoint=False)]
        b_xi = t_ex[rng.integers(0, n_ex, size=nx, endpoint=False)]
        b_en_t = torch.as_tensor(b_en, device=device, dtype=torch.long)
        b_xi_t = torch.as_tensor(b_xi, device=device, dtype=torch.long)

        bxm_ = torch.cat(
            [xm[b_en_t], t_xpm[b_xi_t]],
            dim=0,
        )
        bxr_ = torch.cat(
            [xr[b_en_t], t_xpr[b_xi_t]],
            dim=0,
        )
        zpos = torch.zeros((ne, n_pos), device=device, dtype=torch.float32)
        bxp_ = torch.cat([zpos, t_xpp[b_xi_t]], dim=0)

        with _unified_autocast_cm(device):
            out = net(bxm_, bxr_, bxp_, return_all=True)
            n_ev = int(ne)
            lg = bce(out["gate_logit"][:n_ev], gate_y[b_en_t]).mean()
            lr_ = huber(out["range"][:n_ev], y_range_fit_t[b_en_t]).mean()
            den_ = (mfe_m[b_en_t] * mfe_w[b_en_t]).sum().clamp_min(1e-6)
            lmf_ = (huber(out["mfe"][:n_ev], y_mfe_fit_t[b_en_t]) * mfe_w[b_en_t] * mfe_m[b_en_t]).sum() / den_
            lma_ = huber(out["mae"][:n_ev], y_mae_fit_t[b_en_t]).mean()
            l_ent = lg + 0.8 * lr_ + 0.3 * lmf_ + 0.3 * lma_
            l_mh10 = l_mh20 = l_mhtt = None
            if mh_on and y_r10_t is not None and y_r20_t is not None and y_ttp_t is not None and "range_10" in out:
                l_mh10 = huber(out["range_10"][:n_ev], y_r10_t[b_en_t]).mean()
                l_mh20 = huber(out["range_20"][:n_ev], y_r20_t[b_en_t]).mean()
                l_mhtt = huber(out["ttp90"][:n_ev], y_ttp_t[b_en_t]).mean()
                l_ent = l_ent + 0.2 * l_mh10 + 0.2 * l_mh20 + 0.1 * l_mhtt
            _bce_exit_kw: dict[str, Any] = {"reduction": "mean"}
            if p2_exit_bce_pw != 1.0:
                _bce_exit_kw["pos_weight"] = torch.as_tensor(
                    p2_exit_bce_pw, device=device, dtype=torch.float32
                )
            l_cl = F.binary_cross_entropy_with_logits(
                out["exit_close_logit"][n_ev:],
                t_y_ex[b_xi_t],
                **_bce_exit_kw,
            )
            l_ur = F.mse_loss(
                torch.sigmoid(out["exit_urgency_logit"][n_ev:]),
                t_u[b_xi_t],
            )
            l_val = F.smooth_l1_loss(
                out["value_remaining_pnl"][n_ev:],
                t_y_va_n[b_xi_t],
                beta=1.0,
                reduction="mean",
            )
            l_ba = (
                F.huber_loss(
                    out["value_remaining_bars"][n_ev:],
                    t_y_b_n[b_xi_t],
                    reduction="mean",
                    delta=5.0,
                )
                if use_bars
                else l_val * 0.0
            )
            l_pq = (
                _pinball_loss_quantiles(
                    out["value_pnl_quantiles"][n_ev:],
                    t_y_va_n[b_xi_t],
                    q_levels,
                )
                if use_q
                else l_val * 0.0
            )
            l_val_g = l_val + (p2_w_bars * l_ba if use_bars else 0.0) + (p2_w_q * l_pq if use_q else 0.0)
            l_tot = w_e * l_ent + w_ex * (l_cl + 0.3 * l_ur) + w_v * l_val_g
        opt2.zero_grad(set_to_none=True)
        if p2_scaler is not None:
            p2_scaler.scale(l_tot).backward()
            p2_scaler.unscale_(opt2)
            torch.nn.utils.clip_grad_norm_(net.parameters(), _unified_grad_clip_max())
            p2_scaler.step(opt2)
            p2_scaler.update()
        else:
            l_tot.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), _unified_grad_clip_max())
            opt2.step()
        p2bar.set_postfix(loss=float(l_tot.item()), refresh=True)
        p2_comp: dict[str, float] = {
            "entry_block": float(l_ent.item()),
            "gate_bce": float(lg.item()),
            "range_huber": float(lr_.item()),
            "mfe_huber": float(lmf_.item()),
            "mae_huber": float(lma_.item()),
            "exit_bce": float(l_cl.item()),
            "exit_urgency_mse": float(l_ur.item()),
            "value_pnl_smoothl1": float(l_val.item()),
            "value_bars_huber": float(l_ba.item()) if use_bars else 0.0,
            "value_quantile_pinball": float(l_pq.item()) if use_q else 0.0,
            "total": float(l_tot.item()),
        }
        if l_mh10 is not None and l_mh20 is not None and l_mhtt is not None:
            p2_comp["range_10_huber"] = float(l_mh10.item())
            p2_comp["range_20_huber"] = float(l_mh20.item())
            p2_comp["ttp90_huber"] = float(l_mhtt.item())
        print(
            f"  [L2 unified] Phase2 epoch {p2_i + 1}/{p2_ep} loss_components: {p2_comp}",
            flush=True,
        )
        if (
            checkpoint_path
            and _unified_checkpoint_writes_enabled()
            and _unified_should_write_phase_checkpoint(p2_i + 1, p2_ep)
            and opt_p1 is not None
            and rng_for_ckpt is not None
        ):
            _write_unified_training_checkpoint(
                checkpoint_path,
                net=net,
                optimizer_p1=opt_p1,
                optimizer_p2=opt2,
                rng=rng_for_ckpt,
                phase1_completed=p1_epochs_total,
                phase2_completed=p2_i + 1,
            )
    net.eval()
    with torch.inference_mode():
        o_gate1 = net(xm[bf_gate_t], xr[bf_gate_t], z_fix, return_all=True)
        gate_p_after = float(torch.sigmoid(o_gate1["gate_logit"]).mean().item())
    print(
        f"  [L2 unified] Phase2 fixed-batch gate mean(sigmoid): before={gate_p_before:.4f}  "
        f"after={gate_p_after:.4f}  delta={gate_p_after - gate_p_before:+.4f}  (rows={ne_fix})",
        flush=True,
    )
    print("  [L2 unified] Phase2 done.", flush=True)
    return {
        "l2_unified_value_target_norm": l2_unified_value_target_norm,
        "policy_build_cache": policy_tup,
        "phase2_exit_label_pos_rate": exit_pos_rate,
        "phase2_exit_bce_pos_weight": p2_exit_bce_pw,
        "phase2_gate_entry_mean_prob": {
            "before": gate_p_before,
            "after": gate_p_after,
            "batch_rows": int(ne_fix),
        },
    }


def train_l2_pytorch_unified(ctx: L2PytorchContext) -> Any:
    from core.training.l2.train import (  # noqa: WPS433
        L2TrainingBundle,
        L2_STRADDLE_VOL_REGIME_NAMES,
        _apply_binary_calibrator,
        _l2_straddle_size,
        _l2_straddle_expected_edge,
        _l2_straddle_predicted_profit,
        _l2_apply_expected_edge_regime_blacklist,
        _l2_implied_proxy_range_atr,
        _l2_range_horizon_bars,
        _l2_iv_proxy_rv_lookback,
        _l2_iv_proxy_vixy_z_beta,
        _l2_rv_to_iv_mult,
        _l2_implied_range_sigma_mult,
        _l2_vol_long_edge_thr,
        _l2_vol_short_edge_thr,
        _l2_threshold_row_from_state_map,
        _l2_fit_vol_quantiles_vol_forecast,
        _l2_vol_regime_ids,
        _l2_vol_regime_names_from_ids,
        l2_l3_synthetic_decision_class,
    )
    from core.training.common.lgbm_utils import _tqdm_stream

    device = _tensor_device()
    _ = L2_STRADDLE_VOL_REGIME_NAMES
    print(
        f"  [L2 unified] training on {device}  (set L2_LEGACY_LGBM=1 for LightGBM)  workers≈{_l2_n_jobs_silent()}",
        flush=True,
    )

    market_idx, regime_idx = split_feature_indices(ctx.feature_cols)
    n_m, n_r = len(market_idx), len(regime_idx)
    n_pos = int(os.environ.get("L2_UNIFIED_POSITION_DIM", "8") or 8)
    ucfg = UnifiedL2L3Config(position_dim=n_pos)
    net = UnifiedL2L3Net(
        n_market=n_m,
        n_regime=n_r,
        n_position=n_pos,
        cfg=ucfg,
        multi_horizon=bool(ctx.mh_on),
    ).to(device)
    if n_m < 1 or n_r < 1:
        raise RuntimeError("L2 unified: need at least one 'market' and one l1a_* (regime) column.")

    ckpt_path = _unified_checkpoint_path()
    resume: dict[str, Any] | None = None
    if _unified_resume_enabled():
        resume = _load_unified_training_checkpoint(ckpt_path)
        if resume is not None:
            print(
                f"  [L2 unified] resume: loaded {ckpt_path}  "
                f"p1_done={int(resume.get('phase1_completed', 0))}  p2_done={int(resume.get('phase2_completed', 0))}",
                flush=True,
            )
            try:
                net.load_state_dict(resume["model"])
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"L2 unified: checkpoint model incompatible with this network: {e}") from e

    net = _maybe_compile_unified_net(net)

    tr = np.asarray(ctx.fit_train_mask, dtype=bool)
    tr_idx = np.flatnonzero(tr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    huber = nn.HuberLoss(reduction="none", delta=1.0)
    opt = torch.optim.AdamW(net.parameters(), lr=env_float("L2_UNIFIED_LR", 0.001), weight_decay=env_float("L2_UNIFIED_WD", 1e-4))
    if resume is not None and resume.get("optimizer_phase1") is not None:
        try:
            opt.load_state_dict(resume["optimizer_phase1"])
        except Exception as e:  # noqa: BLE001
            print(f"  [L2 unified][warn] could not load Phase1 optimizer: {e}", flush=True)
    epochs = env_int("L2_UNIFIED_EPOCHS", 32, lo=1)
    batch_sz = env_int("L2_UNIFIED_BATCH", 16384, lo=1)

    rng = (
        _restore_rng_from_checkpoint(resume["rng_state"])
        if resume is not None and resume.get("rng_state") is not None
        else np.random.default_rng(2026)
    )

    xm, xr, xp = build_column_tensors(ctx.X, market_idx, regime_idx, n_pos, device)
    gate_y = torch.as_tensor(ctx.y_trade, dtype=torch.float32, device=device)
    y_range_fit_t = torch.as_tensor(ctx.y_range_fit, dtype=torch.float32, device=device)
    y_mfe_fit_t = torch.as_tensor(ctx.y_mfe_fit, dtype=torch.float32, device=device)
    y_mae_fit_t = torch.as_tensor(ctx.y_mae_fit, dtype=torch.float32, device=device)
    mfe_w = torch.as_tensor(ctx.mfe_w_all, dtype=torch.float32, device=device)
    y_r10_t = torch.as_tensor(ctx.y_r10_fit, device=device, dtype=torch.float32) if ctx.mh_on and ctx.y_r10_fit is not None else None
    y_r20_t = torch.as_tensor(ctx.y_r20_fit, device=device, dtype=torch.float32) if ctx.mh_on and ctx.y_r20_fit is not None else None
    y_ttp_t = torch.as_tensor(ctx.y_ttp_fit, device=device, dtype=torch.float32) if ctx.mh_on and ctx.y_ttp_fit is not None else None
    tr_t = torch.as_tensor(ctx.fit_train_mask.astype(np.float32), device=device)
    mfe_m = (gate_y > 0.5) * tr_t
    p1_scaler: torch.amp.GradScaler | None = (
        torch.amp.GradScaler("cuda", enabled=True)
        if (device.type == "cuda" and _unified_amp_enabled(device))
        else None
    )
    p1_start = int(resume.get("phase1_completed", 0)) if resume is not None else 0
    if p1_start < 0:
        p1_start = 0
    if p1_start > epochs:
        print("  [L2 unified][warn] checkpoint phase1_completed > epochs; clamping.", flush=True)
        p1_start = epochs
    if p1_start >= epochs and _unified_resume_enabled():
        print("  [L2 unified] skipping Phase1 (checkpoint already complete).", flush=True)
    pbar = tqdm(
        range(p1_start, epochs),
        desc="[L2 unified] epochs",
        file=_tqdm_stream(),
        leave=True,
        total=epochs,
        initial=p1_start,
    )
    for ep in pbar:
        net.train()
        rng.shuffle(tr_idx)
        eloss: list[float] = []
        sum_lg = sum_lr = sum_lmf = sum_lma = 0.0
        sum_mh10 = sum_mh20 = sum_mhtt = 0.0
        n_b = 0
        for s in range(0, tr_idx.size, batch_sz):
            b = tr_idx[s : s + batch_sz]
            bxm, bxr, bxp = xm[b], xr[b], xp[b]
            with _unified_autocast_cm(device):
                out = net(bxm, bxr, bxp)
                lg = bce(out["gate_logit"], gate_y[b]).mean()
                lr = huber(out["range"], y_range_fit_t[b]).mean()
                den = (mfe_m[b] * mfe_w[b]).sum().clamp_min(1e-6)
                lmf = (huber(out["mfe"], y_mfe_fit_t[b]) * mfe_w[b] * mfe_m[b]).sum() / den
                lma = huber(out["mae"], y_mae_fit_t[b]).mean()
                ltot = lg + 0.8 * lr + 0.3 * lmf + 0.3 * lma
                if ctx.mh_on and y_r10_t is not None and "range_10" in out:
                    h10 = huber(out["range_10"], y_r10_t[b]).mean()
                    h20 = huber(out["range_20"], y_r20_t[b]).mean()  # type: ignore[index]
                    htt = huber(out["ttp90"], y_ttp_t[b]).mean()  # type: ignore[index]
                    ltot = ltot + 0.2 * h10 + 0.2 * h20 + 0.1 * htt
                    sum_mh10 += float(h10.item())
                    sum_mh20 += float(h20.item())
                    sum_mhtt += float(htt.item())
            opt.zero_grad(set_to_none=True)
            if p1_scaler is not None:
                p1_scaler.scale(ltot).backward()
                p1_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), _unified_grad_clip_max())
                p1_scaler.step(opt)
                p1_scaler.update()
            else:
                ltot.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), _unified_grad_clip_max())
                opt.step()
            eloss.append(float(ltot.item()))
            sum_lg += float(lg.item())
            sum_lr += float(lr.item())
            sum_lmf += float(lmf.item())
            sum_lma += float(lma.item())
            n_b += 1
        pbar.set_postfix(loss=float(np.mean(eloss) if eloss else 0.0), refresh=True)
        if n_b > 0:
            comp: dict[str, float] = {
                "gate_bce": sum_lg / n_b,
                "range_huber": sum_lr / n_b,
                "mfe_huber": sum_lmf / n_b,
                "mae_huber": sum_lma / n_b,
                "total_weighted_mean": float(np.mean(eloss) if eloss else 0.0),
            }
            if ctx.mh_on and y_r10_t is not None:
                comp["range_10_huber"] = sum_mh10 / n_b
                comp["range_20_huber"] = sum_mh20 / n_b
                comp["ttp90_huber"] = sum_mhtt / n_b
            print(f"  [L2 unified] epoch {ep + 1}/{epochs} loss_components: {comp}", flush=True)
        if _unified_checkpoint_writes_enabled() and _unified_should_write_phase_checkpoint(ep + 1, epochs):
            _write_unified_training_checkpoint(
                ckpt_path,
                net=net,
                optimizer_p1=opt,
                optimizer_p2=None,
                rng=rng,
                phase1_completed=ep + 1,
                phase2_completed=0,
            )

    p2_wanted = (os.environ.get("L2_UNIFIED_PHASE2", "") or "").strip().lower() in ("1", "true", "yes", "on")
    n_all = int(ctx.X.shape[0])

    def _forward_once() -> tuple[dict[str, np.ndarray], Any, pd.DataFrame]:
        net.eval()
        raw_chunks2: list[dict[str, np.ndarray]] = []
        with torch.inference_mode():
            for s2 in range(0, n_all, batch_sz):
                e2 = min(n_all, s2 + batch_sz)
                bxm2, bxr2, bxp2 = xm[s2:e2], xr[s2:e2], xp[s2:e2]
                o2 = net(bxm2, bxr2, bxp2, return_all=False)
                raw_chunks2.append({k: v.detach().float().cpu().numpy() for k, v in o2.items()})
        assert raw_chunks2
        raw_out = {k: np.concatenate([c[k] for c in raw_chunks2]) for k in raw_chunks2[0]}

        gate_logits2 = raw_out["gate_logit"]
        gate_cal2 = _fit_gate_calibrator(ctx.val_tune_mask, ctx.y_trade, gate_logits2)
        trade_p = _apply_binary_calibrator(
            (1.0 / (1.0 + np.exp(-np.clip(gate_logits2, -20, 20)))).astype(np.float64),
            gate_cal2,
        ).astype(np.float32)
        range_pred = _regression_pred_inverse(raw_out["range"], ctx.range_head_prep, clip_max=10.0)
        mfe_pred = _regression_pred_inverse(raw_out["mfe"], ctx.mfe_head_prep, clip_max=5.0)
        mae_pred = _regression_pred_inverse(raw_out["mae"], ctx.mae_head_prep, clip_max=4.0)
        z0 = np.zeros(n_all, dtype=np.float32)
        if ctx.mh_on and ctx.r10_head_prep and ctx.r20_head_prep and ctx.ttp_head_prep:
            range10_pred = _regression_pred_inverse(raw_out.get("range_10", z0), ctx.r10_head_prep, clip_max=15.0)
            range20_pred = _regression_pred_inverse(raw_out.get("range_20", z0), ctx.r20_head_prep, clip_max=15.0)
            ttp90_pred = _regression_pred_inverse(raw_out.get("ttp90", z0), ctx.ttp_head_prep, clip_max=1.0)
        else:
            range10_pred = range20_pred = ttp90_pred = z0

        vol_cfg: dict = {}
        hold_bars = int(ctx.straddle_label_stats.get("range_horizon_bars", _l2_range_horizon_bars()))
        implied_proxy_range = _l2_implied_proxy_range_atr(
            ctx.df,
            ctx.frame,
            hold_bars=hold_bars,
            rv_lookback=_l2_iv_proxy_rv_lookback(),
            vixy_z_beta=_l2_iv_proxy_vixy_z_beta(),
            vrp_base_mult=_l2_rv_to_iv_mult(),
            implied_range_sigma_mult=_l2_implied_range_sigma_mult(),
        )
        long_edge_thr = float(_l2_vol_long_edge_thr())
        short_edge_thr = float(_l2_vol_short_edge_thr())
        predicted_profit = _l2_straddle_predicted_profit(range_pred, cost_atr=implied_proxy_range)
        predicted_profit = _l2_apply_expected_edge_regime_blacklist(np.asarray(predicted_profit, np.float32), ctx.frame).astype(
            np.float32
        )
        vol_edge = (
            (np.asarray(range_pred, np.float64) - np.asarray(implied_proxy_range, np.float64))
            / np.maximum(np.asarray(implied_proxy_range, np.float64), 1e-6)
        ).astype(np.float32)
        expected_edge = _l2_straddle_expected_edge(trade_p, range_pred, cost_atr=implied_proxy_range)
        expected_edge = _l2_apply_expected_edge_regime_blacklist(np.asarray(expected_edge, np.float32), ctx.frame).astype(
            np.float32
        )
        vol_all_i = ctx.frame["l1a_vol_forecast"].to_numpy(dtype=np.float64)
        trend_all_i = (
            ctx.frame["l1a_vol_trend"].to_numpy(dtype=np.float64)
            if "l1a_vol_trend" in ctx.frame.columns
            else np.zeros(n_all, dtype=np.float64)
        )
        trend_eps_m = float(
            np.clip(float(os.environ.get("L2_REGIME_TREND_EPS", "0.02") or 0.02), 1e-6, 1.0)
        )
        regime_adjust_m = bool(os.environ.get("L2_REGIME_STRADDLE_ADJUST", "0").strip().lower() in ("1", "true", "yes"))
        trade_thr = float(os.environ.get("L2_TRADE_THR", "0.5") or 0.5)
        q33_m, q66_m = 0.33, 0.66
        vol_q: dict[str, float] = {}
        if regime_adjust_m:
            vol_q33, vol_q66 = _l2_fit_vol_quantiles_vol_forecast(vol_all_i, np.asarray(ctx.val_tune_mask, dtype=bool))
            q33_m, q66_m = float(vol_q33), float(vol_q66)
            vol_q = {"q33": q33_m, "q66": q66_m}
        reg_ids_i = _l2_vol_regime_ids(vol_all_i, trend_all_i, q33_m, q66_m, trend_eps_m)
        dyn_state_keys = _l2_vol_regime_names_from_ids(reg_ids_i)
        trade_thr_row = _l2_threshold_row_from_state_map(
            trade_thr,
            np.asarray(dyn_state_keys, dtype=object).ravel(),
            "",
        )
        trade_thr_eval: float | np.ndarray = trade_thr_row if trade_thr_row is not None else trade_thr
        from core.training.l2.train import _l2_straddle_size, _l2_vol_signal_from_edge

        vol_signal = _l2_vol_signal_from_edge(
            trade_p,
            trade_thr_eval,
            vol_edge,
            long_thr=long_edge_thr,
            short_thr=short_edge_thr,
        )
        straddle_on = vol_signal != 0
        size_mult_i = np.ones(n_all, dtype=np.float32)
        vol_regime_i = np.full(n_all, "", dtype=object)
        vol_regime_score_i = np.zeros(n_all, dtype=np.float32)
        if regime_adjust_m:
            from core.training.l2.train import _l2_regime_straddle_config_merged
            from core.training.l2.train import _l2_vol_regime_names_from_ids as _nids

            cfg_inf = _l2_regime_straddle_config_merged()
            mult_sm_i = np.array([float(cfg_inf[k]["size_mult"]) for k in L2_STRADDLE_VOL_REGIME_NAMES], dtype=np.float64)
            base_sz = _l2_straddle_size(trade_p, np.abs(vol_edge.astype(np.float32)), straddle_on)
            size_mult_i = mult_sm_i[reg_ids_i].astype(np.float32)
            size_pred = (base_sz * size_mult_i).astype(np.float32)
            vol_regime_i = _nids(reg_ids_i)
            vol_regime_score_i = (reg_ids_i.astype(np.float32) / 4.0).astype(np.float32)
        else:
            size_pred = _l2_straddle_size(
                trade_p,
                np.abs(np.asarray(vol_edge, dtype=np.float32)),
                straddle_on,
            )
        decision_confidence = trade_p.astype(np.float32)
        out_df = ctx.df[["symbol", "time_key"]].copy()
        out_df["l2_straddle_on"] = straddle_on.astype(np.int32)
        out_df["l2_vol_signal"] = vol_signal.astype(np.int32)
        out_df["l2_gate_prob"] = trade_p.astype(np.float32)
        out_df["l2_decision_confidence"] = decision_confidence
        out_df["l2_range_pred"] = range_pred.astype(np.float32)
        out_df["l2_realized_pred_range"] = range_pred.astype(np.float32)
        out_df["l2_implied_proxy_range"] = np.asarray(implied_proxy_range, dtype=np.float32)
        out_df["l2_vol_edge"] = vol_edge.astype(np.float32)
        out_df["l2_range_pred_10"] = range10_pred
        out_df["l2_range_pred_20"] = range20_pred
        out_df["l2_pred_ttp90_norm"] = ttp90_pred
        out_df["l2_predicted_profit"] = predicted_profit.astype(np.float32)
        out_df["l2_size"] = size_pred
        out_df["l2_pred_mfe"] = mfe_pred
        out_df["l2_pred_mae"] = mae_pred
        entry_regime = ctx.frame[L1A_REGIME_COLS].to_numpy(dtype=np.float32, copy=False)
        for idx in range(NUM_REGIME_CLASSES):
            out_df[f"l2_entry_regime_{idx}"] = entry_regime[:, idx]
        out_df["l2_entry_vol"] = ctx.frame["l1a_vol_forecast"].to_numpy(dtype=np.float32, copy=False)
        out_df["l2_vol_regime"] = vol_regime_i.astype(str)
        out_df["l2_vol_regime_score"] = vol_regime_score_i.astype(np.float32)
        out_df["l2_regime_size_mult"] = size_mult_i
        out_df["l2_expected_edge"] = expected_edge.astype(np.float32)
        out_df["l2_rr_proxy"] = out_df["l2_pred_mfe"] / np.maximum(out_df["l2_pred_mae"], 0.05)
        edge_l1b = (
            ctx.frame["l1b_edge_pred"].to_numpy(dtype=np.float32, copy=False)
            if "l1b_edge_pred" in ctx.frame.columns
            else np.zeros(n_all, dtype=np.float32)
        )
        out_df["l2_decision_class"] = l2_l3_synthetic_decision_class(
            vol_signal.astype(np.float32), edge_l1b.astype(np.float64), trade_p.astype(np.float64)
        )
        return {
            "raw_all": raw_out,
            "gate_cal": gate_cal2,
            "outputs": out_df,
            "hold_bars": hold_bars,
            "trade_thr": trade_thr,
            "long_edge_thr": long_edge_thr,
            "short_edge_thr": short_edge_thr,
            "vol_cfg": vol_cfg,
            "trend_eps_m": trend_eps_m,
            "regime_adjust_m": regime_adjust_m,
            "vol_q": vol_q,
        }

    p2_res_done = 0
    p2_res_opt: dict[str, Any] | None = None
    if resume is not None and int(resume.get("phase1_completed", 0)) >= epochs:
        p2_res_done = int(resume.get("phase2_completed", 0))
        p2_res_opt = resume.get("optimizer_phase2")
        if not isinstance(p2_res_opt, dict):
            p2_res_opt = None

    bundle = _forward_once()
    phase2_extras: dict[str, Any] = {}
    if p2_wanted:
        p2p = _unified_l2_phase2_joint(
            net,
            ctx,
            bundle["outputs"],
            market_idx,
            regime_idx,
            n_pos,
            device,
            xm,
            xr,
            tr_idx,
            gate_y,
            y_range_fit_t,
            y_mfe_fit_t,
            y_mae_fit_t,
            mfe_w,
            mfe_m,
            y_r10_t,
            y_r20_t,
            y_ttp_t,
            bool(ctx.mh_on),
            batch_sz,
            rng,
            bce,
            huber,
            resume_phase2_completed=p2_res_done,
            optimizer_phase2_state=p2_res_opt,
            checkpoint_path=ckpt_path if _unified_checkpoint_writes_enabled() else None,
            p1_epochs_total=epochs,
            opt_p1=opt,
            rng_for_ckpt=rng,
        )
        phase2_extras = p2p if isinstance(p2p, dict) else {}
        bundle = _forward_once()

    raw_all = bundle["raw_all"]
    gate_cal = bundle["gate_cal"]
    outputs = bundle["outputs"]
    hold_bars = bundle["hold_bars"]
    trade_thr = float(bundle["trade_thr"])
    long_edge_thr = float(bundle["long_edge_thr"])
    short_edge_thr = float(bundle["short_edge_thr"])
    vol_cfg = bundle["vol_cfg"]
    trend_eps_m = float(bundle["trend_eps_m"])
    regime_adjust_m = bool(bundle["regime_adjust_m"])
    vol_q = bundle["vol_q"]

    os.makedirs(MODEL_DIR, exist_ok=True)
    unified_name = L2_UNIFIED_MODEL_FILE
    u_path = os.path.join(MODEL_DIR, unified_name)
    torch.save(net.state_dict(), u_path)
    if gate_cal is not None:
        with open(os.path.join(MODEL_DIR, "l2_trade_gate_calibrator.pkl"), "wb") as f:
            pickle.dump(gate_cal, f)

    model_files = {
        "unified_l2l3": unified_name,
    }
    if gate_cal is not None:
        model_files["trade_gate_calibrator"] = "l2_trade_gate_calibrator.pkl"

    from core.training.unified.position_features import build_position_matrix as _bpm_dim

    _, _pos_names_meta = _bpm_dim(
        np.zeros((1, len(ctx.feature_cols)), dtype=np.float32), list(ctx.feature_cols), n_pos
    )
    dim_check = _l2_unified_dim_check(
        list(ctx.feature_cols), market_idx, regime_idx, n_m, n_r, n_pos, list(_pos_names_meta)
    )
    l2_unified_train_input_stats = _l2_unified_train_input_stats_numpy(
        ctx.X, ctx.fit_train_mask, market_idx, regime_idx, list(ctx.feature_cols), n_pos
    )
    head_weight_norms = _unified_head_weight_norms(net)
    _vtn = phase2_extras.get("l2_unified_value_target_norm") if p2_wanted else None
    _vtn = _vtn if isinstance(_vtn, dict) else None
    _pol_cache = phase2_extras.get("policy_build_cache") if p2_wanted else None
    if not (isinstance(_pol_cache, tuple) and len(_pol_cache) == 10):
        _pol_cache = None
    try:
        calibration_summary = _l2_unified_calibration_summary(
            net,
            ctx,
            outputs,
            market_idx,
            regime_idx,
            n_pos,
            device,
            tr_idx,
            xm,
            xr,
            batch_sz,
            p2_wanted,
            rng,
            value_target_norm=_vtn,
            policy_build_cache=_pol_cache,
        )
    except Exception as e:  # noqa: BLE001
        calibration_summary = {"error": str(e)}
    print("  [L2 unified] calibration_summary stored in meta (key calibration_summary).", flush=True)

    meta: dict[str, Any] = {
        "schema_version": L2_SCHEMA_VERSION,
        "backend": "pytorch_unified",
        "unified_model_file": unified_name,
        "has_exit_value_heads": bool(p2_wanted),
        "l3_unified_exit": bool(p2_wanted),
        "decision_mode": "straddle",
        "dim_check": dim_check,
        "head_weight_norms": head_weight_norms,
        "calibration_summary": calibration_summary,
        "l2_unified_value_target_norm": _vtn,
        "phase2_gate_entry_mean_prob": (
            phase2_extras.get("phase2_gate_entry_mean_prob") if p2_wanted else None
        ),
        "phase2_exit_label_pos_rate": phase2_extras.get("phase2_exit_label_pos_rate") if p2_wanted else None,
        "phase2_exit_bce_pos_weight": phase2_extras.get("phase2_exit_bce_pos_weight") if p2_wanted else None,
        "l2_unified_train_input_stats": l2_unified_train_input_stats,
        "feature_cols": list(ctx.feature_cols),
        "range_feature_cols": list(ctx.range_feature_cols),
        "l2_derived_feature_stats": ctx.derived_feature_stats,
        "l2_aux_head_target_prep": {
            "range": ctx.range_head_prep,
            "mfe": ctx.mfe_head_prep,
            "mae": ctx.mae_head_prep,
            **(
                {
                    "range_10": ctx.r10_head_prep,
                    "range_20": ctx.r20_head_prep,
                    "ttp90": ctx.ttp_head_prep,
                }
                if ctx.mh_on and ctx.r10_head_prep
                else {}
            ),
        },
        "trade_threshold": float(trade_thr),
        "range_horizon_bars": hold_bars,
        "l2_vol_only_config": vol_cfg,
        "l2_iv_proxy_rv_lookback": _l2_iv_proxy_rv_lookback(),
        "l2_iv_proxy_vixy_z_beta": _l2_iv_proxy_vixy_z_beta(),
        "l2_rv_to_iv_mult": _l2_rv_to_iv_mult(),
        "l2_implied_range_sigma_mult": _l2_implied_range_sigma_mult(),
        "l2_vol_long_edge_thr": long_edge_thr,
        "l2_vol_short_edge_thr": short_edge_thr,
        "l2_regime_trend_eps": trend_eps_m,
        "l2_regime_straddle_adjust": bool(regime_adjust_m),
        "l2_vol_regime_quantiles": vol_q,
        "policy_search": {
            "trade_threshold": float(trade_thr),
            "dynamic_threshold_map": "",
        },
        "straddle_label_stats": ctx.straddle_label_stats,
        "l2_unified_config": {
            "n_market": n_m,
            "n_regime": n_r,
            "n_position": n_pos,
            "position_dim": n_pos,
            "market_idx": market_idx,
            "regime_idx": regime_idx,
            "group_dim": ucfg.group_dim,
            "backbone_dim": ucfg.backbone_dim,
            "n_backbone_layers": ucfg.n_backbone_layers,
            "head_dim": ucfg.head_dim,
            "head_dropout": ucfg.head_dropout,
            "multi_horizon": bool(ctx.mh_on),
            "has_exit_value_heads": bool(p2_wanted),
        },
        "l2_unified_code_version": "0.1",
        "l2_dropped_features": list(ctx.l2_dropped_features),
        "l2_val_tune_frac": float(ctx.tune_frac),
        "l2_min_feature_std": float(ctx.min_std),
        "l2_feature_hard_drop_skipped": bool(ctx.skip_hard),
        "l2_feature_selection_dropped": list(ctx.l2_dropped_features),
        "l2_train_boost_rounds": int(env_int("L2_UNIFIED_EPOCHS", 32, lo=1)),
        "l2_train_backend": "pytorch_unified",
        "output_cache_file": L2_OUTPUT_CACHE_FILE,
        "model_files": model_files,
    }
    meta = attach_threshold_registry(
        meta,
        "l2",
        [
            threshold_entry(
                "trade_threshold",
                float(trade_thr),
                category="adaptive_candidate",
                role="unified L2 default",
                adaptive_hint="set L2_TRADE_THR or search externally",
            ),
        ],
    )
    try:
        from core.training.logging.metrics_file_log import print_l2_unified_meta_readable

        _l2_mlog: dict[str, Any] = {
            "l2_decision_meta_path": os.path.join(MODEL_DIR, L2_META_FILE),
            "unified_model_path": u_path,
            "dim_check": meta.get("dim_check"),
            "head_weight_norms": meta.get("head_weight_norms"),
            "calibration_summary": meta.get("calibration_summary"),
            "l2_unified_value_target_norm": meta.get("l2_unified_value_target_norm"),
            "phase2_gate_entry_mean_prob": meta.get("phase2_gate_entry_mean_prob"),
            "phase2_exit_label_pos_rate": meta.get("phase2_exit_label_pos_rate"),
            "phase2_exit_bce_pos_weight": meta.get("phase2_exit_bce_pos_weight"),
            "l2_unified_train_input_stats": meta.get("l2_unified_train_input_stats"),
            "l2_unified_config": meta.get("l2_unified_config"),
        }
        print_l2_unified_meta_readable("L2_unified (readable)", _l2_mlog)
    except Exception as e:  # noqa: BLE001
        print(f"  [L2 unified][warn] could not print meta block: {e}", flush=True)
    with open(os.path.join(MODEL_DIR, L2_META_FILE), "wb") as f:
        pickle.dump(meta, f)
    save_output_cache(outputs, L2_OUTPUT_CACHE_FILE)
    print(f"  [L2 unified] meta saved  -> {os.path.join(MODEL_DIR, L2_META_FILE)}", flush=True)
    print(f"  [L2 unified] weights  -> {u_path}", flush=True)
    if _unified_checkpoint_writes_enabled() and not _unified_checkpoint_keep_on_success() and os.path.isfile(ckpt_path):
        try:
            os.remove(ckpt_path)
            print(f"  [L2 unified] removed training checkpoint {ckpt_path}", flush=True)
        except OSError as e:
            print(f"  [L2 unified][warn] could not remove checkpoint: {e}", flush=True)
    models_dict: dict[str, Any] = {
        "unified": net.cpu(),
    }
    if gate_cal is not None:
        models_dict["trade_gate_calibrator"] = gate_cal
    return L2TrainingBundle(
        models=models_dict,
        meta=meta,
        outputs=outputs,
    )


def _l2_n_jobs_silent() -> int:
    from core.training.common.lgbm_utils import _lgbm_n_jobs

    return int(_lgbm_n_jobs())


# Preferred name in unified package (L2 + Phase 2 in one run).
train_unified = train_l2_pytorch_unified
