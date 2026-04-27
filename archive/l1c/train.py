from __future__ import annotations

import os
import pickle
import time
from dataclasses import asdict
from datetime import datetime
from statistics import NormalDist
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange

from core.training.common.constants import (
    CAL_END,
    FAST_TRAIN_MODE,
    L1C_META_FILE,
    L1C_MODEL_FILE,
    L1C_OUTPUT_CACHE_FILE,
    L1C_SCHEMA_VERSION,
    MODEL_DIR,
)
from core.training.prep.data_prep import _create_tcn_windows
from archive.l1c.config import L1cConfig
from archive.l1c.evaluate import evaluate_l1c, print_l1c_eval_report
from archive.l1c.losses import L1cRegressionLoss
from archive.l1c.model import L1cDirectionModel
from core.training.common.lgbm_utils import _lgb_round_tqdm_enabled, _tqdm_stream
from core.training.logging.pipeline_train_logs import artifact_path, log_layer_banner
from core.training.common.stack_v2_common import (
    build_stack_time_splits,
    l1_expanding_oof_window_folds,
    l1_oof_folds_from_env,
    l1_oof_mode_from_env,
    l2_val_start_time,
    log_label_baseline,
    save_output_cache,
    split_mask_for_tuning_and_report,
    time_blocked_fold_masks,
)
from core.training.common.threshold_registry import attach_threshold_registry, threshold_entry
from core.training.tcn.tcn_constants import DEVICE
from core.utils.session import mark_session_boundaries


def l1c_output_columns() -> list[str]:
    return [
        "l1c_pred_z",
        "l1c_pred_z_abs",
        "l1c_pred_sign",
        "l1c_direction",
        "l1c_confidence",
        "l1c_direction_strength",
        "l1c_is_warm",
    ]


def _l1c_augment_batch(x: torch.Tensor, *, training: bool, seq_len: int) -> torch.Tensor:
    if not training:
        return x
    if os.environ.get("L1C_AUGMENT", "1").strip().lower() in {"0", "false", "no"}:
        return x
    out = x
    b, t, f = out.shape
    crop_prob = float(os.environ.get("L1C_AUG_CROP_PROB", "0.3"))
    if crop_prob > 0 and t >= 50 and torch.rand((), device=out.device) < crop_prob:
        start = int(torch.randint(0, max(1, t - 49), (1,), device=out.device).item())
        chunk = out[:, start : start + 50, :].transpose(1, 2)
        out = torch.nn.functional.interpolate(chunk, size=t, mode="linear", align_corners=False).transpose(1, 2)
    noise_std = float(os.environ.get("L1C_AUG_NOISE_STD", "0.01"))
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    feat_zero = float(os.environ.get("L1C_AUG_FEAT_ZERO_PROB", "0.10"))
    if feat_zero > 0 and feat_zero < 1:
        keep = (torch.rand(1, 1, f, device=out.device) > feat_zero).to(dtype=out.dtype)
        out = out * keep
    return out


def _sample_weights_from_abs_return(abs_ret: np.ndarray) -> np.ndarray:
    """Larger |return| gets larger weight (rank / N)."""
    n = int(abs_ret.size)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if os.environ.get("L1C_SAMPLE_WEIGHT_MODE", "").strip().lower() == "uniform":
        return np.ones(n, dtype=np.float32)
    abs_ret = np.asarray(abs_ret, dtype=np.float64).ravel()
    order = np.argsort(abs_ret, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    return (ranks / float(max(n, 1))).astype(np.float32)


def _robust_sigma(values: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = float(mad / 0.67448975) if mad > 0 else 0.0
    return med, mad, sigma


def _l1c_forward_dict(model: L1cDirectionModel, x: torch.Tensor) -> dict[str, torch.Tensor]:
    out = model(x)
    if isinstance(out, dict):
        return out
    pred = out
    return {
        "direction_pred": pred,
        "direction_strength": pred.new_zeros(pred.shape),
    }


def _print_l1c_regression_label_stats(y_z: np.ndarray, y_ret: np.ndarray, *, prefix: str) -> None:
    y_z = np.asarray(y_z, dtype=np.float64).ravel()
    y_ret = np.asarray(y_ret, dtype=np.float64).ravel()
    fin = np.isfinite(y_z) & np.isfinite(y_ret)
    y_z, y_ret = y_z[fin], y_ret[fin]
    if y_z.size == 0:
        print(f"  [L1c] {prefix}: no finite labels", flush=True)
        return
    ar = np.abs(y_ret)
    s = pd.Series(y_z)
    sret = pd.Series(y_ret)
    print(
        f"  [L1c] {prefix}: z mean={y_z.mean():.4f} std={y_z.std():.4f}  "
        f"spearman(z,ret)={float(s.corr(sret, method='spearman')):.4f}  "
        f"|ret| mean={ar.mean():.6f} median={np.median(ar):.6f}",
        flush=True,
    )


def _first_epoch_diagnostic(
    model: L1cDirectionModel,
    train_loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 24,
) -> None:
    model.eval()
    probs: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for b_idx, batch in enumerate(train_loader):
            x = batch[0].to(device)
            y = batch[1].numpy().ravel()
            out = _l1c_forward_dict(model, x)
            p = out["direction_pred"].view(-1).cpu().numpy()
            probs.append(p)
            trues.append(y)
            if b_idx + 1 >= max_batches:
                break
    model.train()
    pr = np.concatenate(probs)
    t = np.concatenate(trues)
    print("\n  " + "=" * 50, flush=True)
    print("  [L1c] FIRST EPOCH DIAGNOSTIC (subset of train batches)", flush=True)
    print("  " + "=" * 50, flush=True)
    print(
        f"  pred_prob — mean:{pr.mean():.4f} std:{pr.std():.4f} min:{pr.min():.4f} max:{pr.max():.4f}",
        flush=True,
    )
    print(f"  y_z — mean:{t.mean():.4f} std:{t.std():.4f}", flush=True)
    if pr.std() < 0.01:
        print("  [L1c] WARNING: pred_z std < 0.01 — outputs nearly constant", flush=True)
    acc = np.mean((pr > 0.0) == (t > 0.0))
    confident = np.abs(pr) > np.quantile(np.abs(pr), 0.8)
    print(f"  batch subset sign-accuracy: {acc:.4f}", flush=True)
    print(
        f"  batch subset top20% |pred_z| coverage: {float(np.mean(confident)):.2%}  "
        f"n={int(np.sum(confident)):,}",
        flush=True,
    )
    print("  " + "=" * 50 + "\n", flush=True)


def _l1c_scalar_flat_band(ret: np.ndarray) -> float:
    """Match _build_l1c_windows_labels flat-band (per chunk of returns)."""
    ret = np.asarray(ret, dtype=np.float64).ravel()
    ret = ret[np.isfinite(ret)]
    if ret.size == 0:
        return 0.0
    flat_mode = (os.environ.get("L1C_FLAT_MODE", "mde") or "mde").strip().lower()
    flat_q = float(np.clip(float(os.environ.get("L1C_DIRECTION_FLAT_Q", "0.20")), 0.0, 0.45))
    flat_alpha = float(np.clip(float(os.environ.get("L1C_FLAT_ALPHA", "0.05")), 1e-4, 0.20))
    flat_power = float(np.clip(float(os.environ.get("L1C_FLAT_POWER", "0.80")), 0.50, 0.99))
    abs_ret = np.abs(ret)
    if flat_mode == "quantile":
        return float(np.quantile(abs_ret, flat_q)) if abs_ret.size else 0.0
    med_ret, mad_ret, sigma_ret = _robust_sigma(ret)
    z_alpha = float(NormalDist().inv_cdf(1.0 - flat_alpha / 2.0))
    z_beta = float(NormalDist().inv_cdf(flat_power))
    return float((z_alpha + z_beta) * max(sigma_ret, 1e-8))


def _l1c_collect_rowwise_xy_for_feature_ranking(
    df: pd.DataFrame,
    cand_cols: list[str],
    train_mask: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bar-level (X[end], y) aligned with L1c direction labels, train split only."""
    use_intraday = os.environ.get("L1C_INTRADAY_LABELS", "1").strip().lower() not in {"0", "false", "no", "off"}
    d = df
    if use_intraday:
        d = df.copy()
        gap_s = float(os.environ.get("L1C_SESSION_GAP_SECONDS", str(4 * 3600)))
        open_skip = int(os.environ.get("L1C_OPEN_SKIP_BARS", "0"))
        close_cut = int(os.environ.get("L1C_CLOSE_CUTOFF_BARS", str(horizon)))
        mark_session_boundaries(
            d,
            time_col="time_key",
            symbol_col="symbol",
            overnight_gap_seconds=gap_s,
            open_skip_bars=open_skip,
            close_cutoff_bars=close_cut,
        )
    train_mask = np.asarray(train_mask, dtype=bool)
    per_sym: list[tuple[np.ndarray, np.ndarray]] = []

    for _, grp in d.groupby("symbol", sort=False):
        pos = d.index.get_indexer(grp.index)
        if (pos < 0).any():
            raise ValueError("L1c feature ranking: invalid index alignment")
        close = pd.to_numeric(grp["close"], errors="coerce").astype(np.float64).to_numpy()
        close = pd.Series(close).ffill().bfill().to_numpy(dtype=np.float64)
        n = len(grp)
        if n <= horizon:
            continue
        F = np.column_stack(
            [pd.to_numeric(grp[c], errors="coerce").to_numpy(dtype=np.float64) for c in cand_cols]
        )
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        el = np.arange(0, n - horizon, dtype=np.int32)
        ret = (close[el + horizon] - close[el]) / (np.abs(close[el]) + 1e-9)
        ok = train_mask[pos[el]]
        if use_intraday:
            sess = grp["session_id"].to_numpy(dtype=np.int32, copy=False)
            vlab = grp["valid_for_label"].to_numpy(dtype=bool, copy=False)
            ok = ok & (sess[el + horizon] == sess[el]) & vlab[el]
        if not ok.any():
            continue
        el = el[ok]
        ret = ret[ok]
        r_band = _l1c_scalar_flat_band(ret)
        y_loc = (ret > r_band).astype(np.int32, copy=False)
        X_loc = F[el].astype(np.float32, copy=False)
        per_sym.append((X_loc, y_loc))

    if not per_sym:
        return np.zeros((0, len(cand_cols)), dtype=np.float32), np.zeros(0, dtype=np.int32)
    X = np.concatenate([p[0] for p in per_sym], axis=0)
    y = np.concatenate([p[1] for p in per_sym], axis=0)
    max_rows = max(5000, int(os.environ.get("L1C_FS_MAX_ROWS", "100000")))
    if X.shape[0] > max_rows:
        rng = np.random.default_rng(int(os.environ.get("L1C_FS_SEED", "0")))
        pick = rng.choice(X.shape[0], size=max_rows, replace=False)
        X, y = X[pick], y[pick]
    return X, y


def _l1c_rank_features(
    df: pd.DataFrame,
    cand: list[str],
    cap: int,
    train_mask: np.ndarray,
    horizon: int,
) -> tuple[list[str], str]:
    mode = (os.environ.get("L1C_FEATURE_SELECT", "lgbm") or "lgbm").strip().lower()
    if mode in {"order", "head", "first"}:
        return cand[:cap], "order"
    if len(cand) <= cap:
        return cand, "all_candidates"

    X, y = _l1c_collect_rowwise_xy_for_feature_ranking(df, cand, train_mask, horizon)
    if X.shape[0] < 500 or len(np.unique(y)) < 2:
        print(
            f"  [L1c] feature select: fallback to feat_cols order "
            f"(samples={X.shape[0]} unique_y={len(np.unique(y))})",
            flush=True,
        )
        return cand[:cap], "order_fallback"

    if mode == "mi":
        mi = mutual_info_classif(X, y, random_state=0, n_neighbors=min(3, max(1, X.shape[0] // 100)))
        order = [cand[i] for i in np.argsort(-mi)]
        return order[:cap], "mi"

    # Shallow LightGBM gain ranking (default)
    n_rounds = max(10, int(os.environ.get("L1C_FS_LGBM_ROUNDS", "80")))
    dtrain = lgb.Dataset(X, label=y, feature_name=list(cand))
    params = {
        "objective": "binary",
        "verbosity": -1,
        "num_leaves": 31,
        "max_depth": 5,
        "min_data_in_leaf": max(50, X.shape[0] // 200),
        "learning_rate": 0.06,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
    }
    booster = lgb.train(params, dtrain, num_boost_round=n_rounds)
    gain = booster.feature_importance(importance_type="gain").astype(np.float64)
    order = [cand[i] for i in np.argsort(-gain)]
    return order[:cap], "lgbm_gain"


def _select_l1c_feature_cols(
    df: pd.DataFrame,
    feat_cols: list[str],
    config: L1cConfig,
    *,
    train_mask: np.ndarray,
    horizon: int,
) -> tuple[list[str], str]:
    raw = os.environ.get("L1C_FEATURE_SUBSET", "").strip()
    if raw:
        names = [s.strip() for s in raw.split(",") if s.strip()]
        out = [c for c in names if c in df.columns]
        if not out:
            raise RuntimeError(
                f"L1c: L1C_FEATURE_SUBSET produced no valid columns. First names: {names[:12]!r}"
            )
        return out, "subset"
    cand = [c for c in feat_cols if c in df.columns and c not in {"symbol", "time_key"}]
    cap = max(1, int(config.max_train_features))
    picked, method = _l1c_rank_features(df, cand, cap, train_mask, horizon)
    print(
        f"  [L1c] feature select: method={method}  using {len(picked)}/{len(cand)} columns (cap={cap})",
        flush=True,
    )
    return picked, method


def _normalize_l1c_matrix(
    df: pd.DataFrame, feature_cols: list[str], train_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    tr = X[train_mask]
    mean = tr.mean(axis=0).astype(np.float32)
    std = tr.std(axis=0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0).astype(np.float32)
    Xn = ((X - mean) / std).astype(np.float32)
    return Xn, mean, std


def _l1c_build_symbol_windows(
    df: pd.DataFrame, feature_cols: list[str], seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    windows_list: list[np.ndarray] = []
    end_indices: list[np.ndarray] = []
    for _, grp in df.groupby("symbol", sort=False):
        x = grp[feature_cols].to_numpy(dtype=np.float32, copy=False)
        windows, end_idx = _create_tcn_windows(x, seq_len)
        if len(end_idx) == 0:
            continue
        windows_list.append(windows)
        end_indices.append(grp.index.to_numpy()[end_idx])
    if not windows_list:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.concatenate(windows_list, axis=0), np.concatenate(end_indices, axis=0)


def _build_l1c_windows_labels(
    df: pd.DataFrame,
    Xn: np.ndarray,
    *,
    feature_cols: list[str],
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feat_df = pd.DataFrame(Xn, columns=feature_cols, index=df.index)
    work_x = pd.concat([df[["symbol", "time_key"]], feat_df], axis=1)
    windows_list: list[np.ndarray] = []
    end_list: list[np.ndarray] = []
    yz_list: list[np.ndarray] = []
    ret_list: list[np.ndarray] = []
    z_clip = float(np.clip(float(os.environ.get("L1C_Z_CLIP", "5.0")), 1.0, 20.0))
    vol_window = int(np.clip(int(os.environ.get("L1C_VOL_WINDOW", str(seq_len))), 10, 512))
    use_intraday = os.environ.get("L1C_INTRADAY_LABELS", "1").strip().lower() not in {"0", "false", "no", "off"}
    if use_intraday:
        gap_s = float(os.environ.get("L1C_SESSION_GAP_SECONDS", str(4 * 3600)))
        open_skip = int(os.environ.get("L1C_OPEN_SKIP_BARS", "0"))
        close_cut = int(os.environ.get("L1C_CLOSE_CUTOFF_BARS", str(horizon)))
        mark_session_boundaries(
            df,
            time_col="time_key",
            symbol_col="symbol",
            overnight_gap_seconds=gap_s,
            open_skip_bars=open_skip,
            close_cutoff_bars=close_cut,
        )
        print(
            f"  [L1c] intraday fwd labels: session gap > {gap_s:.0f}s | "
            f"open_skip={open_skip} close_cutoff={close_cut} (disable: L1C_INTRADAY_LABELS=0)",
            flush=True,
        )
    for _, grp in work_x.groupby("symbol", sort=False):
        x = grp[feature_cols].to_numpy(dtype=np.float32, copy=False)
        windows, end_local = _create_tcn_windows(x, seq_len)
        if len(end_local) == 0:
            continue
        sub = df.loc[grp.index]
        close = pd.to_numeric(sub["close"], errors="coerce").astype(np.float64).to_numpy()
        close = pd.Series(close).ffill().bfill().to_numpy(dtype=np.float64)
        valid = end_local + horizon < len(grp)
        if not valid.any():
            continue
        windows = windows[valid]
        end_local = end_local[valid]
        idx_global = grp.index.to_numpy()[end_local]
        el = end_local
        ret = (close[el + horizon] - close[el]) / (np.abs(close[el]) + 1e-9)
        if use_intraday:
            sess = sub["session_id"].to_numpy(dtype=np.int32, copy=False)
            vlab = sub["valid_for_label"].to_numpy(dtype=bool, copy=False)
            sess_ok = (sess[el + horizon] == sess[el]) & vlab[el]
            if not sess_ok.any():
                continue
            windows = windows[sess_ok]
            el = el[sess_ok]
            idx_global = idx_global[sess_ok]
            ret = ret[sess_ok]
        one_ret = (close[1:] - close[:-1]) / (np.abs(close[:-1]) + 1e-9)
        one_ret_s = pd.Series(one_ret, dtype=np.float64)
        rolling_vol = one_ret_s.rolling(vol_window, min_periods=max(8, vol_window // 4)).std().shift(1)
        expanding_vol = one_ret_s.expanding(min_periods=8).std().shift(1)
        local_vol = rolling_vol.fillna(expanding_vol).to_numpy(dtype=np.float64)
        local_vol = np.clip(np.nan_to_num(local_vol, nan=np.nanmedian(np.abs(one_ret)) + 1e-6), 1e-6, np.inf)
        z = ret / local_vol[el]
        z = np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -z_clip, z_clip)
        windows_list.append(windows)
        end_list.append(idx_global)
        yz_list.append(z.astype(np.float32))
        ret_list.append(ret.astype(np.float32))
    if not windows_list:
        z = np.zeros(0, dtype=np.float32)
        return (
            np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32),
            z,
            z,
            z,
            np.zeros(0, dtype=np.int64),
        )
    windows = np.concatenate(windows_list, axis=0)
    y_z = np.concatenate(yz_list, axis=0).astype(np.float32)
    y_ret = np.concatenate(ret_list, axis=0).astype(np.float32)
    sw = _sample_weights_from_abs_return(np.abs(y_ret.astype(np.float64)))
    print(
        f"  [L1c] regression labels: z_clip={z_clip:.2f}  vol_window={vol_window}  "
        f"target_z mean={float(np.mean(y_z)):.4f} std={float(np.std(y_z)):.4f}",
        flush=True,
    )
    return windows, y_z, sw, y_ret, np.concatenate(end_list, axis=0)


def materialize_l1c_outputs(
    model: L1cDirectionModel,
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int,
    device: torch.device,
) -> pd.DataFrame:
    work = df.copy(deep=False)
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    outputs = pd.DataFrame(
        {
            "symbol": work["symbol"].values,
            "time_key": pd.to_datetime(work["time_key"]),
        }
    )
    for col in l1c_output_columns():
        outputs[col] = np.float32(0.0)
    chunk_windows = int(np.clip(int(os.environ.get("L1C_MATERIALIZE_WINDOW_CHUNK", "12000")), 512, 100000))
    infer_batch_size = int(np.clip(int(os.environ.get("L1C_MATERIALIZE_BATCH_SIZE", "1024")), 128, 4096))
    rev = np.arange(seq_len, dtype=np.int64)[None, :]
    model.eval()
    with torch.no_grad():
        for sym, grp in work.groupby("symbol", sort=False):
            x = grp[feature_cols].to_numpy(dtype=np.float32, copy=False)
            x = np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            n = x.shape[0]
            if n < seq_len:
                continue
            end_local = np.arange(seq_len - 1, n, dtype=np.int64)
            grp_index = grp.index.to_numpy(dtype=np.int64, copy=False)
            for st in range(0, len(end_local), chunk_windows):
                ends = end_local[st : st + chunk_windows]
                idx = (ends - (seq_len - 1))[:, None] + rev
                windows = x[idx]
                ds = TensorDataset(torch.from_numpy(windows.astype(np.float32, copy=False)))
                dl = DataLoader(ds, batch_size=infer_batch_size, shuffle=False)
                pred_list: list[np.ndarray] = []
                st_list: list[np.ndarray] = []
                for (xb,) in dl:
                    xb = xb.to(device)
                    out = _l1c_forward_dict(model, xb)
                    pred_list.append(out["direction_pred"].float().cpu().numpy().astype(np.float32, copy=False))
                    st_list.append(torch.relu(out["direction_strength"]).float().cpu().numpy().astype(np.float32, copy=False))
                pred_z = np.concatenate(pred_list, axis=0).ravel() if pred_list else np.zeros(0, dtype=np.float32)
                strength = np.concatenate(st_list, axis=0).ravel() if st_list else np.zeros_like(pred_z)
                direction = pred_z.astype(np.float32)
                pred_abs = np.abs(direction).astype(np.float32)
                pred_sign = np.sign(direction).astype(np.float32)
                idx_global = grp_index[ends]
                outputs.loc[idx_global, "l1c_pred_z"] = direction
                outputs.loc[idx_global, "l1c_pred_z_abs"] = pred_abs
                outputs.loc[idx_global, "l1c_pred_sign"] = pred_sign
                outputs.loc[idx_global, "l1c_direction"] = direction
                outputs.loc[idx_global, "l1c_confidence"] = pred_abs
                outputs.loc[idx_global, "l1c_direction_strength"] = strength.astype(np.float32)
                outputs.loc[idx_global, "l1c_is_warm"] = np.float32(1.0)
            print(
                f"  [L1c] materialize symbol={sym} windows={len(end_local):,} chunk={chunk_windows:,} batch={infer_batch_size}",
                flush=True,
            )
    return outputs


def load_l1c_direction_model() -> tuple[L1cDirectionModel, dict[str, Any]]:
    path_meta = os.path.join(MODEL_DIR, L1C_META_FILE)
    with open(path_meta, "rb") as f:
        meta = pickle.load(f)
    if meta.get("schema_version") != L1C_SCHEMA_VERSION:
        raise RuntimeError(
            f"L1c schema mismatch: artifact has {meta.get('schema_version')} but code expects {L1C_SCHEMA_VERSION}."
        )
    cfg_dict = meta.get("l1c_config") or {}
    cfg = L1cConfig(**{k: v for k, v in cfg_dict.items() if k in L1cConfig.__dataclass_fields__})
    cfg.input_dim = int(meta["input_dim"])
    cfg.seq_len = int(meta.get("seq_len", cfg.seq_len))
    model = L1cDirectionModel(cfg).to(DEVICE)
    state = torch.load(
        os.path.join(MODEL_DIR, meta.get("model_file", L1C_MODEL_FILE)),
        map_location=DEVICE,
    )
    model.load_state_dict(state)
    model.eval()
    return model, meta


def infer_l1c_direction(model: L1cDirectionModel, meta: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = list(meta["feature_cols"])
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    seq_len = int(meta.get("seq_len", 60))
    return materialize_l1c_outputs(
        model,
        df,
        feature_cols,
        mean=mean,
        std=std,
        seq_len=seq_len,
        device=DEVICE,
    )


def _train_one_epoch(
    model: L1cDirectionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: L1cRegressionLoss,
    device: torch.device,
    *,
    seq_len: int,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        xb = batch[0].to(device)
        y_z = batch[1].to(device)
        wb = batch[2].to(device)
        yr = batch[3].to(device)
        xb = _l1c_augment_batch(xb, training=True, seq_len=seq_len)
        optimizer.zero_grad()
        out = _l1c_forward_dict(model, xb)
        loss, _parts = criterion(out["direction_pred"], y_z, wb)
        strength_tgt = torch.asinh(torch.abs(yr.view(-1)) * 100.0)
        strength_pred = torch.relu(out["direction_strength"].view(-1))
        strength_loss = F.smooth_l1_loss(strength_pred, strength_tgt, reduction="none")
        strength_loss = (strength_loss * wb.view(-1)).sum() / torch.clamp(wb.sum(), min=1e-6)
        loss = loss + float(getattr(model.config, "strength_aux_weight", 0.10)) * strength_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(os.environ.get("L1C_MAX_GRAD_NORM", "1.0")))
        optimizer.step()
        total += float(loss.detach().cpu()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def _eval_loss(
    model: L1cDirectionModel,
    loader: DataLoader,
    criterion: L1cRegressionLoss,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device)
            y_z = batch[1].to(device)
            wb = batch[2].to(device)
            yr = batch[3].to(device)
            out = _l1c_forward_dict(model, xb)
            loss, _ = criterion(out["direction_pred"], y_z, wb)
            strength_tgt = torch.asinh(torch.abs(yr.view(-1)) * 100.0)
            strength_pred = torch.relu(out["direction_strength"].view(-1))
            strength_loss = F.smooth_l1_loss(strength_pred, strength_tgt, reduction="none")
            strength_loss = (strength_loss * wb.view(-1)).sum() / torch.clamp(wb.sum(), min=1e-6)
            loss = loss + float(getattr(model.config, "strength_aux_weight", 0.10)) * strength_loss
            total += float(loss.cpu()) * xb.size(0)
            n += xb.size(0)
    return total / max(n, 1)


def _l1c_early_stop_best_epoch(
    train_dl: DataLoader,
    val_dl: DataLoader,
    config: L1cConfig,
    *,
    seq_len: int,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    min_delta: float,
    desc: str,
) -> tuple[int, dict[str, torch.Tensor]]:
    """Return (best 1-based epoch, CPU state_dict). Fresh model each call."""
    criterion = L1cRegressionLoss(config)
    model = L1cDirectionModel(config).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    eta_min = float(np.clip(float(os.environ.get("L1C_COS_ETA_MIN", "1e-6")), 1e-8, 1e-3))
    scheduler: CosineAnnealingLR | None = CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(max_epochs)),
        eta_min=eta_min,
    )
    best_val = float("inf")
    stale = 0
    best_ep = 1
    best_state: dict[str, torch.Tensor] | None = None
    epoch_bar = trange(
        max_epochs,
        desc=desc,
        unit="ep",
        leave=False,
        file=_tqdm_stream(),
        disable=not _lgb_round_tqdm_enabled(),
    )
    for ep in epoch_bar:
        _train_one_epoch(model, train_dl, optimizer, criterion, DEVICE, seq_len=seq_len)
        va_loss = _eval_loss(model, val_dl, criterion, DEVICE)
        if scheduler is not None:
            scheduler.step()
        if hasattr(epoch_bar, "set_postfix"):
            epoch_bar.set_postfix(val=f"{va_loss:.4f}", refresh=False)
        if va_loss < (best_val - min_delta):
            best_val = va_loss
            best_ep = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if (ep + 1) >= max(1, int(min_epochs)) and stale >= patience:
                break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return max(1, int(best_ep)), best_state


def _l1c_overlay_expanding_oof_predictions(
    outputs: pd.DataFrame,
    fold_states: list[dict[str, torch.Tensor]],
    fold_va_masks: list[np.ndarray],
    *,
    windows: np.ndarray,
    end_idx: np.ndarray,
    config: L1cConfig,
) -> None:
    for state, w_va in zip(fold_states, fold_va_masks):
        if state is None or not np.any(w_va):
            continue
        model = L1cDirectionModel(config).to(DEVICE)
        model.load_state_dict(state)
        model.eval()
        w_sel = windows[w_va].astype(np.float32, copy=False)
        ds = TensorDataset(torch.from_numpy(w_sel))
        dl = DataLoader(ds, batch_size=1024, shuffle=False)
        pred_l: list[np.ndarray] = []
        strength_l: list[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(DEVICE)
                out = _l1c_forward_dict(model, xb)
                pred_l.append(out["direction_pred"].cpu().numpy().astype(np.float32))
                strength_l.append(torch.relu(out["direction_strength"]).cpu().numpy().astype(np.float32))
        pred_z = np.concatenate(pred_l, axis=0).ravel()
        st = np.concatenate(strength_l, axis=0).ravel()
        direction = pred_z.astype(np.float32)
        pred_abs = np.abs(direction).astype(np.float32)
        pred_sign = np.sign(direction).astype(np.float32)
        rows = end_idx[w_va]
        outputs.loc[rows, "l1c_pred_z"] = direction
        outputs.loc[rows, "l1c_pred_z_abs"] = pred_abs
        outputs.loc[rows, "l1c_pred_sign"] = pred_sign
        outputs.loc[rows, "l1c_direction"] = direction
        outputs.loc[rows, "l1c_confidence"] = pred_abs
        outputs.loc[rows, "l1c_direction_strength"] = st.astype(np.float32)
        outputs.loc[rows, "l1c_is_warm"] = np.float32(1.0)


def train_l1c_direction(df: pd.DataFrame, feat_cols: list[str]) -> None:
    train_started_at = datetime.now().astimezone()
    train_started_perf = time.perf_counter()
    print(f"  [L1c] training started at {train_started_at.strftime('%Y-%m-%d %H:%M:%S %z')}", flush=True)
    config = L1cConfig()
    for _env, _attr in (
        ("L1C_LAYER_DROP", "layer_drop"),
        ("L1C_WEIGHT_DECAY", "weight_decay"),
        ("L1C_NUM_HEADS", "num_heads"),
        ("L1C_NUM_LAYERS", "num_layers"),
        ("L1C_EMBED_DIM", "embed_dim"),
        ("L1C_FF_DIM", "ff_dim"),
        ("L1C_CONV_KERNEL", "conv_kernel_size"),
        ("L1C_CONV_HIDDEN", "conv_hidden_dim"),
        ("L1C_CONV_DROPOUT", "conv_dropout"),
        ("L1C_ATTN_DROPOUT", "attn_dropout"),
        ("L1C_FF_DROPOUT", "ff_dropout"),
        ("L1C_EMBED_DROPOUT", "embed_dropout"),
        ("L1C_LR", "lr"),
        ("L1C_PATIENCE", "patience"),
        ("L1C_MAX_EPOCHS", "max_epochs"),
        ("L1C_MIN_EPOCHS", "min_epochs"),
        ("L1C_EARLY_STOP_MIN_DELTA", "early_stop_min_delta"),
        ("L1C_HUBER_DELTA", "huber_delta"),
        ("L1C_STRENGTH_AUX_WEIGHT", "strength_aux_weight"),
        ("L1C_COS_T0", "cosine_t0"),
        ("L1C_COS_T_MULT", "cosine_t_mult"),
    ):
        raw = os.environ.get(_env, "").strip()
        if raw and _attr in L1cConfig.__dataclass_fields__:
            field_type = L1cConfig.__dataclass_fields__[_attr].type
            if _attr in (
                "num_heads",
                "num_layers",
                "embed_dim",
                "ff_dim",
                "conv_kernel_size",
                "conv_hidden_dim",
                "patience",
                "max_epochs",
                "min_epochs",
                "cosine_t0",
                "cosine_t_mult",
            ):
                setattr(config, _attr, int(float(raw)))
            else:
                setattr(config, _attr, float(raw))
    if FAST_TRAIN_MODE:
        config.max_epochs = min(config.max_epochs, 8)
        config.patience = min(config.patience, 2)
        config.batch_size = min(config.batch_size, 256)

    work = df.copy(deep=False)
    splits = build_stack_time_splits(work["time_key"])
    feature_cols, l1c_feature_select = _select_l1c_feature_cols(
        work,
        feat_cols,
        config,
        train_mask=np.asarray(splits.train_mask, dtype=bool),
        horizon=int(config.predict_horizon),
    )
    l1_fit_mask = np.asarray(splits.train_mask | splits.cal_mask, dtype=bool)
    n_l1_oof = l1_oof_folds_from_env()
    norm_mask = l1_fit_mask if n_l1_oof >= 2 else splits.train_mask
    Xn, mean, std = _normalize_l1c_matrix(work, feature_cols, norm_mask)
    seq_len = int(config.seq_len)
    horizon = int(config.predict_horizon)
    windows, y_z, sw, y_ret, end_idx = _build_l1c_windows_labels(
        work,
        Xn,
        feature_cols=feature_cols,
        seq_len=seq_len,
        horizon=horizon,
    )
    if len(end_idx) == 0:
        raise RuntimeError("L1c: no windows with valid future horizon.")
    n_w = len(end_idx)
    window_train = splits.train_mask[end_idx]
    window_val = splits.l2_val_mask[end_idx]
    window_pool = l1_fit_mask[end_idx]
    l2_vs = l2_val_start_time()
    if n_l1_oof >= 2:
        if l1_oof_mode_from_env() == "expanding":
            print(
                f"  [L1c] expanding calendar OOF: {n_l1_oof} folds (t < {CAL_END}); L1_OOF_MODE=expanding",
                flush=True,
            )
        else:
            print(
                f"  [L1c] blocked time OOF: L1_OOF_FOLDS={n_l1_oof} on train+cal window ends (t < {CAL_END}) "
                f"(set L1_OOF_FOLDS=1 for legacy train vs l2_val [{l2_vs}, {CAL_END}))",
                flush=True,
            )
        if not window_pool.any():
            raise RuntimeError("L1c OOF: no windows with end bar in train+cal.")
        tune_frac = float(os.environ.get("L1_TUNE_FRAC_WITHIN_FIT", "0.5"))
        val_tune_mask, val_report_mask = split_mask_for_tuning_and_report(
            work["time_key"], l1_fit_mask, tune_frac=tune_frac, min_rows_each=50
        )
        if not val_tune_mask.any() or not val_report_mask.any():
            raise RuntimeError("L1c OOF: failed to build tune/report masks inside train+cal.")
        window_val_report = val_report_mask[end_idx]
        if not window_val_report.any():
            raise RuntimeError("L1c OOF: no validation windows in fit_report slice.")
    else:
        if not window_train.any():
            raise RuntimeError("L1c: no training windows (end bar in train split).")
        if not window_val.any():
            raise RuntimeError("L1c: no validation windows (end bar in l2_val).")
        window_val_report = window_val

    log_layer_banner("[L1c] Direction (causal Transformer, single-head regression)")
    print(
        f"  [L1c] artifact dir: {MODEL_DIR}\n"
        f"  [L1c] will write: {artifact_path(L1C_MODEL_FILE)} | {artifact_path(L1C_META_FILE)} | "
        f"{artifact_path(L1C_OUTPUT_CACHE_FILE)}",
        flush=True,
    )
    win_tr_n = int(window_pool.sum()) if n_l1_oof >= 2 else int(window_train.sum())
    win_va_n = int(window_val_report.sum())
    print(
        f"  [L1c] seq_len={seq_len} horizon={horizon} feats={len(feature_cols)} "
        f"fit_windows={win_tr_n} report_windows={win_va_n}",
        flush=True,
    )
    log_mask = window_pool if n_l1_oof >= 2 else window_train
    log_label_baseline("l1c_target_z", y_z[log_mask], task="reg")
    _print_l1c_regression_label_stats(y_z[log_mask], y_ret[log_mask], prefix="fit windows (z_target)")
    print(
        "  [L1c] targets: y_z=ret_fwd/local_vol (winsorized); loss: weighted Huber; "
        "sample_weight=rank(|ret|)/N (set L1C_SAMPLE_WEIGHT_MODE=uniform to disable).",
        flush=True,
    )

    X_t = torch.from_numpy(windows.astype(np.float32, copy=False))
    ds = TensorDataset(
        X_t,
        torch.from_numpy(y_z),
        torch.from_numpy(sw),
        torch.from_numpy(y_ret),
    )
    pin = DEVICE.type == "cuda"
    max_epochs = int(config.max_epochs)
    min_epochs = int(getattr(config, "min_epochs", 0))
    patience = int(config.patience)
    min_delta = float(config.early_stop_min_delta)
    config.input_dim = len(feature_cols)
    nr = int(max_epochs)
    expanding_l1c_fold_states: list[dict[str, torch.Tensor]] | None = None
    expanding_l1c_va_masks: list[np.ndarray] | None = None

    if n_l1_oof >= 2:
        if l1_oof_mode_from_env() == "expanding":
            t_end = work["time_key"].to_numpy()[end_idx]
            exp_folds = l1_expanding_oof_window_folds(t_end)
            fold_pairs: list[tuple[np.ndarray, np.ndarray]] = []
            for w_tr, w_va in exp_folds:
                w_tr_f = w_tr & window_pool
                w_va_f = w_va & window_pool
                if not w_tr_f.any() or not w_va_f.any():
                    raise RuntimeError(
                        "L1c expanding OOF: empty train or val windows "
                        f"(train={int(w_tr_f.sum())}, val={int(w_va_f.sum())})."
                    )
                fold_pairs.append((w_tr_f, w_va_f))
        else:
            w_pool_idx = np.flatnonzero(window_pool)
            tk_sub = work["time_key"].to_numpy()[end_idx[w_pool_idx]]
            fold_masks = time_blocked_fold_masks(tk_sub, np.ones(len(w_pool_idx), bool), n_l1_oof, context="L1c OOF")
            fold_pairs = []
            for tr_sub, va_sub in fold_masks:
                w_tr = np.zeros(n_w, dtype=bool)
                w_va = np.zeros(n_w, dtype=bool)
                w_tr[w_pool_idx[tr_sub]] = True
                w_va[w_pool_idx[va_sub]] = True
                fold_pairs.append((w_tr, w_va))
        best_eps: list[int] = []
        fold_states_l1c: list[dict[str, torch.Tensor]] = []
        for fk, (w_tr, w_va) in enumerate(fold_pairs):
            train_ds_f = TensorDataset(*[t[w_tr] for t in ds.tensors])
            val_ds_f = TensorDataset(*[t[w_va] for t in ds.tensors])
            train_dl_f = DataLoader(train_ds_f, batch_size=config.batch_size, shuffle=True, pin_memory=pin)
            val_dl_f = DataLoader(val_ds_f, batch_size=config.batch_size, shuffle=False, pin_memory=pin)
            be, st = _l1c_early_stop_best_epoch(
                train_dl_f,
                val_dl_f,
                config,
                seq_len=seq_len,
                max_epochs=max_epochs,
                min_epochs=min_epochs,
                patience=patience,
                min_delta=min_delta,
                desc=f"[L1c] oof {fk + 1}/{n_l1_oof}",
            )
            best_eps.append(be)
            fold_states_l1c.append(st)
            print(f"  [L1c] OOF fold {fk + 1}/{n_l1_oof}: best_epoch={be}", flush=True)
        final_min = int(np.clip(int(os.environ.get("L1C_FINAL_EPOCHS_MIN", "8")), 1, max_epochs))
        nr = int(np.clip(max(max(best_eps), final_min), 1, max_epochs))
        print(
            f"  [L1c] OOF best_epoch list={best_eps}  final_epochs=max(max(best),{final_min}) -> {nr}",
            flush=True,
        )
        if l1_oof_mode_from_env() == "expanding":
            expanding_l1c_fold_states = fold_states_l1c
            expanding_l1c_va_masks = [p[1] for p in fold_pairs]
        train_ds = TensorDataset(*[t[window_pool] for t in ds.tensors])
        train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, pin_memory=pin)
        val_ds = TensorDataset(*[t[window_val_report] for t in ds.tensors])
        val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, pin_memory=pin)
        model = L1cDirectionModel(config).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )
        eta_min = float(np.clip(float(os.environ.get("L1C_COS_ETA_MIN", "1e-6")), 1e-8, 1e-3))
        scheduler: CosineAnnealingLR | None = CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(nr)),
            eta_min=eta_min,
        )
        print(
            f"  [L1c] train_config: batch_size={config.batch_size}  lr={float(config.lr):g}  "
            f"weight_decay={float(config.weight_decay):g}  final_epochs={nr} (OOF median)  "
            f"huber_delta={float(config.huber_delta):.3f}",
            flush=True,
        )
        epoch_bar = trange(
            nr,
            desc="[L1c] final fit",
            unit="ep",
            leave=True,
            file=_tqdm_stream(),
            disable=not _lgb_round_tqdm_enabled(),
        )
        criterion_final = L1cRegressionLoss(config)
        for _ep in epoch_bar:
            tr_loss = _train_one_epoch(model, train_dl, optimizer, criterion_final, DEVICE, seq_len=seq_len)
            if _ep == 0:
                _first_epoch_diagnostic(model, train_dl, DEVICE)
            if scheduler is not None:
                scheduler.step()
            if hasattr(epoch_bar, "set_postfix"):
                epoch_bar.set_postfix(train=f"{tr_loss:.4f}", refresh=False)
            print(f"  [L1c] epoch={_ep + 1:02d} train_loss={tr_loss:.4f}", flush=True)
    else:
        train_ds = TensorDataset(*[t[window_train] for t in ds.tensors])
        val_ds = TensorDataset(*[t[window_val] for t in ds.tensors])
        train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, pin_memory=pin)
        val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, pin_memory=pin)
        model = L1cDirectionModel(config).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )
        eta_min = float(np.clip(float(os.environ.get("L1C_COS_ETA_MIN", "1e-6")), 1e-8, 1e-3))
        scheduler: CosineAnnealingLR | None = CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(max_epochs)),
            eta_min=eta_min,
        )
        best_state: dict[str, torch.Tensor] | None = None
        best_val = float("inf")
        stale = 0
        print(
            f"  [L1c] train_config: batch_size={config.batch_size}  lr={float(config.lr):g}  "
            f"weight_decay={float(config.weight_decay):g}  max_epochs={max_epochs}  patience={patience}  "
            f"min_delta={min_delta:g}  huber_delta={float(config.huber_delta):.3f}",
            flush=True,
        )
        epoch_bar = trange(
            max_epochs,
            desc="[L1c] epochs",
            unit="ep",
            leave=True,
            file=_tqdm_stream(),
            disable=not _lgb_round_tqdm_enabled(),
        )
        criterion = L1cRegressionLoss(config)
        for _ep in epoch_bar:
            tr_loss = _train_one_epoch(model, train_dl, optimizer, criterion, DEVICE, seq_len=seq_len)
            if _ep == 0:
                _first_epoch_diagnostic(model, train_dl, DEVICE)
            va_loss = _eval_loss(model, val_dl, criterion, DEVICE)
            if scheduler is not None:
                scheduler.step()
            if hasattr(epoch_bar, "set_postfix"):
                epoch_bar.set_postfix(train=f"{tr_loss:.4f}", val=f"{va_loss:.4f}", refresh=False)
            print(f"  [L1c] epoch={_ep + 1:02d} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}", flush=True)
            if va_loss < (best_val - min_delta):
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if (_ep + 1) >= max(1, int(min_epochs)) and stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("L1c: training failed to produce a checkpoint.")
        model.load_state_dict(best_state)

    val_metrics = evaluate_l1c(model, val_dl, DEVICE)
    print_l1c_eval_report(val_metrics)

    outputs = materialize_l1c_outputs(
        model,
        work,
        feature_cols,
        mean=mean,
        std=std,
        seq_len=seq_len,
        device=DEVICE,
    )
    if (
        n_l1_oof >= 2
        and l1_oof_mode_from_env() == "expanding"
        and expanding_l1c_fold_states is not None
        and expanding_l1c_va_masks is not None
    ):
        _l1c_overlay_expanding_oof_predictions(
            outputs,
            expanding_l1c_fold_states,
            expanding_l1c_va_masks,
            windows=windows,
            end_idx=end_idx,
            config=config,
        )
        print(
            "  [L1c] stitched expanding OOF preds on calibration window ends (honest L1c for L2)",
            flush=True,
        )
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, L1C_MODEL_FILE))
    cfg_dump = asdict(config)
    meta = {
        "schema_version": L1C_SCHEMA_VERSION,
        "feature_cols": feature_cols,
        "l1c_feature_select": l1c_feature_select,
        "seq_len": seq_len,
        "predict_horizon": horizon,
        "mean": mean,
        "std": std,
        "input_dim": len(feature_cols),
        "l1c_config": cfg_dump,
        "output_cols": l1c_output_columns(),
        "device": str(DEVICE),
        "model_file": L1C_MODEL_FILE,
        "output_cache_file": L1C_OUTPUT_CACHE_FILE,
        "direction_target_semantics": (
            f"per-symbol y_z=(close[t+H]-close[t])/close[t] divided by local rolling vol; "
            f"ret=(close[t+H]-close[t])/close[t] at H={horizon} bars; "
            f"fwd return only inside same session when L1C_INTRADAY_LABELS=1 (gap>{os.environ.get('L1C_SESSION_GAP_SECONDS', str(4 * 3600))}s, "
            f"close_cutoff default H); y_z clipped by L1C_Z_CLIP; "
            f"shared backbone + single regression head; huber_delta={float(config.huber_delta):.4f}"
        ),
        "direction_aux_semantics": "dual-branch encoder (Transformer+CNN); auxiliary head predicts asinh(|future_ret|*100)",
        "early_stopping": {
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "min_delta": float(min_delta),
        },
        "l1_oof_mode": l1_oof_mode_from_env(),
        "l1_oof_folds": int(n_l1_oof),
        "l1_oof_enabled": bool(n_l1_oof >= 2),
        "l1_final_epochs_after_oof": int(nr) if n_l1_oof >= 2 else None,
        "val_metrics": val_metrics,
    }
    meta = attach_threshold_registry(
        meta,
        "l1c",
        [
            threshold_entry(
                "L1C_FLAT_MODE",
                str((os.environ.get("L1C_FLAT_MODE", "mde") or "mde").strip().lower()),
                category="adaptive_candidate",
                role="flat-band estimator mode for binary direction labels",
                adaptive_hint="mde default; quantile fallback for compatibility",
                statistical_principle="estimator_selection",
                method_selected=str((os.environ.get("L1C_FLAT_MODE", "mde") or "mde").strip().lower()),
            ),
            threshold_entry(
                "L1C_FLAT_ALPHA",
                float(np.clip(float(os.environ.get("L1C_FLAT_ALPHA", "0.05")), 1e-4, 0.20)),
                category="adaptive_candidate",
                role="type-I error control for MDE flat band",
                statistical_principle="two_sided_significance_level",
                alpha=float(np.clip(float(os.environ.get("L1C_FLAT_ALPHA", "0.05")), 1e-4, 0.20)),
            ),
            threshold_entry(
                "L1C_FLAT_POWER",
                float(np.clip(float(os.environ.get("L1C_FLAT_POWER", "0.80")), 0.50, 0.99)),
                category="adaptive_candidate",
                role="target power for MDE flat band",
                statistical_principle="minimum_detectable_effect",
                power=float(np.clip(float(os.environ.get("L1C_FLAT_POWER", "0.80")), 0.50, 0.99)),
            ),
            threshold_entry(
                "L1C_FLAT_WEIGHT_MODE",
                str((os.environ.get("L1C_FLAT_WEIGHT_MODE", "snr_gaussian") or "snr_gaussian").strip().lower()),
                category="adaptive_candidate",
                role="flat-region sample weighting rule",
                statistical_principle="snr_soft_weighting",
                method_selected=str((os.environ.get("L1C_FLAT_WEIGHT_MODE", "snr_gaussian") or "snr_gaussian").strip().lower()),
            ),
            threshold_entry(
                "L1C_CONF_MODE",
                str((os.environ.get("L1C_CONF_MODE", "cost_based") or "cost_based").strip().lower()),
                category="adaptive_candidate",
                role="confidence zone derivation mode",
                statistical_principle="cost_aware_decision_boundary",
                method_selected=str((os.environ.get("L1C_CONF_MODE", "cost_based") or "cost_based").strip().lower()),
            ),
            threshold_entry(
                "L1C_TX_COST_RATE",
                float(max(0.0, float(os.environ.get("L1C_TX_COST_RATE", "0.0005")))),
                category="safety_constraint",
                role="transaction cost floor for confidence threshold",
                statistical_principle="observed_execution_cost",
                cost_input=float(max(0.0, float(os.environ.get("L1C_TX_COST_RATE", "0.0005")))),
            ),
            threshold_entry(
                "L1C_PATIENCE",
                int(patience),
                category="data_guardrail",
                role="early-stop patience",
            ),
        ],
    )
    with open(os.path.join(MODEL_DIR, L1C_META_FILE), "wb") as f:
        pickle.dump(meta, f)
    cache_path = save_output_cache(outputs, L1C_OUTPUT_CACHE_FILE)
    print(f"  [L1c] model saved -> {os.path.join(MODEL_DIR, L1C_MODEL_FILE)}", flush=True)
    print(f"  [L1c] meta saved  -> {os.path.join(MODEL_DIR, L1C_META_FILE)}", flush=True)
    print(f"  [L1c] cache saved -> {cache_path}", flush=True)
    train_finished_at = datetime.now().astimezone()
    elapsed_sec = max(0.0, time.perf_counter() - train_started_perf)
    print(
        f"  [L1c] training finished at {train_finished_at.strftime('%Y-%m-%d %H:%M:%S %z')}  "
        f"elapsed={elapsed_sec:.1f}s",
        flush=True,
    )
