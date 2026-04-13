from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import asdict, dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from core.trainers.constants import (
    CAL_END,
    L1A_REGIME_COLS,
    L2_META_FILE,
    L2_OUTPUT_CACHE_FILE,
    L3_EXIT_FILE,
    L3_META_FILE,
    L3_POLICY_DATASET_CACHE_FILE,
    L3_SCHEMA_VERSION,
    L3_TRAJECTORY_ENCODER_FILE,
    L3_VALUE_FILE,
    MODEL_DIR,
    TEST_END,
)
from core.trainers.lgbm_utils import (
    TQDM_FILE,
    _decision_edge_atr_array,
    _lgb_round_tqdm_enabled,
    _lgb_train_callbacks_with_round_tqdm,
    _live_trade_state_from_bar,
    _lgbm_n_jobs,
    _net_edge_atr_from_state,
    _options_target_config,
)
from core.trainers.pipeline_train_logs import artifact_path, log_layer_banner, log_numpy_x_stats, log_time_key_arrays
from core.trainers.stack_v2_common import log_label_baseline
from core.trainers.val_metrics_extra import (
    brier_binary,
    directional_accuracy_regression,
    ece_binary,
    flip_rate_sorted,
    pearson_corr,
    regression_degen_flag,
)
from core.trainers.l3_trajectory_hybrid import (
    L3TrajectoryConfig,
    l3_encode_trajectories,
    l3_trajectory_embed_importance_ratio,
    l3_traj_step_features,
    train_l3_trajectory_encoder,
)
from core.trainers.tcn_constants import DEVICE as TORCH_DEVICE


@dataclass
class L3TrainingBundle:
    models: dict[str, Any]
    meta: dict[str, Any]


def _l3_policy_dataset_cache_path() -> str:
    return os.path.join(MODEL_DIR, L3_POLICY_DATASET_CACHE_FILE)


def _hash_frame_columns(df: pd.DataFrame, cols: list[str]) -> str:
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return "empty"
    hashed = pd.util.hash_pandas_object(df[keep], index=True).to_numpy(dtype=np.uint64, copy=False)
    return hashlib.sha1(hashed.tobytes()).hexdigest()


def _l3_policy_dataset_fingerprint(
    df: pd.DataFrame,
    l1a_outputs: pd.DataFrame,
    l2_outputs: pd.DataFrame,
    *,
    max_hold: int,
    build_traj: bool,
    traj_cfg: L3TrajectoryConfig,
) -> dict[str, Any]:
    return {
        "schema_version": L3_SCHEMA_VERSION,
        "max_hold": int(max_hold),
        "build_traj": bool(build_traj),
        "traj_cfg": asdict(traj_cfg),
        "env": {
            "STACK_DECISION_EDGE_TAU": os.environ.get("STACK_DECISION_EDGE_TAU", ""),
            "L3_EXIT_EPSILON_ATR": os.environ.get("L3_EXIT_EPSILON_ATR", ""),
            "L3_TARGET_HORIZON_BARS": os.environ.get("L3_TARGET_HORIZON_BARS", ""),
            "L3_ALLOW_TRUTH_FALLBACK": os.environ.get("L3_ALLOW_TRUTH_FALLBACK", ""),
            "L3_ENTRY_MIN_CONFIDENCE_GRID": os.environ.get("L3_ENTRY_MIN_CONFIDENCE_GRID", ""),
            "L3_ENTRY_MIN_SIZE_GRID": os.environ.get("L3_ENTRY_MIN_SIZE_GRID", ""),
            "L3_ENTRY_POLICY_MIN_STATE_ROWS": os.environ.get("L3_ENTRY_POLICY_MIN_STATE_ROWS", ""),
            "L3_HORIZON_MIN_STATE_ROWS": os.environ.get("L3_HORIZON_MIN_STATE_ROWS", ""),
        },
        "df_hash": _hash_frame_columns(df, ["symbol", "time_key", "open", "high", "low", "close", "lbl_atr"]),
        "l1a_hash": _hash_frame_columns(l1a_outputs, ["symbol", "time_key", *L1A_REGIME_COLS, "l1a_vol_forecast"]),
        "l2_hash": _hash_frame_columns(
            l2_outputs,
            [
                "symbol",
                "time_key",
                "l2_decision_class",
                "l2_decision_confidence",
                "l2_size",
                "l2_pred_mfe",
                "l2_pred_mae",
                *[f"l2_entry_regime_{idx}" for idx in range(len(L1A_REGIME_COLS))],
                "l2_entry_vol",
            ],
        ),
    }


def _load_or_build_l3_policy_dataset(
    df: pd.DataFrame,
    l1a_outputs: pd.DataFrame,
    l2_outputs: pd.DataFrame,
    *,
    max_hold: int,
    traj_cfg: L3TrajectoryConfig,
    build_traj: bool,
):
    use_cache = os.environ.get("L3_POLICY_DATASET_CACHE", "1").strip().lower() in {"1", "true", "yes"}
    force_rebuild = os.environ.get("L3_POLICY_DATASET_CACHE_REBUILD", "").strip().lower() in {"1", "true", "yes"}
    fingerprint = _l3_policy_dataset_fingerprint(
        df,
        l1a_outputs,
        l2_outputs,
        max_hold=max_hold,
        build_traj=build_traj,
        traj_cfg=traj_cfg,
    )
    path = _l3_policy_dataset_cache_path()
    if use_cache and not force_rebuild and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and obj.get("fingerprint") == fingerprint and "payload" in obj:
                print(f"  [L3] loading cached policy dataset -> {path}", flush=True)
                return obj["payload"]
        except Exception as ex:
            print(f"  [L3] policy dataset cache ignored ({ex})", flush=True)
    payload = _build_l3_policy_dataset(df, l1a_outputs, l2_outputs, max_hold=max_hold, traj_cfg=traj_cfg, build_traj=build_traj)
    if use_cache:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"fingerprint": fingerprint, "payload": payload}, f)
        print(f"  [L3] policy dataset cache saved -> {path}", flush=True)
    return payload


def _l3_oot_train_val_masks_by_trade(
    t_state: np.ndarray | pd.Series,
    rows_entry: np.ndarray,
    oot_mask: np.ndarray,
    *,
    train_frac: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Split OOT policy rows into train/val by whole trade (entry signal row id).

    Avoids putting prefixes of the same position in train and later bars in val, which
    would leak trajectory structure into GRU/LGBM validation.
    """
    n = int(rows_entry.shape[0])
    ts = np.asarray(pd.to_datetime(t_state))
    oot = np.asarray(oot_mask, dtype=bool)
    oot_idx = np.flatnonzero(oot)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    if oot_idx.size == 0:
        return train_mask, val_mask

    unique_e = np.unique(rows_entry[oot_idx])
    trade_t0: list[tuple[np.datetime64, int]] = []
    for e in unique_e:
        e = int(e)
        m = (rows_entry == e) & oot
        if not np.any(m):
            continue
        ix = np.flatnonzero(m)
        tmin = np.min(ts[ix])
        trade_t0.append((tmin, e))
    trade_t0.sort(key=lambda x: x[0])
    n_tr = len(trade_t0)
    if n_tr == 0:
        return train_mask, val_mask

    if n_tr >= 2:
        split = max(1, min(n_tr - 1, int(round(n_tr * train_frac))))
        train_entries = {e for _, e in trade_t0[:split]}
        val_entries = {e for _, e in trade_t0[split:]}
        for idx in oot_idx:
            e = int(rows_entry[idx])
            if e in train_entries:
                train_mask[idx] = True
            elif e in val_entries:
                val_mask[idx] = True
    else:
        # Single OOT trade: cannot separate whole trades; fall back to time-ordered bar split.
        print(
            "  [L3] WARNING: only one distinct entry in OOT window; val split is intra-trade "
            "(prefix leakage possible for GRU). Prefer more OOT trades or longer calendar span.",
            flush=True,
        )
        order = oot_idx[np.argsort(ts[oot_idx])]
        split = max(1, min(int(order.size) - 1, int(round(float(order.size) * train_frac))))
        train_mask[order[:split]] = True
        val_mask[order[split:]] = True

    if not val_mask.any() and oot_idx.size:
        val_mask[oot_idx[-1:]] = True
        train_mask[oot_idx[-1:]] = False
    return train_mask, val_mask


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0)
    q = np.clip(q, 1e-6, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1).astype(np.float32)


def _split_l3_val_for_calibration(
    t_state: pd.Series | np.ndarray,
    val_mask: np.ndarray,
    *,
    tune_frac: float,
    min_rows_each: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(val_mask, dtype=bool)
    idx = np.flatnonzero(base)
    tune_mask = np.zeros_like(base, dtype=bool)
    report_mask = np.zeros_like(base, dtype=bool)
    if idx.size == 0:
        return tune_mask, report_mask
    ts = np.asarray(pd.to_datetime(t_state))
    idx = idx[np.argsort(ts[idx])]
    if idx.size < 2 * min_rows_each:
        tune_mask[idx] = True
        report_mask[idx] = True
        return tune_mask, report_mask
    split = int(round(idx.size * float(np.clip(tune_frac, 0.2, 0.8))))
    split = max(min_rows_each, min(idx.size - min_rows_each, split))
    tune_mask[idx[:split]] = True
    report_mask[idx[split:]] = True
    return tune_mask, report_mask


def _fit_l3_exit_calibrator(y_true: np.ndarray, raw_p: np.ndarray) -> IsotonicRegression | None:
    y = np.asarray(y_true, dtype=np.int32).ravel()
    p = np.clip(np.asarray(raw_p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    if y.size < 100 or len(np.unique(y)) < 2:
        return None
    calib = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calib.fit(p, y.astype(np.float64))
    return calib


def _apply_l3_exit_calibrator(p: np.ndarray, calibrator: IsotonicRegression | None) -> np.ndarray:
    arr = np.clip(np.asarray(p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    if calibrator is None:
        return arr
    return np.clip(np.asarray(calibrator.predict(arr), dtype=np.float64).ravel(), 0.0, 1.0)


def _env_float_candidates(key: str, default: list[float], *, lo: float, hi: float) -> list[float]:
    raw = os.environ.get(key, "").strip()
    vals = default
    if raw:
        parsed: list[float] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            parsed.append(float(part))
        if parsed:
            vals = parsed
    clipped = sorted({float(np.clip(v, lo, hi)) for v in vals})
    return clipped or [float(np.clip(default[0], lo, hi))]


def _env_float_clipped(key: str, default: float, *, lo: float, hi: float) -> float:
    raw = os.environ.get(key, "").strip()
    val = default if not raw else float(raw)
    return float(np.clip(val, lo, hi))


def _l3_exit_class_weights() -> tuple[float, float]:
    exit_pos_w = _env_float_clipped("L3_EXIT_POS_WEIGHT", 0.90, lo=0.10, hi=5.0)
    hold_neg_w = _env_float_clipped("L3_EXIT_NEG_WEIGHT", 1.10, lo=0.10, hi=5.0)
    return exit_pos_w, hold_neg_w


def _l3_prepare_value_targets(y_value: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, dict[str, float | str | bool]]:
    y = np.asarray(y_value, dtype=np.float32).copy()
    train = np.asarray(train_mask, dtype=bool).ravel()
    finite_train = train & np.isfinite(y)
    clip_enabled = os.environ.get("L3_VALUE_CLIP", "1").strip().lower() in {"1", "true", "yes"}
    q_lo = _env_float_clipped("L3_VALUE_CLIP_LO_Q", 0.01, lo=0.0, hi=0.25)
    q_hi = _env_float_clipped("L3_VALUE_CLIP_HI_Q", 0.99, lo=0.75, hi=1.0)
    abs_cap = max(0.0, float(os.environ.get("L3_VALUE_CLIP_ABS", "0").strip() or 0.0))
    clip_lo = float("nan")
    clip_hi = float("nan")
    clipped_frac = 0.0
    if clip_enabled and finite_train.any():
        clip_lo = float(np.quantile(y[finite_train], q_lo))
        clip_hi = float(np.quantile(y[finite_train], q_hi))
        if abs_cap > 0.0:
            clip_lo = max(clip_lo, -abs_cap)
            clip_hi = min(clip_hi, abs_cap)
        if clip_hi < clip_lo:
            clip_lo, clip_hi = clip_hi, clip_lo
        below = y < clip_lo
        above = y > clip_hi
        clipped_frac = float(np.mean((below | above)[finite_train])) if finite_train.any() else 0.0
        y[below] = clip_lo
        y[above] = clip_hi
    objective = (os.environ.get("L3_VALUE_OBJECTIVE", "huber").strip().lower() or "huber")
    if objective not in {"huber", "fair", "regression"}:
        objective = "huber"
    metric_default = "l1" if objective in {"huber", "fair"} else "l2"
    metric = (os.environ.get("L3_VALUE_METRIC", metric_default).strip().lower() or metric_default)
    stats: dict[str, float | str | bool] = {
        "clip_enabled": bool(clip_enabled),
        "clip_lo_q": float(q_lo),
        "clip_hi_q": float(q_hi),
        "clip_abs_cap": float(abs_cap),
        "clip_lo": float(clip_lo),
        "clip_hi": float(clip_hi),
        "train_clipped_frac": float(clipped_frac),
        "objective": str(objective),
        "metric": str(metric),
    }
    if objective == "huber":
        stats["huber_alpha"] = _env_float_clipped("L3_VALUE_HUBER_ALPHA", 0.90, lo=0.50, hi=0.99)
    elif objective == "fair":
        stats["fair_c"] = _env_float_clipped("L3_VALUE_FAIR_C", 1.0, lo=0.10, hi=10.0)
    return y, stats


def _l3_value_lgb_params(exit_params: dict[str, Any], *, seed: int, prep: dict[str, float | str | bool]) -> dict[str, Any]:
    params = {**exit_params, "objective": str(prep["objective"]), "metric": str(prep["metric"]), "seed": int(seed)}
    if prep["objective"] == "huber":
        params["alpha"] = float(prep.get("huber_alpha", 0.90))
    elif prep["objective"] == "fair":
        params["fair_c"] = float(prep.get("fair_c", 1.0))
    return params


def _policy_vol_quantiles(values: np.ndarray, *, fit_mask: np.ndarray | None = None, n_buckets: int = 3) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    mask = np.isfinite(arr)
    if fit_mask is not None:
        mask &= np.asarray(fit_mask, dtype=bool).ravel()
    finite = arr[mask]
    if finite.size == 0:
        finite = arr[np.isfinite(arr)]
    if finite.size == 0 or n_buckets <= 1:
        return []
    qs = np.linspace(0.0, 1.0, int(n_buckets) + 1)[1:-1]
    return [float(np.quantile(finite, q)) for q in qs]


def _bucketize_by_quantiles(values: np.ndarray, quantiles: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if not quantiles:
        return np.zeros(len(arr), dtype=np.int32)
    bins = np.asarray(sorted(float(x) for x in quantiles), dtype=np.float64)
    safe = np.nan_to_num(arr, nan=float(np.nanmedian(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0.0)
    return np.searchsorted(bins, safe, side="right").astype(np.int32)


def _hold_bucket_ids(hold_values: np.ndarray) -> np.ndarray:
    arr = np.asarray(hold_values, dtype=np.float64).ravel()
    bins = np.asarray([3.0, 8.0, 15.0], dtype=np.float64)
    safe = np.nan_to_num(arr, nan=0.0)
    return np.searchsorted(bins, safe, side="right").astype(np.int32)


def _regime_ids_from_probs(regime_probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(regime_probs, dtype=np.float64)
    safe = np.nan_to_num(probs, nan=0.0)
    return np.argmax(safe, axis=1).astype(np.int32) if safe.ndim == 2 and safe.shape[1] else np.zeros(len(safe), dtype=np.int32)


def _state_keys_from_regime_vol(regime_probs: np.ndarray, vol_values: np.ndarray, *, vol_quantiles: list[float]) -> np.ndarray:
    reg = _regime_ids_from_probs(regime_probs)
    vb = _bucketize_by_quantiles(vol_values, vol_quantiles)
    return np.asarray([f"r{int(r)}_v{int(v)}" for r, v in zip(reg, vb)], dtype=object)


def _exit_state_keys_from_regime_vol_hold(
    regime_probs: np.ndarray,
    vol_values: np.ndarray,
    hold_values: np.ndarray,
    *,
    vol_quantiles: list[float],
) -> np.ndarray:
    base = _state_keys_from_regime_vol(regime_probs, vol_values, vol_quantiles=vol_quantiles)
    hb = _hold_bucket_ids(hold_values)
    return np.asarray([f"{b}_h{int(h)}" for b, h in zip(base, hb)], dtype=object)


def _l3_entry_policy_defaults() -> tuple[float, float]:
    min_conf = float(np.clip(float(os.environ.get("L3_ENTRY_MIN_CONFIDENCE", "0.0")), 0.0, 1.0))
    min_size = float(max(0.0, float(os.environ.get("L3_ENTRY_MIN_SIZE", "0.05"))))
    return min_conf, min_size


def _l3_lookup_policy_map(
    state_keys: np.ndarray,
    mapping: dict[str, dict[str, float]] | None,
    *,
    defaults: dict[str, float],
) -> dict[str, np.ndarray]:
    keys = np.asarray(state_keys, dtype=object).ravel()
    out = {name: np.full(len(keys), float(value), dtype=np.float32) for name, value in defaults.items()}
    for key, params in (mapping or {}).items():
        m = keys == key
        if not np.any(m):
            continue
        for name in out:
            if name in params:
                out[name][m] = float(params[name])
    return out


def _l3_entry_policy_config(state_key: str | None, meta: dict[str, Any] | None = None) -> tuple[float, float]:
    base_conf, base_size = _l3_entry_policy_defaults()
    if meta is None or state_key is None:
        return base_conf, base_size
    params = (meta.get("l3_entry_policy_by_state") or {}).get(str(state_key), {})
    return (
        float(params.get("min_confidence", meta.get("l3_entry_min_confidence", base_conf))),
        float(params.get("min_size", meta.get("l3_entry_min_size", base_size))),
    )


def _l3_exit_epsilon_atr() -> float:
    return float(max(0.0, float(os.environ.get("L3_EXIT_EPSILON_ATR", "0.03"))))


def _l3_target_horizon_bars(max_hold: int) -> int:
    cfg = _options_target_config()
    raw = int(os.environ.get("L3_TARGET_HORIZON_BARS", str(cfg["decision_horizon_bars"])))
    return max(1, min(int(max_hold), raw))


def _l3_search_entry_policy(
    state_keys: np.ndarray,
    decision_class: np.ndarray,
    decision_confidence: np.ndarray,
    size: np.ndarray,
    edge_atr: np.ndarray,
    tau_edge: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    cls = np.asarray(decision_class, dtype=np.int64).ravel()
    conf = np.asarray(decision_confidence, dtype=np.float64).ravel()
    sz = np.asarray(size, dtype=np.float64).ravel()
    edge = np.asarray(edge_atr, dtype=np.float64).ravel()
    keys = np.asarray(state_keys, dtype=object).ravel()
    truth_dir = np.full(len(edge), 1, dtype=np.int64)
    truth_dir[edge > tau_edge] = 0
    truth_dir[edge < -tau_edge] = 2
    truth_active = truth_dir != 1
    conf_default, size_default = _l3_entry_policy_defaults()
    conf_candidates = _env_float_candidates(
        "L3_ENTRY_MIN_CONFIDENCE_GRID",
        [0.0, float(np.quantile(conf[np.isfinite(conf)], 0.35)) if np.isfinite(conf).any() else conf_default, 0.35, 0.55],
        lo=0.0,
        hi=1.0,
    )
    size_candidates = _env_float_candidates(
        "L3_ENTRY_MIN_SIZE_GRID",
        [0.0, float(np.quantile(sz[np.isfinite(sz)], 0.35)) if np.isfinite(sz).any() else size_default, 0.05, 0.12],
        lo=0.0,
        hi=1.0,
    )
    def _search(mask: np.ndarray) -> dict[str, float]:
        best: dict[str, float] | None = None
        best_score = -1e18
        active_rate_target = float(np.mean(truth_active[mask])) if np.any(mask) else 0.0
        for min_conf in conf_candidates:
            for min_sz in size_candidates:
                entered = mask & (sz >= min_sz) & (conf >= min_conf) & np.isin(cls, [0, 2])
                if not np.any(entered):
                    continue
                correct_side = float(np.mean(cls[entered] == truth_dir[entered]))
                precision = float(np.mean(truth_active[entered]))
                avg_abs_edge = float(np.mean(np.abs(edge[entered])))
                trade_rate = float(np.mean(entered[mask])) if np.any(mask) else 0.0
                score = 0.55 * precision + 0.30 * correct_side + 0.20 * avg_abs_edge - 0.20 * abs(trade_rate - active_rate_target)
                if score > best_score:
                    best_score = score
                    best = {
                        "min_confidence": float(min_conf),
                        "min_size": float(min_sz),
                        "score": float(score),
                        "trade_rate": float(trade_rate),
                        "precision_active": float(precision),
                        "correct_side": float(correct_side),
                        "avg_abs_edge": float(avg_abs_edge),
                    }
        if best is None:
            best = {"min_confidence": conf_default, "min_size": size_default, "score": float("nan")}
        return best
    global_policy = _search(np.isfinite(conf) & np.isfinite(sz) & np.isfinite(edge))
    min_rows = max(80, int(os.environ.get("L3_ENTRY_POLICY_MIN_STATE_ROWS", "140")))
    by_state: dict[str, dict[str, float]] = {}
    for key in sorted({str(k) for k in keys.tolist()}):
        m = (keys == key) & np.isfinite(conf) & np.isfinite(sz) & np.isfinite(edge)
        if int(np.sum(m)) < min_rows:
            continue
        by_state[key] = _search(m)
    print(
        f"  [L3] entry policy: global(min_conf={global_policy['min_confidence']:.3f}, min_size={global_policy['min_size']:.3f})  "
        f"states={len(by_state)}",
        flush=True,
    )
    return global_policy, by_state


def _l3_target_horizon_by_state(
    state_keys: np.ndarray,
    peak_bar: np.ndarray,
    *,
    max_hold: int,
) -> tuple[int, dict[str, int]]:
    base = _l3_target_horizon_bars(max_hold)
    keys = np.asarray(state_keys, dtype=object).ravel()
    peak = np.asarray(peak_bar, dtype=np.float64).ravel()
    finite = np.isfinite(peak)
    global_h = int(np.clip(np.nanmedian(peak[finite]) if finite.any() else base, 2, max_hold))
    by_state: dict[str, int] = {}
    min_rows = max(80, int(os.environ.get("L3_HORIZON_MIN_STATE_ROWS", "120")))
    for key in sorted({str(k) for k in keys.tolist()}):
        m = (keys == key) & finite
        if int(np.sum(m)) < min_rows:
            continue
        by_state[key] = int(np.clip(np.nanmedian(peak[m]), 2, max_hold))
    print(f"  [L3] target horizon: global={global_h}  states={len(by_state)}", flush=True)
    return global_h, by_state


def _l3_policy_row_state_keys(X: np.ndarray, feature_cols: list[str], *, vol_quantiles: list[float]) -> np.ndarray:
    reg_cols = [feature_cols.index(c) for c in L1A_REGIME_COLS]
    vol_idx = feature_cols.index("l1a_vol_forecast")
    regime_probs = np.asarray(X[:, reg_cols], dtype=np.float32)
    vol_values = np.asarray(X[:, vol_idx], dtype=np.float32)
    return _state_keys_from_regime_vol(regime_probs, vol_values, vol_quantiles=vol_quantiles)


def _l3_exit_policy_row_state_keys(X: np.ndarray, feature_cols: list[str], *, vol_quantiles: list[float]) -> np.ndarray:
    reg_cols = [feature_cols.index(c) for c in L1A_REGIME_COLS]
    vol_idx = feature_cols.index("l1a_vol_forecast")
    hold_idx = feature_cols.index("l3_hold_bars")
    regime_probs = np.asarray(X[:, reg_cols], dtype=np.float32)
    vol_values = np.asarray(X[:, vol_idx], dtype=np.float32)
    hold_values = np.asarray(X[:, hold_idx], dtype=np.float32)
    return _exit_state_keys_from_regime_vol_hold(regime_probs, vol_values, hold_values, vol_quantiles=vol_quantiles)


def _l3_search_exit_policy(
    exit_prob: np.ndarray,
    value_pred: np.ndarray,
    y_exit: np.ndarray,
    *,
    value_policy_mode: str | None = None,
    value_tie_margin: float | None = None,
) -> dict[str, float]:
    prob = np.asarray(exit_prob, dtype=np.float64).ravel()
    value = np.asarray(value_pred, dtype=np.float64).ravel()
    y = np.asarray(y_exit, dtype=np.int32).ravel()
    valid = np.isfinite(prob) & np.isfinite(value)
    mode = str(value_policy_mode or os.environ.get("L3_VALUE_POLICY_MODE", "prob_only")).strip().lower() or "prob_only"
    tie_margin = float(max(0.0, float(value_tie_margin if value_tie_margin is not None else os.environ.get("L3_VALUE_TIE_MARGIN", "0.03"))))
    if not valid.any():
        return {
            "exit_prob_threshold": 0.55,
            "value_left_threshold": 0.02,
            "value_policy_mode": mode,
            "value_tie_margin": tie_margin,
            "score": float("nan"),
        }
    prob = prob[valid]
    value = value[valid]
    y = y[valid]
    prob_default = np.quantile(prob, max(0.05, 1.0 - float(np.mean(y)))) if len(prob) else 0.55
    prob_candidates = _env_float_candidates(
        "L3_EXIT_PROB_THRESHOLD_GRID",
        [0.50, float(prob_default), 0.65, 0.75],
        lo=0.05,
        hi=0.95,
    )
    value_candidates = _env_float_candidates(
        "L3_VALUE_LEFT_THRESHOLD_GRID",
        [float(np.quantile(value, q)) for q in (0.25, 0.40, 0.55)] + [0.0, 0.02],
        lo=-2.0,
        hi=2.0,
    )
    target_exit_rate = float(np.mean(y))
    best: dict[str, float] | None = None
    best_score = -1e18
    print("\n  [L3] exit policy search on val_tune", flush=True)
    print(
        f"    prob_thresholds={np.round(prob_candidates, 4).tolist()}  "
        f"value_thresholds={np.round(value_candidates, 4).tolist()}",
        flush=True,
    )
    for prob_thr in prob_candidates:
        for value_thr in value_candidates:
            if mode == "hard_gate":
                pred = ((prob >= prob_thr) & (value <= value_thr)).astype(np.int32)
            elif mode == "tie_break":
                pred = (
                    (prob >= prob_thr)
                    | (((prob >= max(0.0, prob_thr - tie_margin)) & (prob < prob_thr)) & (value <= value_thr))
                ).astype(np.int32)
            else:
                pred = (prob >= prob_thr).astype(np.int32)
            f1 = float(f1_score(y, pred, zero_division=0))
            acc = float(accuracy_score(y, pred))
            exit_rate = float(np.mean(pred))
            rate_pen = abs(exit_rate - target_exit_rate)
            score = 1.00 * f1 + 0.35 * acc - 0.25 * rate_pen
            if score > best_score:
                best_score = score
                best = {
                    "exit_prob_threshold": float(prob_thr),
                    "value_left_threshold": float(value_thr),
                    "score": float(score),
                    "f1": float(f1),
                    "acc": float(acc),
                    "exit_rate": float(exit_rate),
                    "target_exit_rate": float(target_exit_rate),
                    "value_policy_mode": mode,
                    "value_tie_margin": tie_margin,
                }
    if best is None:
        best = {
            "exit_prob_threshold": 0.55,
            "value_left_threshold": 0.02,
            "value_policy_mode": mode,
            "value_tie_margin": tie_margin,
            "score": float("nan"),
        }
    print(
        f"  [L3] selected exit policy: mode={best['value_policy_mode']}  p_exit>={best['exit_prob_threshold']:.4f}  "
        f"value_left<={best['value_left_threshold']:.4f}  F1={best.get('f1', float('nan')):.4f}  acc={best.get('acc', float('nan')):.4f}  "
        f"exit_rate={best.get('exit_rate', float('nan')):.3f}",
        flush=True,
    )
    return best


def _l3_search_conditional_exit_policy(
    state_keys: np.ndarray,
    exit_prob: np.ndarray,
    value_pred: np.ndarray,
    y_exit: np.ndarray,
    *,
    value_policy_mode: str | None = None,
    value_tie_margin: float | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    global_policy = _l3_search_exit_policy(
        exit_prob,
        value_pred,
        y_exit,
        value_policy_mode=value_policy_mode,
        value_tie_margin=value_tie_margin,
    )
    keys = np.asarray(state_keys, dtype=object).ravel()
    valid = np.isfinite(np.asarray(exit_prob, dtype=np.float64).ravel()) & np.isfinite(np.asarray(value_pred, dtype=np.float64).ravel())
    keys = keys[valid]
    by_state: dict[str, dict[str, float]] = {}
    min_rows = max(80, int(os.environ.get("L3_EXIT_POLICY_MIN_STATE_ROWS", "140")))
    for key in sorted({str(k) for k in keys.tolist()}):
        m = keys == key
        if int(np.sum(m)) < min_rows:
            continue
        by_state[key] = _l3_search_exit_policy(
            np.asarray(exit_prob, dtype=np.float64).ravel()[valid][m],
            np.asarray(value_pred, dtype=np.float64).ravel()[valid][m],
            np.asarray(y_exit, dtype=np.int32).ravel()[valid][m],
            value_policy_mode=value_policy_mode,
            value_tie_margin=value_tie_margin,
        )
    print(f"  [L3] conditional exit states learned={len(by_state)}", flush=True)
    return global_policy, by_state


def _choose_l3_value_policy_mode(
    y_true: np.ndarray,
    pred: np.ndarray,
) -> str:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(pred, dtype=np.float64).ravel()
    valid = np.isfinite(yt) & np.isfinite(yp)
    if not valid.any():
        return "prob_only"
    yt = yt[valid]
    yp = yp[valid]
    corr = pearson_corr(yt, yp)
    std_pred = float(np.std(yp))
    if len(np.unique(yt)) > 1:
        try:
            r2 = float(r2_score(yt, yp))
        except Exception:
            r2 = float("nan")
    else:
        r2 = float("nan")
    mode = "tie_break" if std_pred >= 0.05 and corr >= 0.05 and (np.isnan(r2) or r2 >= -0.01) else "prob_only"
    print(
        f"  [L3] value-policy mode selector: corr={corr:.4f}  r2={r2:.4f}  pred_std={std_pred:.6f}  -> {mode}",
        flush=True,
    )
    return mode


def _l3_trade_normalized_exit_weights(
    rows_entry: np.ndarray,
    hold_bars: np.ndarray,
    y_exit: np.ndarray,
) -> np.ndarray:
    entry = np.asarray(rows_entry, dtype=np.int64).ravel()
    hold = np.asarray(hold_bars, dtype=np.float64).ravel()
    y = np.asarray(y_exit, dtype=np.int32).ravel()
    if len(entry) == 0:
        return np.empty(0, dtype=np.float32)
    uniq, inv, counts = np.unique(entry, return_inverse=True, return_counts=True)
    del uniq
    trade_norm = 1.0 / np.maximum(counts[inv].astype(np.float64), 1.0)
    hb = _hold_bucket_ids(hold)
    hold_weight_map = {
        0: float(os.environ.get("L3_HOLD_BUCKET_W0", "1.10")),
        1: float(os.environ.get("L3_HOLD_BUCKET_W1", "1.00")),
        2: float(os.environ.get("L3_HOLD_BUCKET_W2", "0.95")),
        3: float(os.environ.get("L3_HOLD_BUCKET_W3", "0.90")),
    }
    hold_w = np.asarray([hold_weight_map.get(int(h), 1.0) for h in hb], dtype=np.float64)
    exit_pos_w, hold_neg_w = _l3_exit_class_weights()
    cls_w = np.where(y == 1, exit_pos_w, hold_neg_w).astype(np.float64)
    w = trade_norm * hold_w * cls_w
    w = w / max(float(np.mean(w)), 1e-8)
    return w.astype(np.float32)


def _fit_l3_static_ablation(
    X_static: np.ndarray,
    y_exit: np.ndarray,
    y_value_fit: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    rows_entry: np.ndarray,
    feature_cols: list[str],
    *,
    rounds: int,
    es_rounds: int,
    value_prep: dict[str, float | str | bool],
) -> tuple[lgb.Booster, lgb.Booster]:
    exit_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 48,
        "max_depth": 6,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 80,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 171,
        "n_jobs": _lgbm_n_jobs(),
    }
    value_params = _l3_value_lgb_params(exit_params, seed=172, prep=value_prep)
    ih = feature_cols.index("l3_hold_bars")
    w = _l3_trade_normalized_exit_weights(rows_entry[train_mask], X_static[train_mask, ih], y_exit[train_mask])
    cbs, cl = _lgb_train_callbacks_with_round_tqdm(min(es_rounds, 30), min(rounds, 120), "[L3] static-exit")
    try:
        exit_model = lgb.train(
            exit_params,
            lgb.Dataset(
                X_static[train_mask],
                label=y_exit[train_mask],
                weight=w,
                feature_name=feature_cols,
                free_raw_data=False,
            ),
            num_boost_round=min(rounds, 120),
            valid_sets=[lgb.Dataset(X_static[val_mask], label=y_exit[val_mask], feature_name=feature_cols, free_raw_data=False)],
            callbacks=cbs,
        )
    finally:
        for fn in cl:
            fn()
    cbs, cl = _lgb_train_callbacks_with_round_tqdm(min(es_rounds, 30), min(rounds, 120), "[L3] static-value")
    try:
        value_model = lgb.train(
            value_params,
            lgb.Dataset(X_static[train_mask], label=y_value_fit[train_mask], feature_name=feature_cols, free_raw_data=False),
            num_boost_round=min(rounds, 120),
            valid_sets=[lgb.Dataset(X_static[val_mask], label=y_value_fit[val_mask], feature_name=feature_cols, free_raw_data=False)],
            callbacks=cbs,
        )
    finally:
        for fn in cl:
            fn()
    return exit_model, value_model


def l3_entry_side_from_l2(decision_class: int, decision_confidence: float, size: float, *, min_confidence: float, min_size: float) -> float:
    if float(size) < float(min_size) or float(decision_confidence) < float(min_confidence):
        return 0.0
    if int(decision_class) == 0:
        return 1.0
    if int(decision_class) == 2:
        return -1.0
    return 0.0


def l3_policy_state_key(regime_probs: np.ndarray, vol_value: float, meta: dict[str, Any]) -> str:
    vol_quantiles = [float(x) for x in (meta.get("policy_state_vol_quantiles") or [])]
    keys = _state_keys_from_regime_vol(np.asarray(regime_probs, dtype=np.float32).reshape(1, -1), np.asarray([vol_value], dtype=np.float32), vol_quantiles=vol_quantiles)
    return str(keys[0]) if len(keys) else "r0_v0"


def l3_entry_policy_params(regime_probs: np.ndarray, vol_value: float, meta: dict[str, Any]) -> tuple[float, float, int, str]:
    state_key = l3_policy_state_key(regime_probs, vol_value, meta)
    params = (meta.get("l3_entry_policy_by_state") or {}).get(state_key, {})
    min_conf = float(params.get("min_confidence", meta.get("l3_entry_min_confidence", _l3_entry_policy_defaults()[0])))
    min_size = float(params.get("min_size", meta.get("l3_entry_min_size", _l3_entry_policy_defaults()[1])))
    hold_map = meta.get("l3_target_horizon_bars_by_state") or {}
    max_hold = int(hold_map.get(state_key, meta.get("l3_target_horizon_bars", _l3_target_horizon_bars(30))))
    return min_conf, min_size, max_hold, state_key


def l3_exit_policy_params(regime_probs: np.ndarray, vol_value: float, hold_bars: int, meta: dict[str, Any]) -> tuple[float, float, int, str, str, float]:
    state_key = str(
        _exit_state_keys_from_regime_vol_hold(
            np.asarray(regime_probs, dtype=np.float32).reshape(1, -1),
            np.asarray([vol_value], dtype=np.float32),
            np.asarray([hold_bars], dtype=np.float32),
            vol_quantiles=[float(x) for x in (meta.get("policy_state_vol_quantiles") or [])],
        )[0]
    )
    params = (meta.get("l3_exit_policy_by_state") or {}).get(state_key, {})
    prob_thr = float(params.get("exit_prob_threshold", meta.get("l3_exit_prob_threshold", 0.55)))
    value_thr = float(params.get("value_left_threshold", meta.get("l3_value_left_threshold", 0.02)))
    hold_map = meta.get("l3_target_horizon_bars_by_state") or {}
    max_hold = int(hold_map.get(state_key, meta.get("l3_target_horizon_bars", _l3_target_horizon_bars(30))))
    mode = str(params.get("value_policy_mode", meta.get("l3_value_policy_mode", "prob_only")))
    tie_margin = float(params.get("value_tie_margin", meta.get("l3_value_tie_margin", 0.03)))
    return prob_thr, value_thr, max_hold, state_key, mode, tie_margin


def l3_should_exit_by_policy(
    exit_prob: float,
    value_left: float,
    *,
    exit_prob_threshold: float,
    value_left_threshold: float,
    value_policy_mode: str,
    value_tie_margin: float,
) -> bool:
    prob = float(exit_prob)
    value = float(value_left)
    mode = str(value_policy_mode).strip().lower()
    if mode == "hard_gate":
        return prob >= float(exit_prob_threshold) and value <= float(value_left_threshold)
    if mode == "tie_break":
        thr = float(exit_prob_threshold)
        margin = float(max(0.0, value_tie_margin))
        return prob >= thr or (prob >= max(0.0, thr - margin) and value <= float(value_left_threshold))
    return prob >= float(exit_prob_threshold)


def _build_l3_policy_dataset(
    df: pd.DataFrame,
    l1a_outputs: pd.DataFrame,
    l2_outputs: pd.DataFrame,
    *,
    max_hold: int = 30,
    exit_epsilon_atr: float | None = None,
    traj_cfg: L3TrajectoryConfig | None = None,
    build_traj: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    merged = (
        df.reset_index(drop=True)
        .merge(l1a_outputs, on=["symbol", "time_key"], how="left")
        .merge(l2_outputs, on=["symbol", "time_key"], how="left")
    )
    safe_atr = np.where(pd.to_numeric(merged["lbl_atr"], errors="coerce").fillna(0.0).to_numpy() > 1e-3, merged["lbl_atr"].to_numpy(dtype=np.float64), 1e-3)
    open_px = merged["open"].to_numpy(dtype=np.float64)
    high_px = merged["high"].to_numpy(dtype=np.float64)
    low_px = merged["low"].to_numpy(dtype=np.float64)
    close_px = merged["close"].to_numpy(dtype=np.float64)
    symbols = merged["symbol"].to_numpy()
    times = pd.to_datetime(merged["time_key"]).to_numpy()
    current_regime = merged[L1A_REGIME_COLS].to_numpy(dtype=np.float32, copy=False)
    entry_regime = merged[[f"l2_entry_regime_{idx}" for idx in range(len(L1A_REGIME_COLS))]].to_numpy(dtype=np.float32, copy=False)
    current_vol = merged["l1a_vol_forecast"].to_numpy(dtype=np.float32, copy=False)
    entry_vol = merged["l2_entry_vol"].to_numpy(dtype=np.float32, copy=False)
    decision_class = (
        pd.to_numeric(merged["l2_decision_class"], errors="coerce").fillna(1).astype(np.int64).to_numpy()
    )
    decision_conf = merged["l2_decision_confidence"].to_numpy(dtype=np.float32, copy=False)
    size = merged["l2_size"].to_numpy(dtype=np.float32, copy=False)
    edge_atr = _decision_edge_atr_array(merged).astype(np.float64)
    tau_edge = float(max(0.0, float(os.environ.get("STACK_DECISION_EDGE_TAU", "0.05"))))
    exit_epsilon_atr = _l3_exit_epsilon_atr() if exit_epsilon_atr is None else float(max(0.0, exit_epsilon_atr))
    pred_mfe = merged["l2_pred_mfe"].to_numpy(dtype=np.float32, copy=False)
    pred_mae = merged["l2_pred_mae"].to_numpy(dtype=np.float32, copy=False)
    _t_cfg = traj_cfg or L3TrajectoryConfig()
    _t_max = _t_cfg.max_seq_len
    _t_ref = max(_t_max, int(max_hold))
    oot_mask = (times >= np.datetime64(CAL_END)) & (times < np.datetime64(TEST_END))
    policy_vol_quantiles = _policy_vol_quantiles(current_vol, fit_mask=oot_mask)
    state_keys_all = _state_keys_from_regime_vol(current_regime, current_vol, vol_quantiles=policy_vol_quantiles)
    if "decision_peak_bar" in merged.columns:
        peak_bar = pd.to_numeric(merged["decision_peak_bar"], errors="coerce").fillna(_l3_target_horizon_bars(max_hold)).to_numpy(dtype=np.float32)
    else:
        peak_bar = np.full(len(merged), _l3_target_horizon_bars(max_hold), dtype=np.float32)
    target_horizon_global, target_horizon_by_state = _l3_target_horizon_by_state(
        state_keys_all[oot_mask] if oot_mask.any() else state_keys_all,
        peak_bar[oot_mask] if oot_mask.any() else peak_bar,
        max_hold=max_hold,
    )
    entry_policy_global, entry_policy_by_state = _l3_search_entry_policy(
        state_keys_all[oot_mask] if oot_mask.any() else state_keys_all,
        decision_class[oot_mask] if oot_mask.any() else decision_class,
        decision_conf[oot_mask] if oot_mask.any() else decision_conf,
        size[oot_mask] if oot_mask.any() else size,
        edge_atr[oot_mask] if oot_mask.any() else edge_atr,
        tau_edge,
    )
    entry_policy_arrays = _l3_lookup_policy_map(
        state_keys_all,
        entry_policy_by_state,
        defaults={
            "min_confidence": float(entry_policy_global["min_confidence"]),
            "min_size": float(entry_policy_global["min_size"]),
        },
    )
    horizon_arr = np.full(len(merged), int(target_horizon_global), dtype=np.int32)
    for key, horizon in target_horizon_by_state.items():
        horizon_arr[state_keys_all == key] = int(horizon)

    run_end = np.empty(len(merged), dtype=np.int32)
    run_start = 0
    for idx in range(1, len(merged) + 1):
        if idx == len(merged) or symbols[idx] != symbols[run_start]:
            run_end[run_start:idx] = idx
            run_start = idx

    rows_x_blocks: list[np.ndarray] = []
    rows_exit_blocks: list[np.ndarray] = []
    rows_value_blocks: list[np.ndarray] = []
    rows_time_blocks: list[np.ndarray] = []
    rows_entry_blocks: list[np.ndarray] = []
    rows_from_model_blocks: list[np.ndarray] = []
    rows_traj: list[np.ndarray] = []
    rows_traj_len: list[int] = []
    n_policy_signals_model = 0
    n_policy_signals_truth = 0
    allow_truth_fallback = os.environ.get("L3_ALLOW_TRUTH_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}
    _hold_bin_edges = np.array([3, 8, 15, 30, 999], dtype=np.int64)
    feature_cols = [
        "l2_decision_confidence",
        "l2_size",
        "l2_pred_mfe",
        "l2_pred_mae",
        *[f"l2_entry_regime_{idx}" for idx in range(len(L1A_REGIME_COLS))],
        "l2_entry_vol",
        *L1A_REGIME_COLS,
        "l1a_vol_forecast",
        "l3_regime_divergence",
        "l3_vol_surprise",
        "l3_hold_bars",
        "l3_unreal_pnl_atr",
        "l3_live_mfe",
        "l3_live_mae",
        "l3_live_edge",
        "l3_side",
        "l3_log_hold_bars",
        "l3_hold_bars_sq",
        "l3_hold_bucket",
    ]
    row_it = range(len(merged))
    if _lgb_round_tqdm_enabled():
        row_it = tqdm(
            row_it,
            desc="[L3] policy dataset",
            unit="bar",
            leave=False,
            file=TQDM_FILE,
            mininterval=1.0,
        )
    for i in row_it:
        if i + 1 >= len(merged) or run_end[i] <= i + 1:
            continue
        sz = float(size[i])
        min_confidence = float(entry_policy_arrays["min_confidence"][i])
        min_size = float(entry_policy_arrays["min_size"][i])
        model_side = l3_entry_side_from_l2(
            int(decision_class[i]),
            float(decision_conf[i]),
            sz,
            min_confidence=min_confidence,
            min_size=min_size,
        )
        model_active = model_side != 0.0
        ed = float(edge_atr[i])
        truth_dir = 1
        if ed > tau_edge:
            truth_dir = 0
        elif ed < -tau_edge:
            truth_dir = 2
        truth_active = truth_dir != 1
        if model_active:
            side = float(model_side)
            n_policy_signals_model += 1
            from_model = 1
        elif truth_active and allow_truth_fallback:
            # Optional legacy fallback when L2 is too sparse; disabled by default to match deployment.
            side = 1.0 if truth_dir == 0 else -1.0
            n_policy_signals_truth += 1
            from_model = 0
        else:
            continue
        entry_price = float(open_px[i + 1])
        atr = float(safe_atr[i])
        target_horizon = int(max(1, min(int(max_hold), int(horizon_arr[i]))))
        end = min(run_end[i], i + target_horizon + 1, i + max_hold + 1)
        n_steps = int(end - (i + 1))
        if n_steps <= 0:
            continue
        idx_arr = np.arange(i + 1, end, dtype=np.int32)
        holds = np.arange(1, n_steps + 1, dtype=np.float32)
        high_seg = high_px[idx_arr]
        low_seg = low_px[idx_arr]
        close_seg = close_px[idx_arr]
        safe_entry_vol = max(float(entry_vol[i]), 1e-3)
        if side > 0.0:
            fav_seg = np.maximum(0.0, (high_seg - entry_price) / atr)
            adv_seg = np.maximum(0.0, (entry_price - low_seg) / atr)
            unreal_seg = (close_seg - entry_price) / atr
        else:
            fav_seg = np.maximum(0.0, (entry_price - low_seg) / atr)
            adv_seg = np.maximum(0.0, (high_seg - entry_price) / atr)
            unreal_seg = (entry_price - close_seg) / atr
        live_mfe_seg = np.maximum.accumulate(fav_seg.astype(np.float32, copy=False))
        live_mae_seg = np.maximum.accumulate(adv_seg.astype(np.float32, copy=False))
        live_edge_seg = _net_edge_atr_from_state(live_mfe_seg, live_mae_seg, holds).astype(np.float32, copy=False)
        regime_div_seg = _kl_divergence(np.repeat(entry_regime[i : i + 1], n_steps, axis=0), current_regime[idx_arr]).astype(np.float32, copy=False)
        vol_surprise_seg = (current_vol[idx_arr] / safe_entry_vol).astype(np.float32, copy=False)
        log_h_seg = np.log1p(holds).astype(np.float32, copy=False)
        h_sq_seg = ((holds * holds) / 100.0).astype(np.float32, copy=False)
        h_bkt_seg = np.searchsorted(_hold_bin_edges, holds.astype(np.int64), side="right").astype(np.float32, copy=False)
        feat_block = np.column_stack(
            [
                np.full(n_steps, decision_conf[i], dtype=np.float32),
                np.full(n_steps, size[i], dtype=np.float32),
                np.full(n_steps, pred_mfe[i], dtype=np.float32),
                np.full(n_steps, pred_mae[i], dtype=np.float32),
                np.repeat(entry_regime[i : i + 1], n_steps, axis=0),
                np.full(n_steps, entry_vol[i], dtype=np.float32),
                current_regime[idx_arr],
                current_vol[idx_arr].astype(np.float32, copy=False),
                regime_div_seg,
                vol_surprise_seg,
                holds,
                unreal_seg.astype(np.float32, copy=False),
                live_mfe_seg,
                live_mae_seg,
                live_edge_seg,
                np.full(n_steps, side, dtype=np.float32),
                log_h_seg,
                h_sq_seg,
                h_bkt_seg,
            ]
        ).astype(np.float32, copy=False)
        terminal_unreal = float(unreal_seg[-1])
        future_gain_left = (terminal_unreal - unreal_seg).astype(np.float32, copy=False)
        exit_block = ((np.arange(n_steps) == (n_steps - 1)) | (future_gain_left <= exit_epsilon_atr)).astype(np.int32, copy=False)
        rows_x_blocks.append(feat_block)
        rows_exit_blocks.append(exit_block)
        rows_value_blocks.append(future_gain_left)
        rows_time_blocks.append(times[idx_arr])
        rows_entry_blocks.append(np.full(n_steps, int(i), dtype=np.int64))
        rows_from_model_blocks.append(np.full(n_steps, from_model, dtype=np.int32))
        if build_traj:
            traj_hist = np.zeros((_t_max, _t_cfg.seq_feat_dim), dtype=np.float32)
            traj_len_cur = 0
            peak_unreal = -1e9
            prev_unreal = 0.0
            for local_idx, j in enumerate(idx_arr.tolist()):
                peak_unreal = max(peak_unreal, float(unreal_seg[local_idx]))
                tvec = l3_traj_step_features(
                    float(unreal_seg[local_idx]),
                    prev_unreal,
                    peak_unreal,
                    int(holds[local_idx]),
                    times[j],
                    float(close_px[j - 1]),
                    float(close_px[j]),
                    float(high_px[j]),
                    float(low_px[j]),
                    atr,
                    float(vol_surprise_seg[local_idx]),
                    float(regime_div_seg[local_idx]),
                    float(live_mfe_seg[local_idx]),
                    float(live_mae_seg[local_idx]),
                    max_seq_ref=_t_ref,
                )
                prev_unreal = float(unreal_seg[local_idx])
                if traj_len_cur < _t_max:
                    traj_hist[traj_len_cur] = tvec
                    traj_len_cur += 1
                else:
                    traj_hist[:-1] = traj_hist[1:]
                    traj_hist[-1] = tvec
                rows_traj.append(traj_hist.copy())
                rows_traj_len.append(traj_len_cur)
    if not rows_x_blocks:
        print(
            f"  [L3] policy dataset empty: no bars with L2 model trade (class≠neutral, size>0.05) "
            f"or label edge (|decision_edge_atr|>{tau_edge}) at a row with same-symbol next bar.",
            flush=True,
        )
        return (
            np.empty((0, len(feature_cols)), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype="datetime64[ns]"),
            feature_cols,
            np.empty(0, dtype=np.int64),
            np.empty((0, _t_max, _t_cfg.seq_feat_dim), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            {
                "policy_state_vol_quantiles": policy_vol_quantiles,
                "l3_entry_policy": entry_policy_global,
                "l3_entry_policy_by_state": entry_policy_by_state,
                "l3_target_horizon_bars": target_horizon_global,
                "l3_target_horizon_bars_by_state": target_horizon_by_state,
            },
        )
    print(
        f"  [L3] policy dataset: entry signals model={n_policy_signals_model:,} "
        f"truth_edge_fallback={n_policy_signals_truth:,}  policy_rows={sum(int(x.shape[0]) for x in rows_x_blocks):,}  "
        f"allow_truth_fallback={allow_truth_fallback}  entry_policy_states={len(entry_policy_by_state)}  "
        f"target_horizon_global={target_horizon_global}",
        flush=True,
    )
    if n_policy_signals_truth and not n_policy_signals_model:
        print(
            "  [L3] NOTE: all policy entries from label edge fallback — L2 predictions rarely trade; "
            "L3 still uses merged L2 features at each signal bar.",
            flush=True,
        )
    return (
        np.concatenate(rows_x_blocks, axis=0).astype(np.float32, copy=False),
        np.concatenate(rows_exit_blocks, axis=0).astype(np.int32, copy=False),
        np.concatenate(rows_value_blocks, axis=0).astype(np.float32, copy=False),
        np.concatenate(rows_time_blocks, axis=0),
        feature_cols,
        np.concatenate(rows_entry_blocks, axis=0).astype(np.int64, copy=False),
        (
            np.stack(rows_traj, axis=0).astype(np.float32, copy=False)
            if build_traj and rows_traj
            else np.empty((0, _t_max, _t_cfg.seq_feat_dim), dtype=np.float32)
        ),
        (np.asarray(rows_traj_len, dtype=np.int32) if build_traj and rows_traj_len else np.empty(0, dtype=np.int32)),
        np.concatenate(rows_from_model_blocks, axis=0).astype(np.int32, copy=False),
        {
            "policy_state_vol_quantiles": policy_vol_quantiles,
            "l3_entry_policy": entry_policy_global,
            "l3_entry_policy_by_state": entry_policy_by_state,
            "l3_target_horizon_bars": target_horizon_global,
            "l3_target_horizon_bars_by_state": target_horizon_by_state,
        },
    )


def l3_survival_from_hazard(hazard_probs: np.ndarray) -> np.ndarray:
    """Discrete survival S(t)=prod_{s<=t}(1-h(s)) for ordered hazard per episode."""
    h = np.clip(np.asarray(hazard_probs, dtype=np.float64).ravel(), 0.0, 1.0)
    return np.cumprod(1.0 - h)


def l3_group_hazard_by_entry(entry_row_idx: np.ndarray, hazard_probs: np.ndarray) -> dict[int, np.ndarray]:
    """Map entry bar index -> hazard sequence in row order (contiguous per entry in builder)."""
    order: dict[int, list[float]] = {}
    for e, p in zip(np.asarray(entry_row_idx).tolist(), np.asarray(hazard_probs).tolist()):
        order.setdefault(int(e), []).append(float(p))
    return {k: np.asarray(v, dtype=np.float64) for k, v in order.items()}


def _log_l3_val_extended(
    X: np.ndarray,
    y_exit: np.ndarray,
    y_value: np.ndarray,
    t_state: pd.Series | np.ndarray,
    feature_cols: list[str],
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    exit_model: lgb.Booster,
    value_model: lgb.Booster,
    *,
    exit_calibrator: IsotonicRegression | None = None,
) -> None:
    vm = np.asarray(val_mask, dtype=bool)
    if int(vm.sum()) < 5:
        return
    p_exit = _apply_l3_exit_calibrator(exit_model.predict(X[vm]).astype(np.float64), exit_calibrator)
    yv = y_exit[vm].astype(np.int32)
    try:
        ll = float(log_loss(yv, p_exit))
    except ValueError:
        ll = float("nan")
    try:
        auc = float(roc_auc_score(yv, p_exit))
    except ValueError:
        auc = float("nan")
    br = brier_binary(yv.astype(np.float64), p_exit)
    ece = ece_binary(yv, p_exit)
    yhat = (p_exit >= 0.5).astype(np.int32)
    acc = float(accuracy_score(yv, yhat))
    f1 = float(f1_score(yv, yhat, zero_division=0))
    cm = confusion_matrix(yv, yhat, labels=[0, 1])
    print("\n  [L3] val — exit (extended)", flush=True)
    print(
        f"    AUC={auc:.4f}  log_loss={ll:.4f}  Brier={br:.4f}  ECE={ece:.4f}  acc@0.5={acc:.4f}  F1={f1:.4f}",
        flush=True,
    )
    print(f"    confusion [[TN FP][FN TP]]:\n    {cm}", flush=True)

    ih = feature_cols.index("l3_hold_bars")
    hold_vm = np.asarray(X[vm, ih], dtype=np.int64)
    print(" exit AUC by hold bucket (val):", flush=True)
    for lo, hi in [(0, 3), (3, 8), (8, 15), (15, 30), (30, 10_000)]:
        m_hold = (hold_vm >= lo) & (hold_vm < hi)
        n_sub = int(m_hold.sum())
        if n_sub < 50:
            continue
        yy = yv[m_hold]
        pp = p_exit[m_hold]
        if len(np.unique(yy)) < 2:
            continue
        try:
            auc_h = float(roc_auc_score(yy, pp))
        except ValueError:
            auc_h = float("nan")
        print(f"      hold [{lo:>3d}, {hi:>5d}): n={n_sub:>6d}  AUC={auc_h:.4f}", flush=True)

    reg_cols = sorted(
        [c for c in feature_cols if c.startswith("l2_entry_regime_")],
        key=lambda s: int(s.rsplit("_", 1)[-1]),
    )
    if reg_cols:
        reg_idx = [feature_cols.index(c) for c in reg_cols]
        E = X[vm][:, reg_idx]
        reg_id = np.argmax(E, axis=1)
        for k in range(len(reg_cols)):
            m = reg_id == k
            n_k = int(m.sum())
            if n_k < 20:
                continue
            yy = yv[m]
            pp = p_exit[m]
            if len(np.unique(yy)) < 2:
                continue
            try:
                auc_k = float(roc_auc_score(yy, pp))
                print(f"    entry-regime {k}  AUC={auc_k:.4f}  n={n_k:,}", flush=True)
            except ValueError:
                pass

    idx_div = feature_cols.index("l3_regime_divergence")
    dvm = np.asarray(X[vm, idx_div], dtype=np.float64)
    fin = np.isfinite(dvm)
    if fin.any():
        err_abs = np.abs(p_exit - yv.astype(np.float64))
        c_div_err = pearson_corr(dvm[fin], err_abs[fin])
        print(
            f"    l3_regime_divergence (val): mean={float(np.mean(dvm[fin])):.6f}  std={float(np.std(dvm[fin])):.6f}  "
            f"corr(div, |p-y|)={c_div_err:.4f}",
            flush=True,
        )

    i_conf = feature_cols.index("l2_decision_confidence")
    i_size = feature_cols.index("l2_size")
    c_ec = pearson_corr(X[vm, i_conf].astype(np.float64), p_exit)
    vv_pred = value_model.predict(X[vm]).astype(np.float64)
    c_sz_val = pearson_corr(X[vm, i_size].astype(np.float64), vv_pred)
    print(f"    corr(L2 conf, L3 exit p)={c_ec:.4f}  corr(L2 size, L3 value pred)={c_sz_val:.4f}", flush=True)

    t_vm = np.asarray(pd.to_datetime(t_state))[vm]
    order = np.argsort(t_vm)
    fr = flip_rate_sorted(p_exit, order)
    print(f"    exit prob flip_rate (time-sorted val)={fr:.6f}", flush=True)

    vv_true = y_value[vm].astype(np.float64)
    mae_v = float(mean_absolute_error(vv_true, vv_pred))
    rmse_v = float(np.sqrt(mean_squared_error(vv_true, vv_pred)))
    r2_v = float(r2_score(vv_true, vv_pred)) if len(np.unique(vv_true)) > 1 else float("nan")
    c_v = pearson_corr(vv_true, vv_pred)
    std_v, degen_v = regression_degen_flag(vv_pred)
    dir_acc = directional_accuracy_regression(vv_true, vv_pred)
    print("\n  [L3] val — value (extended)", flush=True)
    print(
        f"    MAE={mae_v:.4f}  RMSE={rmse_v:.4f}  R2={r2_v:.4f}  corr={c_v:.4f}  "
        f"dir_acc={dir_acc:.4f}  pred_std={std_v:.6f}  degen={degen_v}",
        flush=True,
    )
    if reg_cols:
        for k in range(len(reg_cols)):
            m = reg_id == k
            n_k = int(m.sum())
            if n_k < 15:
                continue
            yt = vv_true[m]
            yp = vv_pred[m]
            mae_k = float(mean_absolute_error(yt, yp))
            r2_k = float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
            print(f"    entry-regime {k}  MAE={mae_k:.4f}  R2={r2_k:.4f}  n={n_k:,}", flush=True)

    tm = np.asarray(test_mask, dtype=bool)
    if int(tm.sum()) >= 5:
        p_t = _apply_l3_exit_calibrator(exit_model.predict(X[tm]).astype(np.float64), exit_calibrator)
        yt = y_exit[tm].astype(np.int32)
        try:
            auc_t = float(roc_auc_score(yt, p_t))
        except ValueError:
            auc_t = float("nan")
        print(f"\n  [L3] holdout — exit AUC={auc_t:.4f}  n={int(tm.sum()):,}", flush=True)


def train_l3_exit_manager(df: pd.DataFrame, l1a_outputs: pd.DataFrame, l2_outputs: pd.DataFrame) -> L3TrainingBundle:
    traj_cfg = L3TrajectoryConfig()
    want_traj = os.environ.get("L3_TRAJ_GRU", "1").strip().lower() in {"1", "true", "yes"}
    X, y_exit, y_value, t_state, feature_cols, rows_entry, traj_seq, traj_len, rows_from_model, dataset_policy = _load_or_build_l3_policy_dataset(
        df,
        l1a_outputs,
        l2_outputs,
        max_hold=30,
        traj_cfg=traj_cfg,
        build_traj=want_traj,
    )
    if len(X) == 0:
        raise RuntimeError("L3: policy dataset is empty.")
    t_state = pd.to_datetime(t_state)
    oot_mask = (t_state >= np.datetime64(CAL_END)) & (t_state < np.datetime64(TEST_END))
    holdout_mask = t_state >= np.datetime64(TEST_END)
    oot_idx = np.flatnonzero(oot_mask)
    if len(oot_idx) < 20:
        raise RuntimeError("L3: not enough post-CAL_END state rows for strict OOT training.")
    train_mask, val_mask = _l3_oot_train_val_masks_by_trade(
        t_state, rows_entry, oot_mask, train_frac=0.7
    )
    val_tune_frac = float(os.environ.get("L3_VAL_TUNE_FRAC", "0.5"))
    val_tune_mask, val_report_mask = _split_l3_val_for_calibration(
        t_state, val_mask, tune_frac=val_tune_frac, min_rows_each=40
    )
    n_oot_tr = len(np.unique(rows_entry[oot_idx]))
    print(
        f"  [L3] OOT train/val split by trade_id (rows_entry): {n_oot_tr:,} distinct entries in OOT window",
        flush=True,
    )
    test_mask = np.asarray(holdout_mask, dtype=bool)

    log_layer_banner("[L3] Exit / policy (LGBM)")
    log_time_key_arrays(
        "L3",
        pd.Series(t_state[train_mask]),
        pd.Series(t_state[val_mask]),
        train_label="policy train (OOT trades, ~70% of distinct entries)",
        val_label="policy val (OOT trades, ~30% of distinct entries)",
        extra_note=f"Split by rows_entry (signal bar), not by policy row index. Times in [{CAL_END}, {TEST_END}); holdout t>={TEST_END}.",
    )
    log_time_key_arrays(
        "L3(calibration/report)",
        pd.Series(t_state[val_tune_mask]),
        pd.Series(t_state[val_report_mask]),
        train_label="val_tune (exit calibration)",
        val_label="val_report (headline metrics)",
        extra_note="Exit probability calibration is fit on val_tune; headline validation metrics use val_report.",
    )
    if test_mask.any():
        tt_hold = pd.Series(t_state[test_mask])
        print(
            f"  [L3] holdout samples: {int(test_mask.sum()):,}  time_key: [{tt_hold.min()}, {tt_hold.max()}]",
            flush=True,
        )
    model_row_rate = float(np.mean(rows_from_model.astype(np.float64))) if len(rows_from_model) else float("nan")
    print(
        f"  [L3] policy rows from model entries={model_row_rate:.3f}  "
        f"(fallback rows={int(np.sum(rows_from_model == 0)):,})",
        flush=True,
    )
    idx_div = feature_cols.index("l3_regime_divergence")
    div_all = np.asarray(X[:, idx_div], dtype=np.float64)
    fin = np.isfinite(div_all)
    print(
        f"  [L3] l3_regime_divergence: min={np.nanmin(div_all):.6f}  max={np.nanmax(div_all):.6f}  "
        f"mean={float(np.nanmean(div_all)):.6f}  finite={int(fin.sum()):,}/{len(div_all):,}",
        flush=True,
    )
    if (~fin).any():
        print(f"  [L3] l3_regime_divergence non-finite count: {int((~fin).sum())}", flush=True)
    log_numpy_x_stats("L3", X[train_mask], label="X[policy_train]")
    print(
        f"  [L3] L2 artifact ref (features come from l2_outputs merge): {artifact_path(L2_META_FILE)} / cache {artifact_path(L2_OUTPUT_CACHE_FILE)}",
        flush=True,
    )
    print(
        "  [L3] note: l2_* / l1a_* in policy rows come from supplied upstream outputs; preferred pipeline path uses frozen-artifact inference caches.",
        flush=True,
    )
    print(
        "  [L3] target semantics: continuation value to a fixed deadline (non-oracle), not optimal future exit.",
        flush=True,
    )
    print(f"  [L3] will write: {artifact_path(L3_EXIT_FILE)} | {artifact_path(L3_VALUE_FILE)} | {artifact_path(L3_META_FILE)}", flush=True)
    log_label_baseline("l3_exit", y_exit[train_mask], task="cls")
    log_label_baseline("l3_value", y_value[train_mask], task="reg")
    y_value_fit, value_prep = _l3_prepare_value_targets(y_value, train_mask)
    print(
        f"  [L3] value target prep: objective={value_prep['objective']} metric={value_prep['metric']}  "
        f"clip={bool(value_prep['clip_enabled'])} q=[{float(value_prep['clip_lo_q']):.2f}, {float(value_prep['clip_hi_q']):.2f}]  "
        f"train_clip=[{float(value_prep['clip_lo']):.4f}, {float(value_prep['clip_hi']):.4f}]  "
        f"train_clipped_frac={float(value_prep['train_clipped_frac']):.3f}",
        flush=True,
    )
    exit_pos_w, hold_neg_w = _l3_exit_class_weights()
    print(
        f"  [L3] exit class weights: hold={hold_neg_w:.3f}  exit={exit_pos_w:.3f}",
        flush=True,
    )

    use_hybrid = want_traj
    use_hybrid = (
        use_hybrid
        and int(train_mask.sum()) >= 300
        and int(val_mask.sum()) >= 30
        and traj_seq.shape[0] == len(X)
    )
    static_cols = list(feature_cols)
    n_static = len(static_cols)
    if use_hybrid:
        print("  [L3] hybrid: training trajectory GRU encoder (L3_TRAJ_GRU=1)...", flush=True)
        encoder = train_l3_trajectory_encoder(
            traj_seq[train_mask],
            traj_len[train_mask],
            y_exit[train_mask].astype(np.float32),
            y_value_fit[train_mask].astype(np.float32),
            traj_seq[val_mask],
            traj_len[val_mask],
            y_exit[val_mask].astype(np.float32),
            y_value_fit[val_mask].astype(np.float32),
            cfg=traj_cfg,
            device=TORCH_DEVICE,
        )
        emb_all = l3_encode_trajectories(encoder, traj_seq, traj_len, TORCH_DEVICE)
        emb_names = [f"l3_traj_emb_{k}" for k in range(traj_cfg.embed_dim)]
        X_lgb = np.hstack([X, emb_all.astype(np.float32, copy=False)])
        feature_cols = static_cols + emb_names
        print(
            f"  [L3] hybrid layout: static={n_static}  traj_step_dim={traj_cfg.seq_feat_dim}  "
            f"embed_dim={traj_cfg.embed_dim}  lgbm_cols={len(feature_cols)}",
            flush=True,
        )
    else:
        if not os.environ.get("L3_TRAJ_GRU", "1").strip().lower() in {"1", "true", "yes"}:
            print("  [L3] hybrid disabled (L3_TRAJ_GRU=0); LightGBM on static features only.", flush=True)
        else:
            print(
                f"  [L3] hybrid skipped (need train>=300 & val>=30); "
                f"got train={int(train_mask.sum())} val={int(val_mask.sum())}.",
                flush=True,
            )
        encoder = None
        X_lgb = X
        feature_cols = static_cols

    rounds = 250
    es_rounds = 40
    exit_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 48,
        "max_depth": 6,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 80,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 71,
        "n_jobs": _lgbm_n_jobs(),
    }
    value_params = _l3_value_lgb_params(exit_params, seed=72, prep=value_prep)
    static_exit_model = None
    static_value_model = None
    l3_outer = tqdm(
        total=2,
        desc="[L3] models",
        unit="model",
        leave=True,
        file=TQDM_FILE,
        disable=not _lgb_round_tqdm_enabled(),
    )
    try:
        cbs, cl = _lgb_train_callbacks_with_round_tqdm(es_rounds, rounds, "[L3] exit")
        try:
            ih_tr = feature_cols.index("l3_hold_bars")
            hold_tr = X_lgb[train_mask, ih_tr].astype(np.float64)
            w_exit = _l3_trade_normalized_exit_weights(rows_entry[train_mask], hold_tr, y_exit[train_mask])
            exit_model = lgb.train(
                exit_params,
                lgb.Dataset(
                    X_lgb[train_mask],
                    label=y_exit[train_mask],
                    weight=w_exit.astype(np.float32),
                    feature_name=feature_cols,
                    free_raw_data=False,
                ),
                num_boost_round=rounds,
                valid_sets=[lgb.Dataset(X_lgb[val_mask], label=y_exit[val_mask], feature_name=feature_cols, free_raw_data=False)],
                callbacks=cbs,
            )
        finally:
            for fn in cl:
                fn()
        l3_outer.set_postfix_str("exit", refresh=False)
        l3_outer.update(1)

        cbs, cl = _lgb_train_callbacks_with_round_tqdm(es_rounds, rounds, "[L3] value")
        try:
            value_model = lgb.train(
                value_params,
                lgb.Dataset(X_lgb[train_mask], label=y_value_fit[train_mask], feature_name=feature_cols, free_raw_data=False),
                num_boost_round=rounds,
                valid_sets=[lgb.Dataset(X_lgb[val_mask], label=y_value_fit[val_mask], feature_name=feature_cols, free_raw_data=False)],
                callbacks=cbs,
            )
        finally:
            for fn in cl:
                fn()
        l3_outer.set_postfix_str("value", refresh=False)
        l3_outer.update(1)
    finally:
        l3_outer.close()
    exit_calibrator = _fit_l3_exit_calibrator(
        y_exit[val_tune_mask],
        exit_model.predict(X_lgb[val_tune_mask]).astype(np.float64),
    )
    exit_prob_tune = _apply_l3_exit_calibrator(exit_model.predict(X_lgb[val_tune_mask]).astype(np.float64), exit_calibrator)
    value_pred_tune = value_model.predict(X_lgb[val_tune_mask]).astype(np.float64)
    value_policy_mode = _choose_l3_value_policy_mode(y_value[val_tune_mask], value_pred_tune)
    value_tie_margin = float(max(0.0, float(os.environ.get("L3_VALUE_TIE_MARGIN", "0.03"))))
    exit_state_keys = _l3_exit_policy_row_state_keys(
        X[val_tune_mask],
        static_cols,
        vol_quantiles=[float(x) for x in (dataset_policy.get("policy_state_vol_quantiles") or [])],
    )
    exit_policy, exit_policy_by_state = _l3_search_conditional_exit_policy(
        exit_state_keys,
        exit_prob_tune,
        value_pred_tune,
        y_exit[val_tune_mask],
        value_policy_mode=value_policy_mode,
        value_tie_margin=value_tie_margin,
    )
    if use_hybrid and os.environ.get("L3_GRU_ABLATION", "1").strip().lower() in {"1", "true", "yes"}:
        print("  [L3] GRU ablation: fitting static-only comparators...", flush=True)
        static_exit_model, static_value_model = _fit_l3_static_ablation(
            X,
            y_exit,
            y_value_fit,
            train_mask,
            val_mask,
            rows_entry,
            static_cols,
            rounds=rounds,
            es_rounds=es_rounds,
            value_prep=value_prep,
        )
        p_static = _apply_l3_exit_calibrator(
            static_exit_model.predict(X[val_report_mask]).astype(np.float64),
            _fit_l3_exit_calibrator(y_exit[val_tune_mask], static_exit_model.predict(X[val_tune_mask]).astype(np.float64)),
        )
        p_hybrid = _apply_l3_exit_calibrator(exit_model.predict(X_lgb[val_report_mask]).astype(np.float64), exit_calibrator)
        auc_static = float(roc_auc_score(y_exit[val_report_mask].astype(np.int32), p_static))
        auc_hybrid = float(roc_auc_score(y_exit[val_report_mask].astype(np.int32), p_hybrid))
        v_static = static_value_model.predict(X[val_report_mask]).astype(np.float64)
        v_hybrid = value_model.predict(X_lgb[val_report_mask]).astype(np.float64)
        r2_static = float(r2_score(y_value[val_report_mask].astype(np.float64), v_static)) if len(np.unique(y_value[val_report_mask])) > 1 else float("nan")
        r2_hybrid = float(r2_score(y_value[val_report_mask].astype(np.float64), v_hybrid)) if len(np.unique(y_value[val_report_mask])) > 1 else float("nan")
        print(
            f"  [L3] GRU ablation (val_report): exit_auc static={auc_static:.4f}  hybrid={auc_hybrid:.4f}  "
            f"delta={auc_hybrid - auc_static:+.4f} | value_r2 static={r2_static:.4f}  hybrid={r2_hybrid:.4f}  "
            f"delta={r2_hybrid - r2_static:+.4f}",
            flush=True,
        )
    if use_hybrid:
        st_sh, em_sh = l3_trajectory_embed_importance_ratio(exit_model, n_static, traj_cfg.embed_dim)
        st_v, em_v = l3_trajectory_embed_importance_ratio(value_model, n_static, traj_cfg.embed_dim)
        print(
            f"  [L3] gain importance share — exit: static={st_sh:.1%}  traj_emb={em_sh:.1%}  |  "
            f"value: static={st_v:.1%}  traj_emb={em_v:.1%}",
            flush=True,
        )
    _log_l3_val_extended(
        X_lgb,
        y_exit,
        y_value,
        t_state,
        feature_cols,
        val_report_mask,
        test_mask,
        exit_model,
        value_model,
        exit_calibrator=exit_calibrator,
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    exit_model.save_model(os.path.join(MODEL_DIR, L3_EXIT_FILE))
    value_model.save_model(os.path.join(MODEL_DIR, L3_VALUE_FILE))
    model_files: dict[str, str] = {"exit": L3_EXIT_FILE, "value": L3_VALUE_FILE}
    if exit_calibrator is not None:
        exit_calib_file = "l3_exit_calibrator.pkl"
        with open(os.path.join(MODEL_DIR, exit_calib_file), "wb") as f:
            pickle.dump(exit_calibrator, f)
        model_files["exit_calibrator"] = exit_calib_file
    entry_min_confidence = float(dataset_policy.get("l3_entry_policy", {}).get("min_confidence", _l3_entry_policy_defaults()[0]))
    entry_min_size = float(dataset_policy.get("l3_entry_policy", {}).get("min_size", _l3_entry_policy_defaults()[1]))
    meta = {
        "schema_version": L3_SCHEMA_VERSION,
        "feature_cols": feature_cols,
        "model_files": model_files,
        "derived_features": [
            "l3_regime_divergence",
            "l3_vol_surprise",
            "l3_log_hold_bars",
            "l3_hold_bars_sq",
            "l3_hold_bucket",
        ],
        "l3_hybrid": bool(use_hybrid),
        "l3_val_tune_frac": val_tune_frac,
        "l3_allow_truth_fallback": os.environ.get("L3_ALLOW_TRUTH_FALLBACK", "0").strip().lower() in {"1", "true", "yes"},
        "l3_entry_min_confidence": entry_min_confidence,
        "l3_entry_min_size": entry_min_size,
        "policy_state_vol_quantiles": dataset_policy.get("policy_state_vol_quantiles", []),
        "l3_entry_policy_by_state": dataset_policy.get("l3_entry_policy_by_state", {}),
        "l3_target_horizon_bars": int(dataset_policy.get("l3_target_horizon_bars", _l3_target_horizon_bars(30))),
        "l3_target_horizon_bars_by_state": dataset_policy.get("l3_target_horizon_bars_by_state", {}),
        "l3_exit_epsilon_atr": _l3_exit_epsilon_atr(),
        "l3_exit_prob_threshold": float(exit_policy["exit_prob_threshold"]),
        "l3_value_left_threshold": float(exit_policy["value_left_threshold"]),
        "l3_value_policy_mode": str(exit_policy.get("value_policy_mode", value_policy_mode)),
        "l3_value_tie_margin": float(exit_policy.get("value_tie_margin", value_tie_margin)),
        "l3_exit_policy_search": exit_policy,
        "l3_exit_policy_by_state": exit_policy_by_state,
        "l3_target_semantics": "continuation_value_to_fixed_deadline",
        "l3_exit_class_weights": {
            "hold": float(hold_neg_w),
            "exit": float(exit_pos_w),
        },
        "l3_value_training": value_prep,
    }
    if use_hybrid:
        import torch

        meta["l3_traj_encoder_file"] = L3_TRAJECTORY_ENCODER_FILE
        meta["l3_traj_cfg"] = asdict(traj_cfg)
        torch.save(
            encoder.state_dict(),
            os.path.join(MODEL_DIR, L3_TRAJECTORY_ENCODER_FILE),
        )
    with open(os.path.join(MODEL_DIR, L3_META_FILE), "wb") as f:
        pickle.dump(meta, f)
    print(
        f"  [L3] strict OOT split: train={int(train_mask.sum()):,} val={int(val_mask.sum()):,} "
        f"(tune={int(val_tune_mask.sum()):,}, report={int(val_report_mask.sum()):,}) holdout={int(test_mask.sum()):,}",
        flush=True,
    )
    if test_mask.any():
        prob = _apply_l3_exit_calibrator(exit_model.predict(X_lgb[test_mask]), exit_calibrator)
        print(f"  [L3] test mean exit prob={float(np.mean(prob)):.4f}", flush=True)
    print(f"  [L3] meta saved -> {os.path.join(MODEL_DIR, L3_META_FILE)}", flush=True)
    bundle_models: dict[str, Any] = {"exit": exit_model, "value": value_model}
    if exit_calibrator is not None:
        bundle_models["exit_calibrator"] = exit_calibrator
    return L3TrainingBundle(models=bundle_models, meta=meta)


def load_l3_exit_manager() -> tuple[dict[str, Any], dict[str, Any]]:
    with open(os.path.join(MODEL_DIR, L3_META_FILE), "rb") as f:
        meta = pickle.load(f)
    if meta.get("schema_version") != L3_SCHEMA_VERSION:
        raise RuntimeError(
            f"L3 schema mismatch: artifact has {meta.get('schema_version')} but code expects {L3_SCHEMA_VERSION}. "
            f"Retrain L3 so artifacts match schema {L3_SCHEMA_VERSION}."
        )
    model_files = meta.get("model_files", {})
    models: dict[str, Any] = {
        name: lgb.Booster(model_file=os.path.join(MODEL_DIR, fname))
        for name, fname in model_files.items()
        if name != "exit_calibrator"
    }
    exit_calib_file = model_files.get("exit_calibrator")
    if exit_calib_file:
        with open(os.path.join(MODEL_DIR, exit_calib_file), "rb") as f:
            models["exit_calibrator"] = pickle.load(f)
    return models, meta


def load_l3_trajectory_encoder_for_infer(meta: dict[str, Any]) -> tuple[Any, L3TrajectoryConfig | None]:
    """Load GRU encoder when meta['l3_hybrid'] is True; else (None, None)."""
    if not meta.get("l3_hybrid"):
        return None, None
    import torch

    from core.trainers.l3_trajectory_hybrid import L3TrajectoryEncoder

    cfg_d = meta.get("l3_traj_cfg")
    if not cfg_d:
        return None, None
    cfg = L3TrajectoryConfig(**cfg_d)
    enc = L3TrajectoryEncoder(cfg)
    path = os.path.join(MODEL_DIR, meta.get("l3_traj_encoder_file", L3_TRAJECTORY_ENCODER_FILE))
    enc.load_state_dict(torch.load(path, map_location="cpu"))
    enc.eval()
    return enc, cfg
