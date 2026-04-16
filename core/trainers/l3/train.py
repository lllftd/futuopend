from __future__ import annotations

import hashlib
import os
import pickle
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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
    L3_COX_FILE,
    L3_EXIT_FILE,
    L3_META_FILE,
    L3_POLICY_DATASET_CACHE_FILE,
    L3_SCHEMA_VERSION,
    L3_TRAJECTORY_ENCODER_FILE,
    L3_VALUE_FILE,
    MODEL_DIR,
    PA_STATE_FEATURES,
    TEST_END,
)

try:
    from lifelines import CoxTimeVaryingFitter
except ImportError:  # pragma: no cover
    CoxTimeVaryingFitter = None  # type: ignore[misc, assignment]

from core.trainers.lgbm_utils import (
    TQDM_FILE,
    _decision_edge_atr_array,
    _lgb_round_tqdm_enabled,
    _lgb_train_callbacks_with_round_tqdm,
    _live_trade_state_from_bar,
    _lgbm_n_jobs,
    _net_edge_atr_from_state,
)
from core.trainers.l3_policy_params import derive_policy_params
from core.trainers.pipeline_train_logs import artifact_path, log_layer_banner, log_numpy_x_stats, log_time_key_arrays
from core.trainers.stack_v2_common import log_label_baseline
from core.trainers.threshold_registry import attach_threshold_registry, threshold_entry
from core.trainers.val_metrics_extra import (
    brier_binary,
    directional_accuracy_regression,
    ece_binary,
    flip_rate_sorted,
    pearson_corr,
    regression_degen_flag,
)
from core.trainers.l3.trajectory import (
    L3TrajectoryConfig,
    l3_encode_trajectories,
    l3_trajectory_embed_importance_ratio,
    l3_traj_step_features,
    train_l3_trajectory_encoder,
)
from core.trainers.pa_state_controls import (
    ensure_pa_state_features,
    pa_state_arrays_from_frame,
    pa_state_bucket_label_from_mapping,
    pa_state_bucket_labels_from_arrays,
    pa_state_bucket_labels_from_frame,
)
from core.trainers.tcn_constants import DEVICE as TORCH_DEVICE


@dataclass
class L3TrainingBundle:
    models: dict[str, Any]
    meta: dict[str, Any]


@dataclass
class L3ExitInferenceState:
    """Per-trade runtime for EMA + hysteresis (inference only). Reset on new entry."""

    ema_prob: float | None = None
    latch_exit: bool = False

    def reset(self) -> None:
        self.ema_prob = None
        self.latch_exit = False


def _l3_exit_infer_enabled(meta: Mapping[str, Any] | None) -> bool:
    if meta and "l3_exit_infer_smooth" in meta:
        return bool(meta["l3_exit_infer_smooth"])
    return os.environ.get("L3_EXIT_INFER_SMOOTH", "1").strip().lower() in {"1", "true", "yes"}


def _l3_exit_infer_params(meta: Mapping[str, Any] | None) -> dict[str, float]:
    m = dict(meta or {})
    return {
        "ema_alpha": float(m.get("l3_exit_ema_alpha", float(os.environ.get("L3_EXIT_EMA_ALPHA", "0.3")))),
        "enter_thr": float(m.get("l3_exit_hyst_enter", float(os.environ.get("L3_EXIT_HYST_ENTER", "0.55")))),
        "leave_thr": float(m.get("l3_exit_hyst_leave", float(os.environ.get("L3_EXIT_HYST_LEAVE", "0.35")))),
        "min_hold": float(m.get("l3_min_hold_bars", float(os.environ.get("L3_MIN_HOLD_BARS", "3")))),
    }


def l3_exit_decision_live(
    exit_prob_calibrated: float,
    value_left: float,
    state: L3ExitInferenceState,
    hold_bars: int,
    *,
    exit_prob_threshold: float,
    value_left_threshold: float,
    value_policy_mode: str,
    value_tie_margin: float,
    meta: Mapping[str, Any] | None = None,
) -> tuple[bool, L3ExitInferenceState]:
    """Exit decision for live/backtest: optional EMA + hysteresis + min-hold; else legacy policy.

    EMA: first in-trade update sets ``ema_prob`` to the **current** raw calibrated ``p_exit`` (not 0);
    later bars use exponential smoothing. ``min_hold`` only suppresses *exit* and hysteresis until
    ``hold_bars >= min_hold``; EMA still updates on earlier bars so the first post-min_hold decision
    uses a smoothed history.
    """
    p_raw = float(np.clip(exit_prob_calibrated, 0.0, 1.0))
    ip = _l3_exit_infer_params(meta)
    min_hold = int(max(0, round(ip["min_hold"])))
    if _l3_exit_infer_enabled(meta):
        alpha = float(np.clip(ip["ema_alpha"], 0.01, 0.99))
        if state.ema_prob is None:
            state.ema_prob = p_raw
        else:
            state.ema_prob = alpha * p_raw + (1.0 - alpha) * float(state.ema_prob)
        if hold_bars < min_hold:
            return False, state
        p_s = float(state.ema_prob)
        ent = float(ip["enter_thr"])
        lev = float(ip["leave_thr"])
        if not state.latch_exit:
            if p_s >= ent:
                state.latch_exit = True
        else:
            if p_s <= lev:
                state.latch_exit = False
        return bool(state.latch_exit), state
    ex = l3_should_exit_by_policy(
        p_raw,
        value_left,
        exit_prob_threshold=exit_prob_threshold,
        value_left_threshold=value_left_threshold,
        value_policy_mode=value_policy_mode,
        value_tie_margin=value_tie_margin,
    )
    return ex, state


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
            "L3_EXIT_LOSS_BUFFER_ATR": os.environ.get("L3_EXIT_LOSS_BUFFER_ATR", ""),
            "L3_EXIT_LIVE_EDGE_FLOOR": os.environ.get("L3_EXIT_LIVE_EDGE_FLOOR", ""),
            "L3_CONTINUATION_EDGE_MULT": os.environ.get("L3_CONTINUATION_EDGE_MULT", ""),
            "L3_TARGET_HORIZON_BARS": os.environ.get("L3_TARGET_HORIZON_BARS", ""),
            "L3_ALLOW_TRUTH_FALLBACK": os.environ.get("L3_ALLOW_TRUTH_FALLBACK", ""),
            "L3_ENTRY_MIN_CONFIDENCE_GRID": os.environ.get("L3_ENTRY_MIN_CONFIDENCE_GRID", ""),
            "L3_ENTRY_MIN_SIZE_GRID": os.environ.get("L3_ENTRY_MIN_SIZE_GRID", ""),
            "L3_ENTRY_POLICY_MIN_STATE_ROWS": os.environ.get("L3_ENTRY_POLICY_MIN_STATE_ROWS", ""),
            "L3_HORIZON_MIN_STATE_ROWS": os.environ.get("L3_HORIZON_MIN_STATE_ROWS", ""),
            "L3_TRAJ_GRU": os.environ.get("L3_TRAJ_GRU", ""),
            "L3_EXIT_STATE_GRANULARITY": os.environ.get("L3_EXIT_STATE_GRANULARITY", ""),
            "L3_VALUE_MODE": os.environ.get("L3_VALUE_MODE", ""),
            "L3_EXIT_INFER_SMOOTH": os.environ.get("L3_EXIT_INFER_SMOOTH", ""),
            "L3_MIN_HOLD_BARS": os.environ.get("L3_MIN_HOLD_BARS", ""),
            "L3_EXIT_BOOST_ROUNDS": os.environ.get("L3_EXIT_BOOST_ROUNDS", ""),
        },
        "df_hash": _hash_frame_columns(df, ["symbol", "time_key", "open", "high", "low", "close", "lbl_atr", *PA_STATE_FEATURES]),
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
        strict = os.environ.get("L3_STRICT_TRADE_LEVEL_SPLIT", "1").strip().lower() in {"1", "true", "yes"}
        if strict:
            raise RuntimeError(
                "L3: only one distinct trade in OOT window; strict trade-level split forbids intra-trade train/val leakage. "
                "Expand OOT window (e.g. set L3_OOT_START earlier) before training."
            )
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
    train_entries = set(np.unique(rows_entry[np.asarray(train_mask & oot, dtype=bool)]).tolist())
    val_entries = set(np.unique(rows_entry[np.asarray(val_mask & oot, dtype=bool)]).tolist())
    overlap = train_entries & val_entries
    if overlap:
        raise RuntimeError(
            f"L3 trade-level split violation: {len(overlap)} trade_id(s) appear in both train and val (e.g. {sorted(list(overlap))[:5]})."
        )
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


def _fit_l3_exit_calibrator(y_true: np.ndarray, raw_p: np.ndarray) -> Any:
    y = np.asarray(y_true, dtype=np.int32).ravel()
    p = np.clip(np.asarray(raw_p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    if y.size < 100 or len(np.unique(y)) < 2:
        return None
    mode = (os.environ.get("L3_EXIT_CALIB", "isotonic") or "isotonic").strip().lower()
    if mode in {"none", "off", "raw"}:
        return None
    if mode == "platt":
        logit_p = np.log(p / (1.0 - p))
        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(logit_p.reshape(-1, 1), y.astype(np.int32))
        return ("platt", clf)
    calib = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calib.fit(p, y.astype(np.float64))
    return ("isotonic", calib)


def _apply_l3_exit_calibrator(p: np.ndarray, calibrator: Any) -> np.ndarray:
    arr = np.clip(np.asarray(p, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    if calibrator is None:
        return arr
    if isinstance(calibrator, tuple) and len(calibrator) == 2:
        tag, obj = calibrator[0], calibrator[1]
        if tag == "platt" and isinstance(obj, LogisticRegression):
            logit_a = np.log(arr / (1.0 - arr))
            return np.clip(
                np.asarray(obj.predict_proba(logit_a.reshape(-1, 1))[:, 1], dtype=np.float64).ravel(),
                0.0,
                1.0,
            )
        if tag == "isotonic" and isinstance(obj, IsotonicRegression):
            return np.clip(np.asarray(obj.predict(arr), dtype=np.float64).ravel(), 0.0, 1.0)
    if isinstance(calibrator, IsotonicRegression):
        return np.clip(np.asarray(calibrator.predict(arr), dtype=np.float64).ravel(), 0.0, 1.0)
    return arr


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


def _env_int_clipped(key: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.environ.get(key, "").strip()
    val = default if not raw else int(raw)
    return int(np.clip(val, lo, hi))


def _l3_exit_class_weights(y_exit: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y_exit, dtype=np.int32).ravel()
    if y.size == 0:
        return 1.0, 1.0
    n_exit = max(int(np.sum(y == 1)), 1)
    n_hold = max(int(np.sum(y == 0)), 1)
    gamma = _env_float_clipped("L3_EXIT_CLASS_WEIGHT_GAMMA", 0.5, lo=0.25, hi=2.0)
    exit_pos_w = float(np.clip((n_hold / n_exit) ** gamma, 0.5, 4.0))
    hold_neg_w = float(np.clip((n_exit / n_hold) ** gamma, 0.5, 4.0))
    raw_h = os.environ.get("L3_HOLD_CLASS_WEIGHT", "2.5").strip().lower()
    if raw_h not in {"auto", "compute", "formula"}:
        hold_neg_w = float(np.clip(float(raw_h or "2.5"), 0.5, 5.0))
    return exit_pos_w, hold_neg_w


def _l3_prepare_value_targets(
    y_value: np.ndarray,
    train_mask: np.ndarray,
    pa_state: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, float | str | bool]]:
    y = np.asarray(y_value, dtype=np.float32).copy()
    train = np.asarray(train_mask, dtype=bool).ravel()
    finite_train = train & np.isfinite(y)
    if pa_state is not None and _l3_pa_targets_enabled():
        trend = np.asarray(pa_state.get("pa_state_trend_strength", np.zeros(len(y))), dtype=np.float64).ravel()
        follow = np.asarray(pa_state.get("pa_state_followthrough_quality", np.zeros(len(y))), dtype=np.float64).ravel()
        range_risk = np.asarray(pa_state.get("pa_state_range_risk", np.zeros(len(y))), dtype=np.float64).ravel()
        breakout = np.asarray(pa_state.get("pa_state_breakout_failure_risk", np.zeros(len(y))), dtype=np.float64).ravel()
        pullback = np.asarray(pa_state.get("pa_state_pullback_exhaustion", np.zeros(len(y))), dtype=np.float64).ravel()
        scale = np.clip(1.0 + 0.16 * trend + 0.08 * follow - 0.18 * range_risk - 0.16 * breakout - 0.08 * pullback, 0.75, 1.20)
        y = (np.asarray(y, dtype=np.float64) * scale).astype(np.float32)
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
    if pa_state is not None and _l3_pa_targets_enabled():
        stats["pa_target_scaling"] = True
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


def _l3_value_hurdle_epsilon(y_value: np.ndarray, train_mask: np.ndarray) -> float:
    env = os.environ.get("L3_VALUE_HURDLE_EPS", "").strip()
    if env:
        return float(np.clip(float(env), 0.0, 2.0))
    y = np.asarray(y_value, dtype=np.float64).ravel()
    tr = np.asarray(train_mask, dtype=bool).ravel() & np.isfinite(y)
    if not tr.any():
        return 0.05
    abs_y = np.abs(y[tr])
    q = float(np.quantile(abs_y, 0.70))
    return float(np.clip(max(0.02, 0.15 * q), 0.02, 0.30))


def _l3_hurdle_nonzero_weights(y_nonzero: np.ndarray) -> np.ndarray:
    y = np.asarray(y_nonzero, dtype=np.int32).ravel()
    w = np.ones(len(y), dtype=np.float32)
    if y.size == 0:
        return w
    n0 = max(int(np.sum(y == 0)), 1)
    n1 = max(int(np.sum(y == 1)), 1)
    total = n0 + n1
    w0 = float(np.clip(np.sqrt(total / (2.0 * n0)), 0.5, 3.0))
    w1 = float(np.clip(np.sqrt(total / (2.0 * n1)), 0.5, 3.0))
    w[y == 0] = w0
    w[y == 1] = w1
    return w


def _l3_value_predict_hurdle(
    X: np.ndarray,
    value_reg_model: lgb.Booster | None,
    value_nonzero_model: lgb.Booster | None,
    *,
    prob_power: float = 1.0,
) -> np.ndarray:
    if value_reg_model is None:
        return np.zeros(len(X), dtype=np.float64)
    mu = value_reg_model.predict(X).astype(np.float64)
    if value_nonzero_model is None:
        return mu
    p_nz = np.clip(value_nonzero_model.predict(X).astype(np.float64), 0.0, 1.0)
    pw = float(np.clip(prob_power, 0.5, 2.0))
    return mu * np.power(p_nz, pw)


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


def _l3_pa_policy_enabled() -> bool:
    return os.environ.get("L3_PA_POLICY_STATE", "1").strip().lower() in {"1", "true", "yes"}


def _l3_pa_targets_enabled() -> bool:
    return os.environ.get("L3_PA_TARGETS", "1").strip().lower() in {"1", "true", "yes"}


def _l3_pa_dict_from_frame(df: pd.DataFrame) -> dict[str, np.ndarray]:
    if not _l3_pa_targets_enabled():
        return {col: np.zeros(len(df), dtype=np.float32) for col in PA_STATE_FEATURES}
    return pa_state_arrays_from_frame(df)


def _l3_pa_dict_from_matrix(X: np.ndarray, feature_cols: list[str]) -> dict[str, np.ndarray]:
    if not _l3_pa_targets_enabled():
        return {col: np.zeros(len(X), dtype=np.float32) for col in PA_STATE_FEATURES}
    out: dict[str, np.ndarray] = {}
    for col in PA_STATE_FEATURES:
        if col in feature_cols:
            out[col] = np.asarray(X[:, feature_cols.index(col)], dtype=np.float32)
        else:
            out[col] = np.zeros(len(X), dtype=np.float32)
    return out


def _append_pa_bucket_to_state_keys(base_keys: np.ndarray, pa_bucket: np.ndarray | None) -> np.ndarray:
    if not _l3_pa_policy_enabled() or pa_bucket is None:
        return np.asarray(base_keys, dtype=object)
    bucket = np.asarray(pa_bucket, dtype=object).ravel()
    if bucket.size != len(base_keys):
        return np.asarray(base_keys, dtype=object)
    return np.asarray([f"{key}_pa_{pb}" for key, pb in zip(base_keys, bucket)], dtype=object)


def _state_keys_from_regime_vol(regime_probs: np.ndarray, vol_values: np.ndarray, *, vol_quantiles: list[float]) -> np.ndarray:
    reg = _regime_ids_from_probs(regime_probs)
    vb = _bucketize_by_quantiles(vol_values, vol_quantiles)
    return np.asarray([f"r{int(r)}_v{int(v)}" for r, v in zip(reg, vb)], dtype=object)


def _exit_coarse_bucket(regime_probs: np.ndarray) -> np.ndarray:
    """3 buckets: 0=bull (0–1), 1=range/vol (4–5), 2=bear/adverse (2–3)."""
    rid = _regime_ids_from_probs(regime_probs)
    sid = np.full(len(rid), 2, dtype=np.int32)
    sid[rid <= 1] = 0
    sid[rid >= 4] = 1
    return sid


def _exit_state_keys_from_regime_vol_hold(
    regime_probs: np.ndarray,
    vol_values: np.ndarray,
    hold_values: np.ndarray,
    *,
    vol_quantiles: list[float],
) -> np.ndarray:
    mode = os.environ.get("L3_EXIT_STATE_GRANULARITY", "coarse").strip().lower()
    hb = _hold_bucket_ids(hold_values)
    if mode in {"full", "legacy", "fine"}:
        base = _state_keys_from_regime_vol(regime_probs, vol_values, vol_quantiles=vol_quantiles)
        return np.asarray([f"{b}_h{int(h)}" for b, h in zip(base, hb)], dtype=object)
    sid = _exit_coarse_bucket(regime_probs)
    return np.asarray([f"ex{int(s)}_h{int(h)}" for s, h in zip(sid, hb)], dtype=object)


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


def _l3_exit_loss_buffer_atr() -> float:
    return float(max(0.0, float(os.environ.get("L3_EXIT_LOSS_BUFFER_ATR", "0.08"))))


def _l3_exit_live_edge_floor() -> float:
    return float(os.environ.get("L3_EXIT_LIVE_EDGE_FLOOR", "0.02"))


def _l3_continuation_score(future_gain_left: np.ndarray, live_edge: np.ndarray) -> np.ndarray:
    future = np.asarray(future_gain_left, dtype=np.float64).ravel()
    edge = np.asarray(live_edge, dtype=np.float64).ravel()
    edge_mult = float(os.environ.get("L3_CONTINUATION_EDGE_MULT", "0.20"))
    return (future + edge_mult * np.maximum(edge, 0.0)).astype(np.float32)


def _l3_target_horizon_bars(max_hold: int) -> int:
    raw = int(os.environ.get("L3_TARGET_HORIZON_BARS", "5"))
    return max(1, min(int(max_hold), raw))


def _l3_search_entry_policy(
    state_keys: np.ndarray,
    decision_class: np.ndarray,
    decision_confidence: np.ndarray,
    size: np.ndarray,
    edge_atr: np.ndarray,
    tau_edge: float,
    pa_state: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    cls = np.asarray(decision_class, dtype=np.int64).ravel()
    conf = np.asarray(decision_confidence, dtype=np.float64).ravel()
    sz = np.asarray(size, dtype=np.float64).ravel()
    edge = np.asarray(edge_atr, dtype=np.float64).ravel()
    keys = np.asarray(state_keys, dtype=object).ravel()
    trend = np.asarray((pa_state or {}).get("pa_state_trend_strength", np.zeros(len(keys))), dtype=np.float64).ravel()
    follow = np.asarray((pa_state or {}).get("pa_state_followthrough_quality", np.zeros(len(keys))), dtype=np.float64).ravel()
    range_risk = np.asarray((pa_state or {}).get("pa_state_range_risk", np.zeros(len(keys))), dtype=np.float64).ravel()
    breakout = np.asarray((pa_state or {}).get("pa_state_breakout_failure_risk", np.zeros(len(keys))), dtype=np.float64).ravel()
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
                pa_bonus = 0.0
                if pa_state is not None and np.any(entered):
                    pa_bonus = float(
                        0.08 * np.mean(trend[entered])
                        + 0.04 * np.mean(follow[entered])
                        - 0.10 * np.mean(range_risk[entered])
                        - 0.08 * np.mean(breakout[entered])
                    )
                score = 0.55 * precision + 0.30 * correct_side + 0.20 * avg_abs_edge - 0.20 * abs(trade_rate - active_rate_target) + pa_bonus
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
    pa_state: dict[str, np.ndarray] | None = None,
) -> tuple[int, dict[str, int]]:
    base = _l3_target_horizon_bars(max_hold)
    keys = np.asarray(state_keys, dtype=object).ravel()
    peak = np.asarray(peak_bar, dtype=np.float64).ravel()
    if pa_state is not None and _l3_pa_targets_enabled():
        trend = np.asarray(pa_state.get("pa_state_trend_strength", np.zeros(len(peak))), dtype=np.float64).ravel()
        follow = np.asarray(pa_state.get("pa_state_followthrough_quality", np.zeros(len(peak))), dtype=np.float64).ravel()
        range_risk = np.asarray(pa_state.get("pa_state_range_risk", np.zeros(len(peak))), dtype=np.float64).ravel()
        breakout = np.asarray(pa_state.get("pa_state_breakout_failure_risk", np.zeros(len(peak))), dtype=np.float64).ravel()
        pullback = np.asarray(pa_state.get("pa_state_pullback_exhaustion", np.zeros(len(peak))), dtype=np.float64).ravel()
        horizon_scale = np.clip(1.0 + 0.18 * trend + 0.10 * follow - 0.22 * range_risk - 0.18 * breakout - 0.10 * pullback, 0.55, 1.30)
        peak = peak * horizon_scale
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


def _l3_filter_cox_covariates(
    df_ctv: pd.DataFrame,
    cov_names: list[str],
) -> tuple[list[str], dict[str, str]]:
    dropped: dict[str, str] = {}
    if not cov_names or df_ctv.empty or "event" not in df_ctv.columns:
        return cov_names, dropped
    evt = pd.to_numeric(df_ctv["event"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    var_eps = float(os.environ.get("L3_COX_VAR_EPS", "1e-6"))
    sep_auc_thr = float(os.environ.get("L3_COX_SEPARATION_AUC_THR", "0.995"))
    keep: list[str] = []
    for col in cov_names:
        vals = pd.to_numeric(df_ctv[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        sd = float(np.std(vals))
        if not np.isfinite(sd) or sd < var_eps:
            dropped[col] = f"low_variance(std={sd:.2e})"
            continue
        if len(np.unique(evt)) >= 2:
            try:
                auc = float(roc_auc_score(evt.astype(np.int32), vals))
            except ValueError:
                auc = float("nan")
            sep_score = max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan")
            if np.isfinite(sep_score) and sep_score >= sep_auc_thr:
                dropped[col] = f"near_separation(auc={auc:.4f})"
                continue
        keep.append(col)
    return keep, dropped


def _l3_cox_stabilize_drawdown_feature(
    df_ctv: pd.DataFrame,
    X_cov: np.ndarray,
    cov_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float | int | bool]]:
    meta: dict[str, float | int | bool] = {"applied": False}
    if "l3_drawdown_from_peak_atr" not in cov_names:
        return df_ctv, X_cov, meta
    col = "l3_drawdown_from_peak_atr"
    idx = cov_names.index(col)
    vals_df = pd.to_numeric(df_ctv[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    vals_x = np.asarray(X_cov[:, idx], dtype=np.float64)
    winsor_q = _env_float_clipped("L3_COX_DRAWDOWN_WINSOR_Q", 0.0, lo=0.0, hi=0.20)
    if winsor_q > 0.0 and vals_df.size:
        lo = float(np.quantile(vals_df, winsor_q))
        hi = float(np.quantile(vals_df, 1.0 - winsor_q))
        vals_df = np.clip(vals_df, lo, hi)
        vals_x = np.clip(vals_x, lo, hi)
        meta.update({"winsor_q": float(winsor_q), "winsor_lo": lo, "winsor_hi": hi, "applied": True})
    bin_count = int(_env_int_clipped("L3_COX_DRAWDOWN_BIN_COUNT", 1, lo=1, hi=20))
    if bin_count > 1 and vals_df.size:
        edges = np.quantile(vals_df, np.linspace(0.0, 1.0, bin_count + 1))
        edges = np.unique(np.asarray(edges, dtype=np.float64))
        if len(edges) >= 3:
            div = max(len(edges) - 1, 1)
            vals_df = np.digitize(vals_df, edges[1:-1], right=False).astype(np.float64) / float(div)
            vals_x = np.digitize(vals_x, edges[1:-1], right=False).astype(np.float64) / float(div)
            meta.update({"bin_count": int(len(edges) - 1), "applied": True})
    df_new = df_ctv.copy()
    X_new = np.asarray(X_cov, dtype=np.float64).copy()
    df_new[col] = vals_df
    X_new[:, idx] = vals_x
    return df_new, X_new, meta


def _l3_gain_select_required_names(feature_cols: list[str], cox_bundle: Mapping[str, Any]) -> set[str]:
    """Columns that must stay in LGBM matrices for indexing, Cox alignment, or inference."""
    req: set[str] = {"l3_hold_bars", "l1a_vol_forecast", *L1A_REGIME_COLS}
    for cox_c in cox_bundle.get("cov_names") or []:
        if isinstance(cox_c, str) and cox_c in feature_cols:
            req.add(cox_c)
    for cox_out in ("l3_cox_log_partial_hazard", "l3_cox_baseline_cumhaz_at_stop"):
        if cox_out in feature_cols:
            req.add(cox_out)
    return req


def _l3_maybe_prune_features_by_exit_gain(
    *,
    X_lgb: np.ndarray,
    X: np.ndarray,
    feature_cols: list[str],
    static_cols: list[str],
    use_hybrid: bool,
    emb_matrix: np.ndarray | None,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    y_exit: np.ndarray,
    rows_entry: np.ndarray,
    pa_state_all: dict[str, np.ndarray],
    exit_params: dict[str, Any],
    es_rounds: int,
    cox_bundle: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, Any]]:
    """Train a short exit probe; keep features with >= min share of total gain (default 1%); cap/floor count."""
    meta: dict[str, Any] = {"enabled": True, "pruned": False}
    disabled = os.environ.get("L3_FEATURE_GAIN_SELECT", "1").strip().lower() in {"0", "false", "no"}
    if disabled:
        meta["enabled"] = False
        return X_lgb, X, feature_cols, static_cols, meta

    min_gain = float(np.clip(float(os.environ.get("L3_FEATURE_GAIN_MIN_FRAC", "0.01")), 1e-6, 0.5))
    max_n = int(_env_int_clipped("L3_FEATURE_GAIN_MAX", 25, lo=5, hi=500))
    min_n = int(_env_int_clipped("L3_FEATURE_GAIN_MIN", 20, lo=5, hi=max_n))
    probe_rounds = int(_env_int_clipped("L3_FEATURE_PROBE_ROUNDS", 120, lo=20, hi=5000))

    fc = list(feature_cols)
    n = len(fc)
    if n <= min_n:
        meta["skip_reason"] = "already_small"
        return X_lgb, X, fc, static_cols, meta

    ih = fc.index("l3_hold_bars")
    hold_tr = X_lgb[np.asarray(train_mask, dtype=bool), ih].astype(np.float64)
    w = _l3_trade_normalized_exit_weights(
        rows_entry[np.asarray(train_mask, dtype=bool)],
        hold_tr,
        y_exit[np.asarray(train_mask, dtype=bool)],
        pa_state={k: v[np.asarray(train_mask, dtype=bool)] for k, v in pa_state_all.items()},
    )
    cbs, cl = _lgb_train_callbacks_with_round_tqdm(es_rounds, probe_rounds, "[L3] gain-probe")
    try:
        probe = lgb.train(
            dict(exit_params),
            lgb.Dataset(
                X_lgb[np.asarray(train_mask, dtype=bool)],
                label=y_exit[np.asarray(train_mask, dtype=bool)],
                weight=w.astype(np.float32),
                feature_name=fc,
                free_raw_data=False,
            ),
            num_boost_round=probe_rounds,
            valid_sets=[
                lgb.Dataset(
                    X_lgb[np.asarray(val_mask, dtype=bool)],
                    label=y_exit[np.asarray(val_mask, dtype=bool)],
                    feature_name=fc,
                    free_raw_data=False,
                )
            ],
            callbacks=cbs,
        )
    finally:
        for fn in cl:
            fn()

    gains = np.asarray(probe.feature_importance(importance_type="gain"), dtype=np.float64)
    gsum = float(np.sum(gains))
    if not np.isfinite(gsum) or gsum <= 0:
        meta["skip_reason"] = "zero_gain"
        return X_lgb, X, fc, static_cols, meta

    frac = gains / gsum
    req_names = _l3_gain_select_required_names(fc, cox_bundle)
    always_idx: set[int] = set()
    for name in req_names:
        if name in fc:
            always_idx.add(fc.index(name))
    if len(always_idx) > max_n:
        max_n = len(always_idx)
        meta["max_n_bumped_to_required"] = int(max_n)

    optional_idx = [i for i in range(n) if i not in always_idx]
    selected: set[int] = set(always_idx)
    for i in optional_idx:
        if float(frac[i]) >= min_gain:
            selected.add(i)

    if len(selected) > max_n:
        opt_sorted = sorted((i for i in selected if i not in always_idx), key=lambda i: float(frac[i]))
        for i in opt_sorted:
            if len(selected) <= max_n:
                break
            selected.remove(i)

    if len(selected) < min_n:
        rest = [i for i in range(n) if i not in selected]
        rest.sort(key=lambda i: float(frac[i]), reverse=True)
        for i in rest:
            selected.add(i)
            if len(selected) >= min_n:
                break

    sel_idx = sorted(selected)
    if sel_idx == list(range(n)):
        meta["pruned"] = False
        return X_lgb, X, fc, static_cols, meta

    static_new = [fc[i] for i in sel_idx if not fc[i].startswith("l3_traj_emb_")]
    try:
        X_new = X[:, [static_cols.index(fc[i]) for i in sel_idx if not fc[i].startswith("l3_traj_emb_")]].astype(
            np.float32, copy=False
        )
    except ValueError as ex:
        meta["skip_reason"] = f"static_index:{ex}"
        return X_lgb, X, fc, static_cols, meta

    if use_hybrid and emb_matrix is not None:
        emb_ix = [int(fc[i].rsplit("_", 1)[-1]) for i in sel_idx if fc[i].startswith("l3_traj_emb_")]
        if emb_ix:
            X_lgb_new = np.hstack(
                [X_new, emb_matrix[:, emb_ix].astype(np.float32, copy=False)],
                dtype=np.float32,
            )
            feature_out = static_new + [fc[i] for i in sel_idx if fc[i].startswith("l3_traj_emb_")]
        else:
            X_lgb_new = X_new
            feature_out = static_new
    else:
        X_lgb_new = X_new
        feature_out = static_new

    dropped = [fc[j] for j in range(n) if j not in set(sel_idx)]
    meta.update(
        {
            "pruned": True,
            "n_before": n,
            "n_after": len(feature_out),
            "min_gain_frac": min_gain,
            "max_n": max_n,
            "min_n": min_n,
            "probe_rounds": probe_rounds,
            "dropped": dropped,
            "kept": list(feature_out),
        }
    )
    print(
        f"  [L3] gain feature select: {n}->{len(feature_out)} cols "
        f"(gain>={min_gain:.2%} of total; cap={max_n} floor={min_n}; probe_rounds={probe_rounds})",
        flush=True,
    )
    return X_lgb_new, X_new, feature_out, static_new, meta


def _l3_exit_policy_row_state_keys(X: np.ndarray, feature_cols: list[str], *, vol_quantiles: list[float]) -> np.ndarray:
    reg_cols = [feature_cols.index(c) for c in L1A_REGIME_COLS]
    vol_idx = feature_cols.index("l1a_vol_forecast")
    hold_idx = feature_cols.index("l3_hold_bars")
    regime_probs = np.asarray(X[:, reg_cols], dtype=np.float32)
    vol_values = np.asarray(X[:, vol_idx], dtype=np.float32)
    hold_values = np.asarray(X[:, hold_idx], dtype=np.float32)
    base = _exit_state_keys_from_regime_vol_hold(regime_probs, vol_values, hold_values, vol_quantiles=vol_quantiles)
    pa_bucket = None
    if _l3_pa_policy_enabled() and all(col in feature_cols for col in PA_STATE_FEATURES):
        pa_bucket = pa_state_bucket_labels_from_arrays(
            {col: np.asarray(X[:, feature_cols.index(col)], dtype=np.float32) for col in PA_STATE_FEATURES},
            length=len(X),
        )
    return _append_pa_bucket_to_state_keys(base, pa_bucket)


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return float("nan")
    if weights is None:
        return float(np.mean(v))
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != v.size:
        return float(np.mean(v))
    w = np.clip(w, 0.0, np.inf)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        return float(np.mean(v))
    return float(np.sum(v * w) / s)


def _weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray | None = None) -> float:
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return float("nan")
    qq = float(np.clip(q, 0.0, 1.0))
    if weights is None:
        return float(np.quantile(v, qq))
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != v.size:
        return float(np.quantile(v, qq))
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if int(np.sum(ok)) == 0:
        return float(np.quantile(v[np.isfinite(v)], qq)) if np.isfinite(v).any() else float("nan")
    v_ok = v[ok]
    w_ok = w[ok]
    order = np.argsort(v_ok, kind="mergesort")
    v_s = v_ok[order]
    w_s = w_ok[order]
    cdf = np.cumsum(w_s)
    if cdf[-1] <= 0:
        return float(np.quantile(v_ok, qq))
    cdf = cdf / cdf[-1]
    idx = int(np.searchsorted(cdf, qq, side="left"))
    idx = min(max(idx, 0), len(v_s) - 1)
    return float(v_s[idx])


def _estimate_half_life_ar1(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float("nan")
    x_lag = x[:-1]
    y = x[1:]
    if np.std(x_lag) <= 1e-8 or np.std(y) <= 1e-8:
        return float("nan")
    rho = float(np.corrcoef(x_lag, y)[0, 1])
    if not np.isfinite(rho) or rho <= 0.0 or abs(rho) < 1e-6 or abs(rho) >= 0.999:
        return float("nan")
    return float(-np.log(2.0) / np.log(abs(rho)))


def _l3_exp_decay_weights(
    tune_idx: np.ndarray,
    t_state: pd.Series,
    signal: np.ndarray,
    *,
    min_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = int(tune_idx.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64), {"mode": "empty"}
    if n < min_rows:
        w = np.ones(n, dtype=np.float64) / max(n, 1)
        return w, {"mode": "uniform_fallback", "reason": f"rows<{min_rows}", "half_life_rows": float("nan")}
    ts = pd.to_datetime(np.asarray(t_state)[tune_idx], errors="coerce")
    sig = np.asarray(signal, dtype=np.float64).ravel()
    sig = sig[tune_idx]
    valid = np.isfinite(sig) & pd.notna(ts)
    if int(np.sum(valid)) < max(10, min_rows // 4):
        w = np.ones(n, dtype=np.float64) / max(n, 1)
        return w, {"mode": "uniform_fallback", "reason": "insufficient_valid_signal", "half_life_rows": float("nan")}
    day = pd.to_datetime(ts[valid]).floor("D")
    day_signal = pd.Series(sig[valid]).groupby(day).mean().to_numpy(dtype=np.float64)
    hl_days = _estimate_half_life_ar1(day_signal)
    if not np.isfinite(hl_days) or hl_days <= 0:
        w = np.ones(n, dtype=np.float64) / max(n, 1)
        return w, {"mode": "uniform_fallback", "reason": "unstable_half_life", "half_life_rows": float("nan")}
    unique_days = max(1, int(len(np.unique(day))))
    rows_per_day = max(1.0, float(n) / float(unique_days))
    hl_rows = float(np.clip(hl_days * rows_per_day, 1.0, max(1.0, n * 1.5)))
    lambda_ = float(np.log(2.0) / max(hl_rows, 1e-6))
    age = np.arange(n, dtype=np.float64)[::-1]
    w = np.exp(-lambda_ * age)
    w = w / max(float(np.sum(w)), 1e-12)
    return w, {"mode": "exp_decay", "half_life_days": float(hl_days), "half_life_rows": float(hl_rows), "lambda": lambda_}


def _l3_search_exit_policy(
    exit_prob: np.ndarray,
    value_pred: np.ndarray,
    y_exit: np.ndarray,
    *,
    y_value_true: np.ndarray | None = None,
    value_policy_mode: str | None = None,
    value_tie_margin: float | None = None,
    sample_weight: np.ndarray | None = None,
    hold_bars: np.ndarray | None = None,
    report_guardrail: Mapping[str, float] | None = None,
) -> dict[str, float]:
    prob = np.asarray(exit_prob, dtype=np.float64).ravel()
    value = np.asarray(value_pred, dtype=np.float64).ravel()
    y = np.asarray(y_exit, dtype=np.int32).ravel()
    valid = np.isfinite(prob) & np.isfinite(value)
    mode = str(value_policy_mode or "prob_only").strip().lower() or "prob_only"
    tie_margin = float(max(0.0, float(value_tie_margin if value_tie_margin is not None else 0.03)))
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
    value_true = None if y_value_true is None else np.asarray(y_value_true, dtype=np.float64).ravel()[valid]
    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype=np.float64).ravel()[valid]
        sw = np.clip(sw, 0.0, np.inf)
    else:
        sw = None
    if value_true is not None and value_true.size:
        bias = float(np.median(value_true - value))
        value_adj = value + bias
    else:
        value_adj = value
    value_pred_std = float(np.std(value))
    value_target_std = float(np.std(value_true)) if value_true is not None and value_true.size else float(np.std(value_adj))
    policy_params = derive_policy_params(
        y.astype(np.float64),
        value_true if value_true is not None and value_true.size else value_adj,
        value_pred_std=value_pred_std,
        value_target_std=value_target_std,
    )

    q_lo = _env_float_clipped("L3_POLICY_PROB_Q_LO", 0.45, lo=0.05, hi=0.80)
    q_hi = _env_float_clipped("L3_POLICY_PROB_Q_HI", 0.97, lo=0.55, hi=0.995)
    n_q = _env_int_clipped("L3_POLICY_PROB_Q_N", 19, lo=5, hi=50)
    q_prob = np.linspace(q_lo, q_hi, n_q)
    q_value = np.linspace(0.20, 0.85, 14)
    prob_candidates = np.unique(np.clip(np.quantile(prob, q_prob), 0.05, 0.95))
    value_candidates = (
        np.unique(np.quantile(value_adj, q_value))
        if value_adj.size
        else np.asarray([0.0], dtype=np.float64)
    )
    if value_true is not None and value_true.size:
        util_scale = float(np.median(np.abs(value_true[np.isfinite(value_true)]))) if np.isfinite(value_true).any() else 1.0
        util_scale = max(util_scale, 1e-3)
    else:
        util_scale = 1.0
    target_exit_rate = float(os.environ.get("L3_POLICY_TARGET_EXIT_RATE", f"{policy_params.target_exit_rate:.4f}"))
    exit_rate_penalty = float(
        os.environ.get("L3_POLICY_UTILITY_EXIT_RATE_PENALTY", f"{policy_params.exit_rate_penalty:.4f}")
    )
    hold_recall_w = float(os.environ.get("L3_POLICY_HOLD_RECALL_WEIGHT", f"{policy_params.hold_recall_w:.4f}"))
    hold_recall_floor = float(os.environ.get("L3_POLICY_MIN_HOLD_RECALL", f"{policy_params.hold_recall_floor:.4f}"))
    exit_rate_cap = float(
        os.environ.get(
            "L3_POLICY_MAX_EXIT_RATE",
            f"{float(np.clip(target_exit_rate + _env_float_clipped('L3_POLICY_EXIT_RATE_CAP_BUFFER', 0.08, lo=0.01, hi=0.30), 0.60, 0.95)):.4f}",
        )
    )
    report_exit_hint = float("nan")
    report_hold_floor_hint = float("nan")
    if report_guardrail is not None:
        report_exit_hint = float(report_guardrail.get("exit_rate_hint", float("nan")))
        report_hold_floor_hint = float(report_guardrail.get("hold_recall_floor_hint", float("nan")))
        if np.isfinite(report_hold_floor_hint):
            hold_recall_floor = max(hold_recall_floor, float(np.clip(report_hold_floor_hint, 0.10, 0.75)))
        if np.isfinite(report_exit_hint):
            exit_rate_cap = min(exit_rate_cap, float(np.clip(max(target_exit_rate + 0.02, report_exit_hint + 0.05), 0.60, 0.95)))
    hold_arr = np.zeros_like(prob) if hold_bars is None else np.asarray(hold_bars, dtype=np.float64).ravel()[valid]
    early_split_enabled = os.environ.get("L3_POLICY_SPLIT_EARLY_HOLD", "1").strip().lower() in {"1", "true", "yes"}
    early_hold_bar = int(_env_int_clipped("L3_POLICY_EARLY_HOLD_BARS", 3, lo=1, hi=30))
    early_delta_candidates = (
        _env_float_candidates("L3_POLICY_EARLY_PROB_DELTA_CANDIDATES", [-0.08, -0.04, 0.0, 0.04], lo=-0.30, hi=0.30)
        if early_split_enabled
        else [0.0]
    )
    miss_exit_penalty = float(_env_float_clipped("L3_POLICY_UTILITY_MISS_EXIT_PENALTY", 0.20, lo=0.0, hi=2.0))
    utility_tail_weight = float(_env_float_clipped("L3_POLICY_UTILITY_TAIL_WEIGHT", 0.15, lo=0.0, hi=1.0))
    best: dict[str, float] | None = None
    best_relaxed: dict[str, float] | None = None
    best_under_cap: dict[str, float] | None = None
    best_under_floor: dict[str, float] | None = None
    n_total = 0
    n_under_cap = 0
    n_under_floor = 0
    n_passing = 0
    for prob_thr in prob_candidates.tolist():
        local_values = [0.0] if mode == "prob_only" else value_candidates.tolist()
        for value_thr in local_values:
            for early_delta in early_delta_candidates:
                n_total += 1
                prob_thr_late = float(prob_thr)
                prob_thr_early = float(np.clip(prob_thr + early_delta, 0.05, 0.95))
                thr_vec = np.where(hold_arr < float(early_hold_bar), prob_thr_early, prob_thr_late)
                if mode == "hard_gate":
                    pred = ((prob >= thr_vec) & (value_adj <= value_thr)).astype(np.int32)
                elif mode == "tie_break":
                    lower_vec = np.maximum(0.0, thr_vec - tie_margin)
                    pred = (
                        (prob >= thr_vec)
                        | (((prob >= lower_vec) & (prob < thr_vec)) & (value_adj <= value_thr))
                    ).astype(np.int32)
                else:
                    pred = (prob >= thr_vec).astype(np.int32)
                f1 = float(f1_score(y, pred, zero_division=0))
                acc = float(accuracy_score(y, pred))
                exit_rate = float(_weighted_mean(pred.astype(np.float64), sw))
                hold_mask = y == 0
                pred_hold = pred == 0
                hold_recall = (
                    float(_weighted_mean((pred[hold_mask] == 0).astype(np.float64), None if sw is None else sw[hold_mask]))
                    if hold_mask.any()
                    else 0.0
                )
                hold_precision = (
                    float(_weighted_mean((y[pred_hold] == 0).astype(np.float64), None if sw is None else sw[pred_hold]))
                    if pred_hold.any()
                    else 0.0
                )
                exit_recall = (
                    float(_weighted_mean((pred[y == 1] == 1).astype(np.float64), None if sw is None else sw[y == 1]))
                    if np.any(y == 1)
                    else 0.0
                )
                miss_exit = float(_weighted_mean(((y == 1) & (pred == 0)).astype(np.float64), sw))
                utility_mean = float("nan")
                utility_p10 = float("nan")
                utility_score = 0.0
                if value_true is not None and value_true.size:
                    # Exit realizes current PnL and gives up continuation value; hold keeps continuation value.
                    realized = np.where(pred == 1, -value_true, value_true)
                    utility_mean = float(_weighted_mean(realized, sw))
                    utility_p10 = float(_weighted_quantile(realized, 0.10, sw))
                    utility_score = (
                        (utility_mean / util_scale)
                        + utility_tail_weight * (utility_p10 / util_scale)
                        - exit_rate_penalty * max(0.0, exit_rate - target_exit_rate)
                        - miss_exit_penalty * miss_exit
                    )
                hold_recall_contrib = hold_recall_w * hold_recall
                exit_rate_penalty_contrib = exit_rate_penalty * max(0.0, exit_rate - target_exit_rate)
                score = (
                    utility_score
                    + 0.10 * f1
                    + hold_recall_contrib
                    + 0.08 * hold_precision
                    - 0.10 * max(0.0, hold_recall_floor - hold_recall)
                )
                cand = {
                    "exit_prob_threshold": float(prob_thr),
                    "value_left_threshold": float(value_thr),
                    "exit_prob_threshold_early": float(prob_thr_early),
                    "exit_prob_threshold_late": float(prob_thr_late),
                    "early_prob_threshold_delta": float(early_delta),
                    "early_hold_split_bar": int(early_hold_bar),
                    "score": float(score),
                    "utility_mean": float(utility_mean),
                    "utility_p10": float(utility_p10),
                    "f1": float(f1),
                    "acc": float(acc),
                    "exit_rate": float(exit_rate),
                    "target_exit_rate": float(target_exit_rate),
                    "max_exit_rate": float(exit_rate_cap),
                    "min_hold_recall": float(hold_recall_floor),
                    "hold_recall": float(hold_recall),
                    "hold_precision": float(hold_precision),
                    "exit_recall": float(exit_recall),
                    "value_policy_mode": mode,
                    "value_tie_margin": tie_margin,
                    "hold_recall_contrib": float(hold_recall_contrib),
                    "exit_rate_penalty_contrib": float(exit_rate_penalty_contrib),
                }
                if best_relaxed is None or float(cand["score"]) > float(best_relaxed["score"]):
                    best_relaxed = cand
                if exit_rate <= exit_rate_cap:
                    n_under_cap += 1
                    if best_under_cap is None or float(cand["score"]) > float(best_under_cap["score"]):
                        best_under_cap = cand
                if hold_recall >= hold_recall_floor:
                    n_under_floor += 1
                    if best_under_floor is None or float(cand["score"]) > float(best_under_floor["score"]):
                        best_under_floor = cand
                if hold_recall < hold_recall_floor:
                    continue
                if exit_rate > exit_rate_cap:
                    continue
                n_passing += 1
                if best is None or float(cand["score"]) > float(best["score"]):
                    best = cand
    if best is None:
        fallback_reason = "best_relaxed"
        fallback = best_relaxed
        if best_under_cap is not None:
            fallback_reason = "exit_rate_cap"
            fallback = best_under_cap
        elif best_under_floor is not None:
            fallback_reason = "hold_recall_floor"
            fallback = best_under_floor
        print(
            f"  [L3][warn] EXIT POLICY FALLBACK: no candidate met guardrails "
            f"(hold_recall>={hold_recall_floor:.4f}, exit_rate<={exit_rate_cap:.4f}); "
            f"fallback={fallback_reason}.",
            flush=True,
        )
        best = dict(fallback) if fallback is not None else {
            "exit_prob_threshold": 0.55,
            "value_left_threshold": 0.02,
            "exit_prob_threshold_early": 0.55,
            "exit_prob_threshold_late": 0.55,
            "early_prob_threshold_delta": 0.0,
            "early_hold_split_bar": int(early_hold_bar),
            "value_policy_mode": mode,
            "value_tie_margin": tie_margin,
            "score": float("nan"),
            "utility_mean": float("nan"),
            "utility_p10": float("nan"),
            "f1": float("nan"),
            "acc": float("nan"),
            "exit_rate": float("nan"),
            "target_exit_rate": float(target_exit_rate),
            "max_exit_rate": float(exit_rate_cap),
            "min_hold_recall": float(hold_recall_floor),
            "hold_recall": float("nan"),
            "hold_precision": float("nan"),
            "exit_recall": float("nan"),
            "hold_recall_contrib": float("nan"),
            "exit_rate_penalty_contrib": float("nan"),
        }
    print("\n  [L3] exit policy derivation on val_tune", flush=True)
    print(
        f"  [L3] derived policy params: exit_rate_penalty={exit_rate_penalty:.4f}  hold_recall_w={hold_recall_w:.4f}  "
        f"hold_recall_floor={hold_recall_floor:.4f}  target_exit_rate={target_exit_rate:.4f}  "
        f"exit_rate_cap={exit_rate_cap:.4f}  report_exit_hint={report_exit_hint:.4f}  "
        f"report_hold_floor_hint={report_hold_floor_hint:.4f}  split_early_hold={early_split_enabled}({early_hold_bar})  "
        f"diag_hold_rate={policy_params.diag_hold_rate:.4f}  diag_opp_cost={policy_params.diag_opp_cost:.4f}  "
        f"diag_save_benefit={policy_params.diag_save_benefit:.4f}  diag_value_head_degen={policy_params.diag_value_head_degen}",
        flush=True,
    )
    if best_relaxed is not None:
        print(
            f"  [L3] unconstrained best: p_exit>={best_relaxed['exit_prob_threshold']:.4f}  "
            f"exit_rate={best_relaxed.get('exit_rate', float('nan')):.3f}  "
            f"hold_recall={best_relaxed.get('hold_recall', float('nan')):.4f}  score={best_relaxed.get('score', float('nan')):.4f}",
            flush=True,
        )
    print(
        f"  [L3] selected exit policy: mode={best['value_policy_mode']}  p_exit>={best['exit_prob_threshold']:.4f}  "
        f"value_left<={best['value_left_threshold']:.4f}  utility_mean={best.get('utility_mean', float('nan')):.4f}  "
        f"utility_p10={best.get('utility_p10', float('nan')):.4f}  F1={best.get('f1', float('nan')):.4f}  acc={best.get('acc', float('nan')):.4f}  "
        f"exit_rate={best.get('exit_rate', float('nan')):.3f}  hold_recall={best.get('hold_recall', float('nan')):.4f}  "
        f"hold_precision={best.get('hold_precision', float('nan')):.4f}  "
        f"p_exit_early={best.get('exit_prob_threshold_early', float('nan')):.4f}  "
        f"p_exit_late={best.get('exit_prob_threshold_late', float('nan')):.4f}  "
        f"early_delta={best.get('early_prob_threshold_delta', float('nan')):+.4f}  "
        f"exit_rate_penalty_contrib={best.get('exit_rate_penalty_contrib', float('nan')):.4f}  "
        f"hold_recall_contrib={best.get('hold_recall_contrib', float('nan')):.4f}  "
        f"candidates_passing_floor={n_under_floor}/{n_total}  "
        f"candidates_under_cap={n_under_cap}/{n_total}  "
        f"candidates_guarded={n_passing}/{n_total}",
        flush=True,
    )
    return dict(best)


def _l3_search_conditional_exit_policy(
    state_keys: np.ndarray,
    exit_prob: np.ndarray,
    value_pred: np.ndarray,
    y_exit: np.ndarray,
    *,
    y_value_true: np.ndarray | None = None,
    value_policy_mode: str | None = None,
    value_tie_margin: float | None = None,
    sample_weight: np.ndarray | None = None,
    hold_bars: np.ndarray | None = None,
    report_guardrail: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    global_policy = _l3_search_exit_policy(
        exit_prob,
        value_pred,
        y_exit,
        y_value_true=y_value_true,
        value_policy_mode=value_policy_mode,
        value_tie_margin=value_tie_margin,
        sample_weight=sample_weight,
        hold_bars=hold_bars,
        report_guardrail=report_guardrail,
    )
    keys = np.asarray(state_keys, dtype=object).ravel()
    valid = np.isfinite(np.asarray(exit_prob, dtype=np.float64).ravel()) & np.isfinite(np.asarray(value_pred, dtype=np.float64).ravel())
    keys = keys[valid]
    by_state: dict[str, dict[str, float]] = {}
    min_rows = max(80, int(os.environ.get("L3_EXIT_POLICY_MIN_STATE_ROWS", "300")))
    for key in sorted({str(k) for k in keys.tolist()}):
        m = keys == key
        if int(np.sum(m)) < min_rows:
            continue
        by_state[key] = _l3_search_exit_policy(
            np.asarray(exit_prob, dtype=np.float64).ravel()[valid][m],
            np.asarray(value_pred, dtype=np.float64).ravel()[valid][m],
            np.asarray(y_exit, dtype=np.int32).ravel()[valid][m],
            y_value_true=None if y_value_true is None else np.asarray(y_value_true, dtype=np.float64).ravel()[valid][m],
            value_policy_mode=value_policy_mode,
            value_tie_margin=value_tie_margin,
            sample_weight=None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).ravel()[valid][m],
            hold_bars=None if hold_bars is None else np.asarray(hold_bars, dtype=np.float64).ravel()[valid][m],
            report_guardrail=report_guardrail,
        )
    print(f"  [L3] conditional exit states learned={len(by_state)}", flush=True)
    return global_policy, by_state


def _choose_l3_value_policy_mode(
    y_true: np.ndarray,
    pred: np.ndarray,
) -> str:
    if (os.environ.get("L3_VALUE_MODE", "") or "").strip().lower() == "disabled":
        print("  [L3] value-policy mode forced prob_only (L3_VALUE_MODE=disabled)", flush=True)
        return "prob_only"
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
    mode = "tie_break" if std_pred >= 0.02 and corr >= 0.03 and (np.isnan(r2) or r2 >= -0.05) else "prob_only"
    print(
        f"  [L3] value-policy mode selector: corr={corr:.4f}  r2={r2:.4f}  pred_std={std_pred:.6f}  "
        f"thresholds(corr>=0.03, pred_std>=0.02, r2>=-0.05)  -> {mode}",
        flush=True,
    )
    return mode


def _derive_l3_value_tie_margin(y_true: np.ndarray, pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(pred, dtype=np.float64).ravel()
    valid = np.isfinite(yt) & np.isfinite(yp)
    if not valid.any():
        return 0.03
    resid = np.abs(yt[valid] - yp[valid])
    return float(np.clip(np.median(resid), 0.0, 0.25))


def _l3_trade_normalized_exit_weights(
    rows_entry: np.ndarray,
    hold_bars: np.ndarray,
    y_exit: np.ndarray,
    pa_state: dict[str, np.ndarray] | None = None,
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
    hb_counts = np.bincount(hb.astype(np.int32), minlength=4).astype(np.float64)
    hb_counts = np.maximum(hb_counts, 1.0)
    hold_w = np.asarray([float(np.clip(np.sqrt(np.mean(hb_counts) / hb_counts[int(h)]), 0.75, 1.5)) for h in hb], dtype=np.float64)
    exit_pos_w, hold_neg_w = _l3_exit_class_weights(y)
    cls_w = np.where(y == 1, exit_pos_w, hold_neg_w).astype(np.float64)
    w = trade_norm * hold_w * cls_w
    if pa_state is not None and _l3_pa_targets_enabled():
        range_risk = np.asarray(pa_state.get("pa_state_range_risk", np.zeros(len(w))), dtype=np.float64).ravel()
        breakout = np.asarray(pa_state.get("pa_state_breakout_failure_risk", np.zeros(len(w))), dtype=np.float64).ravel()
        trend = np.asarray(pa_state.get("pa_state_trend_strength", np.zeros(len(w))), dtype=np.float64).ravel()
        pullback = np.asarray(pa_state.get("pa_state_pullback_exhaustion", np.zeros(len(w))), dtype=np.float64).ravel()
        pa_w = np.clip(1.0 + 0.24 * range_risk + 0.18 * breakout + 0.10 * pullback - 0.10 * trend, 0.70, 1.80)
        w *= pa_w
    w = w / max(float(np.mean(w)), 1e-8)
    return w.astype(np.float32)


def _l3_boost_rounds() -> int:
    return _env_int_clipped("L3_BOOST_ROUNDS", 250, lo=50, hi=5000)


def _l3_exit_boost_rounds() -> int:
    return _env_int_clipped("L3_EXIT_BOOST_ROUNDS", 300, lo=50, hi=5000)


def _l3_early_stopping_rounds() -> int:
    return _env_int_clipped("L3_EARLY_STOPPING_ROUNDS", 80, lo=5, hi=1000)


def _l3_lgb_params(prefix: str, *, seed_default: int) -> dict[str, Any]:
    if prefix in {"L3_EXIT", "L3_STATIC_EXIT"}:
        return {
            "learning_rate": _env_float_clipped("L3_EXIT_LEARNING_RATE", 0.03, lo=1e-4, hi=0.5),
            "num_leaves": _env_int_clipped("L3_EXIT_NUM_LEAVES", 15, lo=2, hi=1024),
            "max_depth": _env_int_clipped("L3_EXIT_MAX_DEPTH", 5, lo=-1, hi=64),
            "feature_fraction": _env_float_clipped("L3_EXIT_COLSAMPLE_BYTREE", 0.7, lo=0.1, hi=1.0),
            "bagging_fraction": _env_float_clipped("L3_EXIT_SUBSAMPLE", 0.7, lo=0.1, hi=1.0),
            "bagging_freq": _env_int_clipped("L3_EXIT_BAGGING_FREQ", 5, lo=0, hi=64),
            "min_child_samples": _env_int_clipped("L3_EXIT_MIN_CHILD_SAMPLES", 50, lo=1, hi=10_000),
            "lambda_l1": _env_float_clipped("L3_EXIT_LAMBDA_L1", 0.2, lo=0.0, hi=100.0),
            "lambda_l2": _env_float_clipped("L3_EXIT_LAMBDA_L2", 2.0, lo=0.0, hi=100.0),
            "seed": int(seed_default),
        }
    return {
        "learning_rate": 0.03,
        "num_leaves": 48,
        "max_depth": 6,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 80,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "seed": int(seed_default),
    }


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
    exit_cfg = _l3_lgb_params("L3_STATIC_EXIT", seed_default=171)
    value_cfg = _l3_lgb_params("L3_STATIC_VALUE", seed_default=172)
    exit_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": exit_cfg["learning_rate"],
        "num_leaves": exit_cfg["num_leaves"],
        "max_depth": exit_cfg["max_depth"],
        "feature_fraction": exit_cfg["feature_fraction"],
        "bagging_fraction": exit_cfg["bagging_fraction"],
        "bagging_freq": exit_cfg["bagging_freq"],
        "min_child_samples": exit_cfg["min_child_samples"],
        "lambda_l1": exit_cfg["lambda_l1"],
        "lambda_l2": exit_cfg["lambda_l2"],
        "verbosity": -1,
        "seed": exit_cfg["seed"],
        "n_jobs": _lgbm_n_jobs(),
    }
    value_params = _l3_value_lgb_params({**exit_params, **value_cfg}, seed=value_cfg["seed"], prep=value_prep)
    ih = feature_cols.index("l3_hold_bars")
    pa_state = _l3_pa_dict_from_matrix(X_static, feature_cols)
    w = _l3_trade_normalized_exit_weights(
        rows_entry[train_mask],
        X_static[train_mask, ih],
        y_exit[train_mask],
        pa_state={k: v[train_mask] for k, v in pa_state.items()},
    )
    static_es_rounds = min(es_rounds, _env_int_clipped("L3_STATIC_EARLY_STOPPING_ROUNDS_CAP", 30, lo=1, hi=1000))
    static_rounds = min(rounds, _env_int_clipped("L3_STATIC_BOOST_ROUNDS_CAP", 120, lo=10, hi=5000))
    cbs, cl = _lgb_train_callbacks_with_round_tqdm(static_es_rounds, static_rounds, "[L3] static-exit")
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
            num_boost_round=static_rounds,
            valid_sets=[lgb.Dataset(X_static[val_mask], label=y_exit[val_mask], feature_name=feature_cols, free_raw_data=False)],
            callbacks=cbs,
        )
    finally:
        for fn in cl:
            fn()
    cbs, cl = _lgb_train_callbacks_with_round_tqdm(static_es_rounds, static_rounds, "[L3] static-value")
    try:
        value_model = lgb.train(
            value_params,
            lgb.Dataset(X_static[train_mask], label=y_value_fit[train_mask], feature_name=feature_cols, free_raw_data=False),
            num_boost_round=static_rounds,
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


# Cox covariates only (excludes vol_surprise / vol_surprise_accel / drawdown_from_peak_atr — kept for LGBM).
L3_COX_FEATURE_NAMES: tuple[str, ...] = (
    "l3_unreal_pnl_atr",
    "l3_regime_divergence",
    "l3_price_velocity_3bar_atr",
    "l3_feature_momentum_regdiv_3bar",
    "l3_regime_stability_3bar",
    "l3_side",
    "l2_decision_confidence",
    "l3_signal_conf_decay",
    "l3_signal_direction_agree",
    "l3_regime_changed",
    "l3_l2_gate_current",
    "l3_l2_gate_decay",
    "l3_would_enter_now",
    "l3_regret_ratio",
    "l3_bars_since_peak",
    "l3_at_new_high",
    "l3_regret_velocity",
    "l3_trade_quality_bayes",
)


def _l3_bayes_weights() -> dict[str, float]:
    return {
        "fav": float(os.environ.get("L3_BAYES_LLR_FAV", "0.28")),
        "adv": float(os.environ.get("L3_BAYES_LLR_ADV", "-0.35")),
        "regime": float(os.environ.get("L3_BAYES_LLR_REGIME", "-0.45")),
        "gate": float(os.environ.get("L3_BAYES_LLR_GATE", "-0.18")),
        "gate_thr": float(os.environ.get("L3_BAYES_GATE_DECAY_THR", "-0.12")),
    }


def _l3_episode_aux_feature_block(
    *,
    n_steps: int,
    idx_arr: np.ndarray,
    entry_i: int,
    side: float,
    decision_class: np.ndarray,
    decision_conf: np.ndarray,
    size: np.ndarray,
    neutral: np.ndarray,
    entry_regime_row: np.ndarray,
    current_regime: np.ndarray,
    unreal_seg: np.ndarray,
    min_conf_arr: np.ndarray,
    min_size_arr: np.ndarray,
    drawdown_from_peak: np.ndarray,
) -> np.ndarray:
    """Entry decay, counterfactual regret, and Bayesian trade-quality path (one row per hold step)."""
    w = _l3_bayes_weights()
    dec_e = float(decision_conf[entry_i])
    gate_e = float(1.0 - neutral[entry_i])
    rid_e = int(np.argmax(entry_regime_row.astype(np.float64)))
    sgn_side = 1.0 if side > 0 else -1.0

    signal_conf_decay = np.empty(n_steps, dtype=np.float32)
    direction_agree = np.empty(n_steps, dtype=np.float32)
    regime_changed = np.empty(n_steps, dtype=np.float32)
    gate_curr = np.empty(n_steps, dtype=np.float32)
    gate_decay = np.empty(n_steps, dtype=np.float32)
    would_enter = np.empty(n_steps, dtype=np.float32)
    regret_ratio = np.empty(n_steps, dtype=np.float32)
    bars_since_peak = np.empty(n_steps, dtype=np.float32)
    at_new_high = np.empty(n_steps, dtype=np.float32)
    regret_velocity = np.empty(n_steps, dtype=np.float32)
    quality_bayes = np.empty(n_steps, dtype=np.float32)

    p0 = float(np.clip(dec_e, 0.05, 0.95))
    log_odds = float(np.log(p0 / (1.0 - p0)))

    running_peak = float(unreal_seg[0])
    peak_idx = 0
    peak_cum = np.maximum.accumulate(unreal_seg.astype(np.float64))

    for j in range(n_steps):
        k = int(idx_arr[j])
        signal_conf_decay[j] = np.float32(decision_conf[k] - dec_e)
        curr_side = 1.0 if int(decision_class[k]) == 0 else (-1.0 if int(decision_class[k]) == 2 else 0.0)
        direction_agree[j] = np.float32(1.0 if curr_side == sgn_side else 0.0)
        regime_changed[j] = np.float32(float(int(np.argmax(current_regime[k].astype(np.float64))) != rid_e))
        gc = float(1.0 - neutral[k])
        gate_curr[j] = np.float32(gc)
        gate_decay[j] = np.float32(gc - gate_e)
        would_enter[j] = np.float32(
            1.0
            if l3_entry_side_from_l2(
                int(decision_class[k]),
                float(decision_conf[k]),
                float(size[k]),
                min_confidence=float(min_conf_arr[k]),
                min_size=float(min_size_arr[k]),
            )
            != 0.0
            else 0.0
        )

        u = float(unreal_seg[j])
        if u >= running_peak - 1e-9:
            running_peak = u
            peak_idx = j
        bars_since_peak[j] = np.float32(j - peak_idx)
        at_new_high[j] = np.float32(1.0 if abs(u - running_peak) < 1e-9 else 0.0)

        pk = float(peak_cum[j])
        if pk > 1e-6:
            regret_ratio[j] = np.float32(max(0.0, (pk - u) / pk))
        else:
            regret_ratio[j] = np.float32(0.0)

        dd = float(drawdown_from_peak[j])
        bsp = float(bars_since_peak[j])
        regret_velocity[j] = np.float32(dd / bsp) if bsp > 0.5 else np.float32(0.0)

        du = float(unreal_seg[j] - unreal_seg[j - 1]) if j > 0 else float(unreal_seg[0])
        favorable = sgn_side * du > 0.0
        llr = w["fav"] if favorable else w["adv"]
        if float(regime_changed[j]) > 0.5:
            llr += w["regime"]
        if float(gate_decay[j]) < w["gate_thr"]:
            llr += w["gate"]
        log_odds = float(log_odds + llr)
        quality_bayes[j] = np.float32(1.0 / (1.0 + np.exp(-log_odds)))

    return np.column_stack(
        [
            signal_conf_decay,
            direction_agree,
            regime_changed,
            gate_curr,
            gate_decay,
            would_enter,
            regret_ratio,
            bars_since_peak,
            at_new_high,
            regret_velocity,
            quality_bayes,
        ]
    ).astype(np.float32, copy=False)


def _l3_ctv_frame_first_event(
    entry_ids: np.ndarray,
    holds: np.ndarray,
    y_exit: np.ndarray,
    X_cov: np.ndarray,
    cov_names: list[str],
    row_mask: np.ndarray,
) -> pd.DataFrame:
    rows_out: list[dict[str, float | int]] = []
    e_m = entry_ids[row_mask]
    h_m = holds[row_mask]
    y_m = y_exit[row_mask]
    X_m = np.nan_to_num(X_cov[row_mask].astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    for uid in np.unique(e_m):
        idx = np.where(e_m == uid)[0]
        order = np.argsort(h_m[idx], kind="mergesort")
        ix = idx[order]
        h_u = h_m[ix].astype(np.float64)
        y_u = y_m[ix].astype(np.int32)
        X_u = X_m[ix]
        first: int | None = None
        for t in range(len(y_u)):
            if y_u[t] == 1:
                first = t
                break
        end_u = len(h_u) - 1 if first is None else int(first)
        for t in range(end_u + 1):
            h_t = float(h_u[t])
            evt = int(first is not None and t == int(first))
            d: dict[str, float | int] = {"id": int(uid), "start": h_t - 1.0, "stop": h_t, "event": evt}
            for ci, name in enumerate(cov_names):
                d[name] = float(X_u[t, ci])
            rows_out.append(d)
    return pd.DataFrame(rows_out)


def _l3_lookup_baseline_cumhaz(ctv: Any, stops: np.ndarray) -> np.ndarray:
    bh = ctv.baseline_cumulative_hazard_
    col = bh.columns[0]
    times = bh.index.to_numpy(dtype=np.float64)
    vals = bh[col].to_numpy(dtype=np.float64)
    out = np.zeros(len(stops), dtype=np.float64)
    for i, s in enumerate(stops.astype(np.float64)):
        m = times <= s + 1e-9
        if not np.any(m):
            out[i] = 0.0
        else:
            out[i] = float(vals[np.where(m)[0][-1]])
    return out


def _l3_append_cox_survival_features(
    X: np.ndarray,
    feature_cols: list[str],
    rows_entry: np.ndarray,
    y_exit: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Fit discrete-interval Cox (time-varying) on train rows; append log partial hazard + log baseline H0(stop)."""
    cox_cols = [c for c in L3_COX_FEATURE_NAMES if c in feature_cols]
    if not cox_cols:
        z = np.zeros((len(X), 2), dtype=np.float32)
        return np.hstack([X, z]), feature_cols + [
            "l3_cox_log_partial_hazard",
            "l3_cox_baseline_cumhaz_at_stop",
        ], {"l3_cox_fitted": False, "reason": "no_cox_columns", "dropped_covariates": {}}

    disabled = os.environ.get("L3_COX_DISABLE", "0").strip().lower() in {"1", "true", "yes"}
    if disabled or CoxTimeVaryingFitter is None:
        reason = "disabled" if disabled else "lifelines_missing"
        if CoxTimeVaryingFitter is None:
            print("  [L3] Cox survival: lifelines not installed — install lifelines or set L3_COX_DISABLE=1.", flush=True)
        z = np.zeros((len(X), 2), dtype=np.float32)
        return np.hstack([X, z]), feature_cols + [
            "l3_cox_log_partial_hazard",
            "l3_cox_baseline_cumhaz_at_stop",
        ], {"l3_cox_fitted": False, "reason": reason, "dropped_covariates": {}}

    ih = feature_cols.index("l3_hold_bars")
    holds = X[:, ih].astype(np.float64)
    cov_idx = [feature_cols.index(c) for c in cox_cols]
    X_cov = X[:, cov_idx]
    df_ctv = _l3_ctv_frame_first_event(rows_entry, holds, y_exit, X_cov, cox_cols, train_mask)
    df_ctv, X_cov, drawdown_stab_meta = _l3_cox_stabilize_drawdown_feature(df_ctv, X_cov, cox_cols)
    if bool(drawdown_stab_meta.get("applied")):
        print(
            f"  [L3] Cox stabilization(drawdown): winsor_q={drawdown_stab_meta.get('winsor_q', float('nan'))}  "
            f"bins={drawdown_stab_meta.get('bin_count', 0)}",
            flush=True,
        )
    min_rows = int(os.environ.get("L3_COX_MIN_ROWS", "400"))
    min_ev = int(os.environ.get("L3_COX_MIN_EVENTS", "80"))
    if len(df_ctv) < min_rows or int(df_ctv["event"].sum()) < min_ev:
        print(
            f"  [L3] Cox survival: skipped (ctv_rows={len(df_ctv)} events={int(df_ctv['event'].sum())} "
            f"need>={min_rows}/{min_ev})",
            flush=True,
        )
        z = np.zeros((len(X), 2), dtype=np.float32)
        return np.hstack([X, z]), feature_cols + [
            "l3_cox_log_partial_hazard",
            "l3_cox_baseline_cumhaz_at_stop",
        ], {
            "l3_cox_fitted": False,
            "reason": "too_few_rows",
            "dropped_covariates": {},
            "drawdown_stabilization": drawdown_stab_meta,
        }

    filtered_cols, dropped_covariates = _l3_filter_cox_covariates(df_ctv, cox_cols)
    if dropped_covariates:
        parts = [f"{name}={reason}" for name, reason in dropped_covariates.items()]
        print("  [L3] Cox survival: filtered covariates -> " + ", ".join(parts), flush=True)
        reason_counts: dict[str, int] = {}
        for reason in dropped_covariates.values():
            key = str(reason).split("(")[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1
        print(f"  [L3] Cox dropped summary: {reason_counts}", flush=True)
    if not filtered_cols:
        z = np.zeros((len(X), 2), dtype=np.float32)
        return np.hstack([X, z]), feature_cols + [
            "l3_cox_log_partial_hazard",
            "l3_cox_baseline_cumhaz_at_stop",
        ], {
            "l3_cox_fitted": False,
            "reason": "filtered_all_covariates",
            "dropped_covariates": dropped_covariates,
            "drawdown_stabilization": drawdown_stab_meta,
        }
    if len(filtered_cols) != len(cox_cols):
        cox_cols = filtered_cols
        cov_idx = [feature_cols.index(c) for c in cox_cols]
        X_cov = X[:, cov_idx]

    penal = float(os.environ.get("L3_COX_PENALIZER", "1.0"))
    ctv = CoxTimeVaryingFitter(penalizer=penal)
    try:
        fit_cols = ["id", "event", "start", "stop", *cox_cols]
        ctv.fit(df_ctv[fit_cols], id_col="id", event_col="event", start_col="start", stop_col="stop", show_progress=False)
    except Exception as ex:  # pragma: no cover
        print(f"  [L3] Cox survival: fit failed ({ex!r}) — filling zeros.", flush=True)
        z = np.zeros((len(X), 2), dtype=np.float32)
        return np.hstack([X, z]), feature_cols + [
            "l3_cox_log_partial_hazard",
            "l3_cox_baseline_cumhaz_at_stop",
        ], {
            "l3_cox_fitted": False,
            "reason": f"fit_error:{ex}",
            "dropped_covariates": dropped_covariates,
            "drawdown_stabilization": drawdown_stab_meta,
        }

    pred_df = pd.DataFrame(np.nan_to_num(X_cov, nan=0.0, posinf=0.0, neginf=0.0), columns=cox_cols)
    ph = np.asarray(ctv.predict_partial_hazard(pred_df), dtype=np.float64).ravel()
    log_ph = np.log(np.clip(ph, 1e-12, None))
    h0 = _l3_lookup_baseline_cumhaz(ctv, holds)
    extra = np.column_stack([log_ph, np.log1p(np.clip(h0, 0.0, None))]).astype(np.float32, copy=False)
    bundle = {
        "l3_cox_fitted": True,
        "fitter": ctv,
        "cov_names": list(cox_cols),
        "reason": "ok",
        "dropped_covariates": dropped_covariates,
        "drawdown_stabilization": drawdown_stab_meta,
    }
    print(
        f"  [L3] Cox time-varying survival: fitted on {len(df_ctv):,} CTV rows ({int(df_ctv['event'].sum())} events); "
        f"covariates={len(cox_cols)} penalizer={penal}",
        flush=True,
    )
    return np.hstack([X, extra]), feature_cols + [
        "l3_cox_log_partial_hazard",
        "l3_cox_baseline_cumhaz_at_stop",
    ], bundle


def l3_infer_cox_features(
    cox_bundle: Mapping[str, Any] | None,
    X_base: np.ndarray,
    feature_cols: list[str],
) -> np.ndarray:
    """Two Cox-derived features aligned with training; zeros if bundle missing."""
    if not cox_bundle or not cox_bundle.get("l3_cox_fitted") or cox_bundle.get("fitter") is None:
        return np.zeros(2, dtype=np.float32)
    ctv = cox_bundle["fitter"]
    cov_names: list[str] = list(cox_bundle["cov_names"])
    try:
        vec = np.asarray([float(X_base[feature_cols.index(n)]) for n in cov_names], dtype=np.float64).reshape(1, -1)
    except ValueError:
        return np.zeros(2, dtype=np.float32)
    pred_df = pd.DataFrame(np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0), columns=cov_names)
    ph = float(np.asarray(ctv.predict_partial_hazard(pred_df), dtype=np.float64).ravel()[0])
    ih = feature_cols.index("l3_hold_bars")
    stop = float(X_base[ih])
    h0 = float(_l3_lookup_baseline_cumhaz(ctv, np.asarray([stop], dtype=np.float64))[0])
    return np.asarray(
        [np.float32(np.log(max(ph, 1e-12))), np.float32(np.log1p(max(h0, 0.0)))],
        dtype=np.float32,
    )


def l3_load_cox_bundle(meta: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not meta or not meta.get("l3_cox_fitted"):
        return None
    fname = str(meta.get("l3_cox_artifact_file", L3_COX_FILE))
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def l3_policy_state_key(
    regime_probs: np.ndarray,
    vol_value: float,
    meta: dict[str, Any],
    *,
    pa_state: Mapping[str, Any] | pd.Series | None = None,
) -> str:
    vol_quantiles = [float(x) for x in (meta.get("policy_state_vol_quantiles") or [])]
    keys = _state_keys_from_regime_vol(np.asarray(regime_probs, dtype=np.float32).reshape(1, -1), np.asarray([vol_value], dtype=np.float32), vol_quantiles=vol_quantiles)
    base = str(keys[0]) if len(keys) else "r0_v0"
    if not _l3_pa_policy_enabled():
        return base
    return str(_append_pa_bucket_to_state_keys(np.asarray([base], dtype=object), np.asarray([pa_state_bucket_label_from_mapping(pa_state)], dtype=object))[0])


def l3_entry_policy_params(
    regime_probs: np.ndarray,
    vol_value: float,
    meta: dict[str, Any],
    *,
    pa_state: Mapping[str, Any] | pd.Series | None = None,
) -> tuple[float, float, int, str]:
    state_key = l3_policy_state_key(regime_probs, vol_value, meta, pa_state=pa_state)
    params = (meta.get("l3_entry_policy_by_state") or {}).get(state_key, {})
    min_conf = float(params.get("min_confidence", meta.get("l3_entry_min_confidence", _l3_entry_policy_defaults()[0])))
    min_size = float(params.get("min_size", meta.get("l3_entry_min_size", _l3_entry_policy_defaults()[1])))
    hold_map = meta.get("l3_target_horizon_bars_by_state") or {}
    max_hold = int(hold_map.get(state_key, meta.get("l3_target_horizon_bars", _l3_target_horizon_bars(30))))
    return min_conf, min_size, max_hold, state_key


def l3_exit_policy_params(
    regime_probs: np.ndarray,
    vol_value: float,
    hold_bars: int,
    meta: dict[str, Any],
    *,
    pa_state: Mapping[str, Any] | pd.Series | None = None,
) -> tuple[float, float, int, str, str, float]:
    base_key = str(
        _exit_state_keys_from_regime_vol_hold(
            np.asarray(regime_probs, dtype=np.float32).reshape(1, -1),
            np.asarray([vol_value], dtype=np.float32),
            np.asarray([hold_bars], dtype=np.float32),
            vol_quantiles=[float(x) for x in (meta.get("policy_state_vol_quantiles") or [])],
        )[0]
    )
    state_key = (
        base_key
        if not _l3_pa_policy_enabled()
        else str(
            _append_pa_bucket_to_state_keys(
                np.asarray([base_key], dtype=object),
                np.asarray([pa_state_bucket_label_from_mapping(pa_state)], dtype=object),
            )[0]
        )
    )
    params = (meta.get("l3_exit_policy_by_state") or {}).get(state_key, {})
    prob_thr = float(params.get("exit_prob_threshold", meta.get("l3_exit_prob_threshold", 0.55)))
    split_bar = int(params.get("early_hold_split_bar", meta.get("l3_policy_early_hold_split_bar", 3)))
    prob_thr_early = float(params.get("exit_prob_threshold_early", meta.get("l3_exit_prob_threshold_early", prob_thr)))
    prob_thr_late = float(params.get("exit_prob_threshold_late", meta.get("l3_exit_prob_threshold_late", prob_thr)))
    prob_thr = prob_thr_early if int(hold_bars) < split_bar else prob_thr_late
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
    merged = ensure_pa_state_features(merged)
    pa_state = _l3_pa_dict_from_frame(merged)
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
    state_keys_all = _append_pa_bucket_to_state_keys(
        _state_keys_from_regime_vol(current_regime, current_vol, vol_quantiles=policy_vol_quantiles),
        pa_state_bucket_labels_from_frame(merged) if _l3_pa_policy_enabled() else None,
    )
    if "decision_peak_bar" in merged.columns:
        peak_bar = pd.to_numeric(merged["decision_peak_bar"], errors="coerce").fillna(_l3_target_horizon_bars(max_hold)).to_numpy(dtype=np.float32)
    else:
        peak_bar = np.full(len(merged), _l3_target_horizon_bars(max_hold), dtype=np.float32)
    target_horizon_global, target_horizon_by_state = _l3_target_horizon_by_state(
        state_keys_all[oot_mask] if oot_mask.any() else state_keys_all,
        peak_bar[oot_mask] if oot_mask.any() else peak_bar,
        max_hold=max_hold,
        pa_state=None if not _l3_pa_targets_enabled() else {k: v[oot_mask] if oot_mask.any() else v for k, v in pa_state.items()},
    )
    entry_policy_global, entry_policy_by_state = _l3_search_entry_policy(
        state_keys_all[oot_mask] if oot_mask.any() else state_keys_all,
        decision_class[oot_mask] if oot_mask.any() else decision_class,
        decision_conf[oot_mask] if oot_mask.any() else decision_conf,
        size[oot_mask] if oot_mask.any() else size,
        edge_atr[oot_mask] if oot_mask.any() else edge_atr,
        tau_edge,
        pa_state=None if not _l3_pa_targets_enabled() else {k: v[oot_mask] if oot_mask.any() else v for k, v in pa_state.items()},
    )
    entry_policy_arrays = _l3_lookup_policy_map(
        state_keys_all,
        entry_policy_by_state,
        defaults={
            "min_confidence": float(entry_policy_global["min_confidence"]),
            "min_size": float(entry_policy_global["min_size"]),
        },
    )
    if "l2_decision_neutral" in merged.columns:
        neutral = pd.to_numeric(merged["l2_decision_neutral"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    else:
        neutral = np.where(decision_class == 1, 1.0, 0.25).astype(np.float64)
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
        "l3_drawdown_from_peak_atr",
        "l3_price_velocity_3bar_atr",
        "l3_feature_momentum_regdiv_3bar",
        "l3_vol_surprise_accel",
        "l3_regime_stability_3bar",
        *PA_STATE_FEATURES,
        "l3_signal_conf_decay",
        "l3_signal_direction_agree",
        "l3_regime_changed",
        "l3_l2_gate_current",
        "l3_l2_gate_decay",
        "l3_would_enter_now",
        "l3_regret_ratio",
        "l3_bars_since_peak",
        "l3_at_new_high",
        "l3_regret_velocity",
        "l3_trade_quality_bayes",
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
        min_horizon = _env_int_clipped("L3_POLICY_MIN_HORIZON_BARS", 5, lo=1, hi=max_hold)
        target_horizon = int(max(min_horizon, min(int(max_hold), int(horizon_arr[i]))))
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
        # Causal: peak at step j is max(unreal[0:j]); no look-ahead within the episode.
        peak_unreal_cum = np.maximum.accumulate(unreal_seg.astype(np.float64))
        drawdown_from_peak = (peak_unreal_cum - unreal_seg.astype(np.float64)).astype(np.float32, copy=False)
        close_seg = close_px[idx_arr].astype(np.float64, copy=False)
        atr_d = max(float(atr), 1e-6)
        vel3 = np.zeros(n_steps, dtype=np.float32)
        for j in range(n_steps):
            j0 = max(0, j - 3)
            vel3[j] = float((close_seg[j] - close_seg[j0]) / atr_d)
        reg_div_d = regime_div_seg.astype(np.float64, copy=False)
        mom_rd = np.zeros(n_steps, dtype=np.float32)
        for j in range(n_steps):
            j0 = max(0, j - 3)
            mom_rd[j] = float(reg_div_d[j] - reg_div_d[j0])
        vs = vol_surprise_seg.astype(np.float64, copy=False)
        vs_acc = np.zeros(n_steps, dtype=np.float32)
        for j in range(n_steps):
            if j >= 2:
                vs_acc[j] = float(vs[j] - 2.0 * vs[j - 1] + vs[j - 2])
        cr = current_regime[idx_arr].astype(np.float64, copy=False)
        rid = np.argmax(cr, axis=1).astype(np.int32, copy=False)
        stab = np.zeros(n_steps, dtype=np.float32)
        for j in range(n_steps):
            lo = max(0, j - 2)
            stab[j] = float(np.mean(rid[lo : j + 1] == rid[j]))
        aux_block = _l3_episode_aux_feature_block(
            n_steps=n_steps,
            idx_arr=idx_arr,
            entry_i=i,
            side=float(side),
            decision_class=decision_class,
            decision_conf=decision_conf,
            size=size,
            neutral=neutral,
            entry_regime_row=entry_regime[i],
            current_regime=current_regime,
            unreal_seg=unreal_seg,
            min_conf_arr=entry_policy_arrays["min_confidence"],
            min_size_arr=entry_policy_arrays["min_size"],
            drawdown_from_peak=drawdown_from_peak,
        )
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
                drawdown_from_peak,
                vel3,
                mom_rd,
                vs_acc,
                stab,
                merged.loc[idx_arr, PA_STATE_FEATURES].to_numpy(dtype=np.float32, copy=False),
                aux_block,
            ]
        ).astype(np.float32, copy=False)
        terminal_unreal = float(unreal_seg[-1])
        future_gain_left = (terminal_unreal - unreal_seg).astype(np.float32, copy=False)
        continuation_score = _l3_continuation_score(future_gain_left, live_edge_seg)
        step_trend = np.asarray(pa_state["pa_state_trend_strength"][idx_arr], dtype=np.float32)
        step_follow = np.asarray(pa_state["pa_state_followthrough_quality"][idx_arr], dtype=np.float32)
        step_range = np.asarray(pa_state["pa_state_range_risk"][idx_arr], dtype=np.float32)
        step_breakout = np.asarray(pa_state["pa_state_breakout_failure_risk"][idx_arr], dtype=np.float32)
        step_pullback = np.asarray(pa_state["pa_state_pullback_exhaustion"][idx_arr], dtype=np.float32)
        last_step = np.arange(n_steps) == (n_steps - 1)
        late_hold_start = max(
            _env_int_clipped("L3_LATE_HOLD_MIN_BARS", 2, lo=1, hi=max_hold),
            int(np.ceil(target_horizon * _env_float_clipped("L3_LATE_HOLD_FRAC", 0.75, lo=0.1, hi=1.0))),
        )
        if _l3_pa_targets_enabled():
            eps_arr = exit_epsilon_atr * np.clip(
                1.0 + 0.25 * step_range + 0.20 * step_breakout + 0.10 * step_pullback - 0.12 * step_trend - 0.08 * step_follow,
                0.70,
                1.60,
            )
            loss_buffer_arr = _l3_exit_loss_buffer_atr() * np.clip(
                1.0 - 0.18 * step_range - 0.14 * step_breakout - 0.08 * step_pullback + 0.10 * step_trend,
                0.55,
                1.20,
            )
            live_edge_floor_arr = _l3_exit_live_edge_floor() * np.clip(
                1.0 + 0.20 * step_range + 0.18 * step_breakout - 0.12 * step_trend,
                0.60,
                1.50,
            )
            late_start_scale = np.clip(
                1.0
                + 0.20 * float(pa_state["pa_state_trend_strength"][i])
                + 0.10 * float(pa_state["pa_state_followthrough_quality"][i])
                - 0.24 * float(pa_state["pa_state_range_risk"][i])
                - 0.20 * float(pa_state["pa_state_breakout_failure_risk"][i])
                - 0.10 * float(pa_state["pa_state_pullback_exhaustion"][i]),
                0.55,
                1.25,
            )
            late_hold_start = max(1, int(np.ceil(late_hold_start * late_start_scale)))
        else:
            eps_arr = np.full(n_steps, exit_epsilon_atr, dtype=np.float32)
            loss_buffer_arr = np.full(n_steps, _l3_exit_loss_buffer_atr(), dtype=np.float32)
            live_edge_floor_arr = np.full(n_steps, _l3_exit_live_edge_floor(), dtype=np.float32)
        weak_continuation = continuation_score <= eps_arr
        clearly_spent = future_gain_left <= -loss_buffer_arr
        live_edge_faded = live_edge_seg <= live_edge_floor_arr
        late_flat = (holds >= late_hold_start) & (future_gain_left <= 0.0)
        exit_block = (last_step | clearly_spent | (weak_continuation & live_edge_faded) | late_flat).astype(np.int32, copy=False)
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
                "pa_target_semantics": "PA-aware continuation thresholds, horizon scaling, and entry-quality scoring are applied inside dataset/target construction",
            },
        )
    print(
        f"  [L3] policy dataset: entry signals model={n_policy_signals_model:,} "
        f"truth_edge_fallback={n_policy_signals_truth:,}  policy_rows={sum(int(x.shape[0]) for x in rows_x_blocks):,}  "
        f"allow_truth_fallback={allow_truth_fallback}  entry_policy_states={len(entry_policy_by_state)}  "
        f"target_horizon_global={target_horizon_global}",
        flush=True,
    )
    total_entries = int(n_policy_signals_model + n_policy_signals_truth)
    total_rows = int(sum(int(x.shape[0]) for x in rows_x_blocks))
    avg_rows_all = float(total_rows / total_entries) if total_entries > 0 else float("nan")
    avg_rows_model = float(total_rows / n_policy_signals_model) if n_policy_signals_model > 0 else float("nan")
    print(
        f"  [L3] policy dataset detail: total_entries={total_entries:,}  "
        f"avg_rows_per_entry={avg_rows_all:.2f}  avg_rows_per_model_entry={avg_rows_model:.2f}",
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
    value_model: lgb.Booster | None,
    *,
    value_nonzero_model: lgb.Booster | None = None,
    value_hurdle_prob_power: float = 1.0,
    exit_calibrator: Any = None,
    value_policy_mode: str = "prob_only",
    value_tie_margin: float = 0.03,
    exit_policy_summary: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    out = {
        "val_hold_recall": float("nan"),
        "val_exit_rate": float("nan"),
        "val_auc": float("nan"),
        "holdout_hold_recall": float("nan"),
        "holdout_auc": float("nan"),
    }
    vm = np.asarray(val_mask, dtype=bool)
    if int(vm.sum()) < 5:
        return out
    p_exit = _apply_l3_exit_calibrator(exit_model.predict(X[vm]).astype(np.float64), exit_calibrator)
    yv = y_exit[vm].astype(np.int32)
    ih_hold = feature_cols.index("l3_hold_bars")
    hold_vm = np.asarray(X[vm, ih_hold], dtype=np.int64)
    min_hold_bars = int(max(0, round(_l3_exit_infer_params(None)["min_hold"])))
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
    if min_hold_bars > 0:
        yhat = np.where(hold_vm < min_hold_bars, 0, yhat).astype(np.int32)
    acc = float(accuracy_score(yv, yhat))
    f1 = float(f1_score(yv, yhat, zero_division=0))
    cm = confusion_matrix(yv, yhat, labels=[0, 1])
    hold_recall = float(np.mean(yhat[yv == 0] == 0)) if np.any(yv == 0) else float("nan")
    hold_precision = float(np.mean(yv[yhat == 0] == 0)) if np.any(yhat == 0) else float("nan")
    exit_recall = float(np.mean(yhat[yv == 1] == 1)) if np.any(yv == 1) else float("nan")
    exit_precision = float(np.mean(yv[yhat == 1] == 1)) if np.any(yhat == 1) else float("nan")
    exit_rate = float(np.mean(yhat))
    out["val_hold_recall"] = hold_recall
    out["val_exit_rate"] = exit_rate
    out["val_auc"] = auc
    print("\n  [L3] val — exit (extended)", flush=True)
    print(
        f"    AUC={auc:.4f}  log_loss={ll:.4f}  Brier={br:.4f}  ECE={ece:.4f}  acc@0.5={acc:.4f}  F1={f1:.4f}  "
        f"exit_rate={exit_rate:.3f}  hold_recall={hold_recall:.4f}  hold_precision={hold_precision:.4f}  "
        f"exit_recall={exit_recall:.4f}  exit_precision={exit_precision:.4f}",
        flush=True,
    )
    print(f"    confusion [[TN FP][FN TP]]:\n    {cm}", flush=True)
    if min_hold_bars > 0:
        n_early = int(np.sum(hold_vm < min_hold_bars))
        n_would_exit = int(np.sum((hold_vm < min_hold_bars) & (p_exit >= 0.5)))
        print(
            f"    min_hold_bars={min_hold_bars}: val rows in early window={n_early:,}  "
            f"(of those, raw acc@0.5 would predict exit={n_would_exit:,} -> forced hold)",
            flush=True,
        )
    if exit_policy_summary:
        print(
            "    policy-search decomposition: "
            f"utility_mean={float(exit_policy_summary.get('utility_mean', float('nan'))):.4f}  "
            f"utility_p10={float(exit_policy_summary.get('utility_p10', float('nan'))):.4f}  "
            f"hold_recall_contrib={float(exit_policy_summary.get('hold_recall_contrib', float('nan'))):.4f}  "
            f"exit_rate_penalty_contrib={float(exit_policy_summary.get('exit_rate_penalty_contrib', float('nan'))):.4f}",
            flush=True,
        )
    if np.isfinite(hold_recall) and hold_recall < 0.20:
        print(f"    WARNING: hold recall still very low ({hold_recall:.3f})", flush=True)

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
                print(f"    entry-regime {k}  n={n_k:,}  [WARN tiny slice: skip AUC]", flush=True)
                continue
            yy = yv[m]
            pp = p_exit[m]
            if len(np.unique(yy)) < 2:
                continue
            try:
                auc_k = float(roc_auc_score(yy, pp))
                warn = "  [WARN n<50]" if n_k < 50 else ""
                print(f"    entry-regime {k}  AUC={auc_k:.4f}  n={n_k:,}{warn}", flush=True)
            except ValueError:
                pass

    if "l3_regime_divergence" in feature_cols:
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

    c_ec = float("nan")
    if "l2_decision_confidence" in feature_cols:
        i_conf = feature_cols.index("l2_decision_confidence")
        c_ec = pearson_corr(X[vm, i_conf].astype(np.float64), p_exit)
    if value_model is None:
        print(f"    corr(L2 conf, L3 exit p)={c_ec:.4f}  (L3 value model disabled)", flush=True)
    else:
        vv_pred = _l3_value_predict_hurdle(
            X[vm],
            value_model,
            value_nonzero_model,
            prob_power=value_hurdle_prob_power,
        )
        c_sz_val = float("nan")
        if "l2_size" in feature_cols:
            i_size = feature_cols.index("l2_size")
            c_sz_val = pearson_corr(X[vm, i_size].astype(np.float64), vv_pred)
        print(
            f"    corr(L2 conf, L3 exit p)={c_ec:.4f}  corr(L2 size, L3 value pred)={c_sz_val:.4f}",
            flush=True,
        )

    t_vm = np.asarray(pd.to_datetime(t_state))[vm]
    order = np.argsort(t_vm)
    fr = flip_rate_sorted(p_exit, order)
    infer_cfg = _l3_exit_infer_params(None)
    alpha = float(np.clip(infer_cfg["ema_alpha"], 0.01, 0.99))
    p_ord = p_exit[order]
    if p_ord.size:
        ema = np.empty_like(p_ord)
        ema[0] = p_ord[0]
        for i in range(1, len(p_ord)):
            ema[i] = alpha * p_ord[i] + (1.0 - alpha) * ema[i - 1]
        fr_s = flip_rate_sorted(ema, np.arange(len(ema), dtype=np.int64))
    else:
        fr_s = float("nan")
    print(
        f"    exit prob flip_rate (time-sorted val): raw={fr:.6f}  ema(alpha={alpha:.2f})={fr_s:.6f}  "
        f"hysteresis=({infer_cfg['enter_thr']:.2f},{infer_cfg['leave_thr']:.2f})  min_hold={int(round(infer_cfg['min_hold']))}",
        flush=True,
    )

    if value_model is None:
        print("\n  [L3] val — value (extended): skipped (L3_VALUE_MODE=disabled)", flush=True)
    else:
        vv_pred = _l3_value_predict_hurdle(
            X[vm],
            value_model,
            value_nonzero_model,
            prob_power=value_hurdle_prob_power,
        )
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
            f"dir_acc={dir_acc:.4f}  pred_std={std_v:.6f}  degen={degen_v}  "
            f"policy_mode={value_policy_mode}  tie_margin={float(value_tie_margin):.4f}",
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
        hold_tm = np.asarray(X[tm, ih_hold], dtype=np.int64)
        try:
            auc_t = float(roc_auc_score(yt, p_t))
        except ValueError:
            auc_t = float("nan")
        br_t = brier_binary(yt.astype(np.float64), p_t)
        ece_t = ece_binary(yt, p_t)
        yhat_t = (p_t >= 0.5).astype(np.int32)
        if min_hold_bars > 0:
            yhat_t = np.where(hold_tm < min_hold_bars, 0, yhat_t).astype(np.int32)
        hold_recall_t = float(np.mean(yhat_t[yt == 0] == 0)) if np.any(yt == 0) else float("nan")
        hold_precision_t = float(np.mean(yt[yhat_t == 0] == 0)) if np.any(yhat_t == 0) else float("nan")
        print(
            f"\n  [L3] holdout — exit AUC={auc_t:.4f}  Brier={br_t:.4f}  ECE={ece_t:.4f}  "
            f"acc@0.5={float(accuracy_score(yt, yhat_t)):.4f}  F1={float(f1_score(yt, yhat_t, zero_division=0)):.4f}  "
            f"hold_recall={hold_recall_t:.4f}  hold_precision={hold_precision_t:.4f}  n={int(tm.sum()):,}",
            flush=True,
        )
        out["holdout_hold_recall"] = hold_recall_t
        out["holdout_auc"] = auc_t
    return out


def train_l3_exit_manager(df: pd.DataFrame, l1a_outputs: pd.DataFrame, l2_outputs: pd.DataFrame) -> L3TrainingBundle:
    train_started_at = datetime.now().astimezone()
    train_started_perf = time.perf_counter()
    print(f"  [L3] training started at {train_started_at.strftime('%Y-%m-%d %H:%M:%S %z')}", flush=True)
    traj_cfg = L3TrajectoryConfig()
    value_disabled = (os.environ.get("L3_VALUE_MODE", "") or "").strip().lower() == "disabled"
    want_traj = os.environ.get("L3_TRAJ_GRU", "0").strip().lower() in {"1", "true", "yes"}
    if value_disabled:
        want_traj = False
        print("  [L3] L3_VALUE_MODE=disabled — skipping value head & GRU hybrid.", flush=True)
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
    oot_start = np.datetime64(os.environ.get("L3_OOT_START", CAL_END))
    holdout_start = np.datetime64(os.environ.get("L3_HOLDOUT_START", TEST_END))
    if holdout_start <= oot_start:
        raise RuntimeError(
            f"L3: invalid split config holdout_start={holdout_start} must be after oot_start={oot_start}."
        )
    oot_mask = (t_state >= oot_start) & (t_state < holdout_start)
    holdout_mask = t_state >= holdout_start
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

    X, feature_cols, cox_bundle = _l3_append_cox_survival_features(X, feature_cols, rows_entry, y_exit, train_mask)

    log_layer_banner("[L3] Exit / policy (LGBM)")
    print(
        f"  [L3] exit policy states: granularity={os.environ.get('L3_EXIT_STATE_GRANULARITY', 'coarse')!r} "
        f"(set L3_EXIT_STATE_GRANULARITY=full for legacy keys)",
        flush=True,
    )
    log_time_key_arrays(
        "L3",
        pd.Series(t_state[train_mask]),
        pd.Series(t_state[val_mask]),
        train_label="policy train (OOT trades, ~70% of distinct entries)",
        val_label="policy val (OOT trades, ~30% of distinct entries)",
        extra_note=f"Split by rows_entry (signal bar), not by policy row index. Times in [{oot_start}, {holdout_start}); holdout t>={holdout_start}.",
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
    if "l3_regime_divergence" in feature_cols:
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
        "  [L3] target semantics: hold means meaningful continuation remains to a fixed deadline; "
        "exit fires on spent continuation, late-flat paths, or terminal bar.",
        flush=True,
    )
    print(
        f"  [L3] will write: {artifact_path(L3_EXIT_FILE)}"
        f"{' | ' + artifact_path(L3_VALUE_FILE) if not value_disabled else ''} | {artifact_path(L3_META_FILE)}",
        flush=True,
    )
    log_label_baseline("l3_exit", y_exit[train_mask], task="cls")
    log_label_baseline("l3_value", y_value[train_mask], task="reg")
    pa_state_all = _l3_pa_dict_from_matrix(X, feature_cols)
    value_hurdle_enabled = False
    value_hurdle_eps = 0.0
    value_hurdle_prob_power = 1.0
    value_nonzero_target = np.zeros(len(y_value), dtype=np.int32)
    value_nonzero_train = np.zeros(len(y_value), dtype=bool)
    value_nonzero_val = np.zeros(len(y_value), dtype=bool)
    if value_disabled:
        y_value_fit = y_value.astype(np.float64)
        value_prep = {
            "objective": "disabled",
            "metric": "none",
            "clip_enabled": False,
            "clip_lo_q": 0.0,
            "clip_hi_q": 1.0,
            "clip_lo": 0.0,
            "clip_hi": 0.0,
            "train_clipped_frac": 0.0,
        }
        print("  [L3] value target prep: disabled (exit-only policy)", flush=True)
    else:
        y_value_fit, value_prep = _l3_prepare_value_targets(y_value, train_mask, pa_state=pa_state_all)
        value_hurdle_enabled = os.environ.get("L3_VALUE_HURDLE", "1").strip().lower() in {"1", "true", "yes"}
        value_hurdle_eps = _l3_value_hurdle_epsilon(y_value, train_mask)
        value_hurdle_prob_power = float(_env_float_clipped("L3_VALUE_HURDLE_PROB_POWER", 1.0, lo=0.5, hi=2.0))
        value_nonzero_target = (np.abs(y_value) >= value_hurdle_eps).astype(np.int32)
        value_nonzero_train = train_mask & (value_nonzero_target == 1)
        value_nonzero_val = val_mask & (value_nonzero_target == 1)
        value_hurdle_min_train = _env_int_clipped("L3_VALUE_HURDLE_MIN_TRAIN_ROWS", 120, lo=20, hi=200_000)
        value_hurdle_min_val = _env_int_clipped("L3_VALUE_HURDLE_MIN_VAL_ROWS", 40, lo=10, hi=50_000)
        if int(value_nonzero_train.sum()) < value_hurdle_min_train or int(value_nonzero_val.sum()) < value_hurdle_min_val:
            value_hurdle_enabled = False
        value_prep["hurdle_enabled"] = bool(value_hurdle_enabled)
        value_prep["hurdle_nonzero_epsilon"] = float(value_hurdle_eps)
        value_prep["hurdle_prob_power"] = float(value_hurdle_prob_power)
        value_prep["hurdle_nonzero_train_rows"] = int(value_nonzero_train.sum())
        value_prep["hurdle_nonzero_val_rows"] = int(value_nonzero_val.sum())
        print(
            f"  [L3] value target prep: objective={value_prep['objective']} metric={value_prep['metric']}  "
            f"clip={bool(value_prep['clip_enabled'])} q=[{float(value_prep['clip_lo_q']):.2f}, {float(value_prep['clip_hi_q']):.2f}]  "
            f"train_clip=[{float(value_prep['clip_lo']):.4f}, {float(value_prep['clip_hi']):.4f}]  "
            f"train_clipped_frac={float(value_prep['train_clipped_frac']):.3f}",
            flush=True,
        )
        print(
            f"  [L3] value hurdle: enabled={bool(value_hurdle_enabled)}  eps={float(value_hurdle_eps):.4f}  "
            f"nonzero train/val={int(value_nonzero_train.sum()):,}/{int(value_nonzero_val.sum()):,}  "
            f"prob_power={float(value_hurdle_prob_power):.2f}",
            flush=True,
        )
    exit_pos_w, hold_neg_w = _l3_exit_class_weights(y_exit[train_mask])
    print(
        f"  [L3] exit class weights: hold={hold_neg_w:.3f}  exit={exit_pos_w:.3f}",
        flush=True,
    )

    use_hybrid = want_traj
    gru_min_train_rows = _env_int_clipped("L3_GRU_MIN_TRAIN_ROWS", 300, lo=1, hi=1_000_000)
    gru_min_val_rows = _env_int_clipped("L3_GRU_MIN_VAL_ROWS", 30, lo=1, hi=1_000_000)
    use_hybrid = (
        use_hybrid
        and int(train_mask.sum()) >= gru_min_train_rows
        and int(val_mask.sum()) >= gru_min_val_rows
        and traj_seq.shape[0] == len(X)
    )
    static_cols = list(feature_cols)
    n_static = len(static_cols)
    emb_all: np.ndarray | None = None
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
        if not os.environ.get("L3_TRAJ_GRU", "0").strip().lower() in {"1", "true", "yes"}:
            print("  [L3] hybrid disabled (L3_TRAJ_GRU=0); LightGBM on static features only.", flush=True)
        else:
            print(
                f"  [L3] hybrid skipped (need train>={gru_min_train_rows} & val>={gru_min_val_rows}); "
                f"got train={int(train_mask.sum())} val={int(val_mask.sum())}.",
                flush=True,
            )
        encoder = None
        X_lgb = X
        feature_cols = static_cols

    rounds_exit = _l3_exit_boost_rounds()
    rounds_value = _l3_boost_rounds()
    es_rounds = _l3_early_stopping_rounds()
    exit_cfg = _l3_lgb_params("L3_EXIT", seed_default=71)
    value_cfg = _l3_lgb_params("L3_VALUE", seed_default=72)
    value_nz_cfg = _l3_lgb_params("L3_VALUE_NZ", seed_default=73)
    exit_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": exit_cfg["learning_rate"],
        "num_leaves": exit_cfg["num_leaves"],
        "max_depth": exit_cfg["max_depth"],
        "feature_fraction": exit_cfg["feature_fraction"],
        "bagging_fraction": exit_cfg["bagging_fraction"],
        "bagging_freq": exit_cfg["bagging_freq"],
        "min_child_samples": exit_cfg["min_child_samples"],
        "lambda_l1": exit_cfg["lambda_l1"],
        "lambda_l2": exit_cfg["lambda_l2"],
        "verbosity": -1,
        "seed": exit_cfg["seed"],
        "n_jobs": _lgbm_n_jobs(),
    }
    value_params = (
        None
        if value_disabled
        else _l3_value_lgb_params(
            {
                **exit_params,
                **value_cfg,
            },
            seed=value_cfg["seed"],
            prep=value_prep,
        )
    )
    value_nonzero_params = (
        None
        if value_disabled or not value_hurdle_enabled
        else {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": value_nz_cfg["learning_rate"],
            "num_leaves": value_nz_cfg["num_leaves"],
            "max_depth": value_nz_cfg["max_depth"],
            "feature_fraction": value_nz_cfg["feature_fraction"],
            "bagging_fraction": value_nz_cfg["bagging_fraction"],
            "bagging_freq": value_nz_cfg["bagging_freq"],
            "min_child_samples": value_nz_cfg["min_child_samples"],
            "lambda_l1": value_nz_cfg["lambda_l1"],
            "lambda_l2": value_nz_cfg["lambda_l2"],
            "verbosity": -1,
            "seed": value_nz_cfg["seed"],
            "n_jobs": _lgbm_n_jobs(),
        }
    )
    X_lgb, X, feature_cols, static_cols, l3_gain_sel_meta = _l3_maybe_prune_features_by_exit_gain(
        X_lgb=np.asarray(X_lgb, dtype=np.float32, order="C"),
        X=np.asarray(X, dtype=np.float32, order="C"),
        feature_cols=list(feature_cols),
        static_cols=list(static_cols),
        use_hybrid=use_hybrid,
        emb_matrix=emb_all,
        train_mask=train_mask,
        val_mask=val_mask,
        y_exit=y_exit,
        rows_entry=rows_entry,
        pa_state_all=pa_state_all,
        exit_params=exit_params,
        es_rounds=es_rounds,
        cox_bundle=cox_bundle,
    )
    n_static = len(static_cols)
    feat_ratio = float(train_mask.sum()) / max(float(len(feature_cols)), 1.0)
    print(
        f"  [L3] sample/feature ratio (train rows / features): {feat_ratio:.2f}:1  "
        f"(train_rows={int(train_mask.sum()):,}, features={len(feature_cols)}; after gain-select)",
        flush=True,
    )
    static_exit_model = None
    static_value_model = None
    l3_model_total = 1 if value_disabled else (3 if value_hurdle_enabled else 2)
    l3_outer = tqdm(
        total=l3_model_total,
        desc="[L3] models",
        unit="model",
        leave=True,
        file=TQDM_FILE,
        disable=not _lgb_round_tqdm_enabled(),
    )
    try:
        ih_tr = feature_cols.index("l3_hold_bars")
        hold_tr = X_lgb[train_mask, ih_tr].astype(np.float64)
        y_exit_tr = y_exit[train_mask]
        w_exit = _l3_trade_normalized_exit_weights(
            rows_entry[train_mask],
            hold_tr,
            y_exit_tr,
            pa_state={k: v[train_mask] for k, v in pa_state_all.items()},
        )
        exit_pos_w, hold_neg_w = _l3_exit_class_weights(y_exit_tr)
        hb_tr = _hold_bucket_ids(hold_tr)
        hb_stats = []
        for hb in range(4):
            m = hb_tr == hb
            if not np.any(m):
                hb_stats.append(f"b{hb}:n=0")
                continue
            hb_stats.append(
                f"b{hb}:n={int(np.sum(m))} w={float(np.mean(w_exit[m])):.3f}"
            )
        print(
            f"  [L3] exit weight profile: hold_rate={float(np.mean(y_exit_tr == 0)):.4f}  "
            f"exit_rate={float(np.mean(y_exit_tr == 1)):.4f}  "
            f"class_w(exit={exit_pos_w:.3f}, hold={hold_neg_w:.3f})  "
            f"bucket_w({' | '.join(hb_stats)})",
            flush=True,
        )
        cbs, cl = _lgb_train_callbacks_with_round_tqdm(es_rounds, rounds_exit, "[L3] exit")
        try:
            exit_model = lgb.train(
                exit_params,
                lgb.Dataset(
                    X_lgb[train_mask],
                    label=y_exit[train_mask],
                    weight=w_exit.astype(np.float32),
                    feature_name=feature_cols,
                    free_raw_data=False,
                ),
                num_boost_round=rounds_exit,
                valid_sets=[lgb.Dataset(X_lgb[val_mask], label=y_exit[val_mask], feature_name=feature_cols, free_raw_data=False)],
                callbacks=cbs,
            )
        finally:
            for fn in cl:
                fn()
        l3_outer.set_postfix_str("exit", refresh=False)
        l3_outer.update(1)

        if value_disabled:
            value_model = None
            value_nonzero_model = None
        else:
            if value_hurdle_enabled and value_nonzero_params is not None:
                nz_train_w = _l3_hurdle_nonzero_weights(value_nonzero_target[train_mask])
                cbs, cl = _lgb_train_callbacks_with_round_tqdm(es_rounds, rounds_value, "[L3] value-nonzero")
                try:
                    value_nonzero_model = lgb.train(
                        value_nonzero_params,
                        lgb.Dataset(
                            X_lgb[train_mask],
                            label=value_nonzero_target[train_mask],
                            weight=nz_train_w,
                            feature_name=feature_cols,
                            free_raw_data=False,
                        ),
                        num_boost_round=rounds_value,
                        valid_sets=[
                            lgb.Dataset(
                                X_lgb[val_mask],
                                label=value_nonzero_target[val_mask],
                                feature_name=feature_cols,
                                free_raw_data=False,
                            )
                        ],
                        callbacks=cbs,
                    )
                finally:
                    for fn in cl:
                        fn()
                l3_outer.set_postfix_str("value-nz", refresh=False)
                l3_outer.update(1)
            else:
                value_nonzero_model = None
            cbs, cl = _lgb_train_callbacks_with_round_tqdm(es_rounds, rounds_value, "[L3] value")
            try:
                value_train_mask = train_mask if value_nonzero_model is None else value_nonzero_train
                value_val_mask = val_mask if value_nonzero_model is None else value_nonzero_val
                if int(value_train_mask.sum()) < 20 or int(value_val_mask.sum()) < 10:
                    value_train_mask = train_mask
                    value_val_mask = val_mask
                value_model = lgb.train(
                    value_params,
                    lgb.Dataset(
                        X_lgb[value_train_mask],
                        label=y_value_fit[value_train_mask],
                        feature_name=feature_cols,
                        free_raw_data=False,
                    ),
                    num_boost_round=rounds_value,
                    valid_sets=[
                        lgb.Dataset(
                            X_lgb[value_val_mask],
                            label=y_value_fit[value_val_mask],
                            feature_name=feature_cols,
                            free_raw_data=False,
                        )
                    ],
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
    if value_disabled:
        value_pred_tune = np.zeros(int(np.sum(val_tune_mask)), dtype=np.float64)
        value_policy_mode = "prob_only"
        value_tie_margin = 0.03
    else:
        value_pred_tune = _l3_value_predict_hurdle(
            X_lgb[val_tune_mask],
            value_model,
            value_nonzero_model,
            prob_power=value_hurdle_prob_power,
        )
        value_policy_mode = _choose_l3_value_policy_mode(y_value[val_tune_mask], value_pred_tune)
        value_tie_margin = _derive_l3_value_tie_margin(y_value[val_tune_mask], value_pred_tune)
    policy_roll_min_rows = int(max(40, round(float(os.environ.get("L3_POLICY_ROLLING_MIN_ROWS", "80")))))
    policy_recalib_mode = (os.environ.get("L3_POLICY_RECALIB_MODE", "exp_decay") or "exp_decay").strip().lower()
    tune_idx = np.flatnonzero(val_tune_mask)
    if tune_idx.size:
        tune_idx = tune_idx[np.argsort(np.asarray(pd.to_datetime(t_state))[tune_idx])]
    if tune_idx.size == 0:
        tune_idx = np.flatnonzero(np.asarray(val_tune_mask, dtype=bool))
    policy_signal = y_value if not value_disabled else y_exit.astype(np.float64)
    if policy_recalib_mode == "exp_decay":
        tune_weights, weight_meta = _l3_exp_decay_weights(
            tune_idx,
            t_state,
            np.asarray(policy_signal, dtype=np.float64),
            min_rows=policy_roll_min_rows,
        )
    else:
        tune_weights = np.ones(tune_idx.size, dtype=np.float64) / max(tune_idx.size, 1)
        weight_meta = {"mode": "uniform", "reason": f"policy_recalib_mode={policy_recalib_mode}"}
    print(
        f"  [L3] policy recalibration: mode={weight_meta.get('mode')}  rows={tune_idx.size}  "
        f"half_life_rows={weight_meta.get('half_life_rows', float('nan'))}  "
        f"lambda={weight_meta.get('lambda', float('nan'))}",
        flush=True,
    )
    exit_state_keys = _l3_exit_policy_row_state_keys(
        X[tune_idx],
        static_cols,
        vol_quantiles=[float(x) for x in (dataset_policy.get("policy_state_vol_quantiles") or [])],
    )
    hold_idx = static_cols.index("l3_hold_bars")
    report_guardrail: dict[str, float] = {}
    report_idx = np.flatnonzero(val_report_mask)
    if report_idx.size:
        report_exit_rate_hint = float(np.mean(y_exit[report_idx].astype(np.float64)))
        report_hold_recall_floor_hint = float(np.clip((1.0 - report_exit_rate_hint) * 0.5, 0.10, 0.75))
        report_guardrail = {
            "exit_rate_hint": report_exit_rate_hint,
            "hold_recall_floor_hint": report_hold_recall_floor_hint,
        }
        print(
            f"  [L3] report guardrail hints: exit_rate_hint={report_exit_rate_hint:.4f}  "
            f"hold_recall_floor_hint={report_hold_recall_floor_hint:.4f}",
            flush=True,
        )
    exit_policy, exit_policy_by_state = _l3_search_conditional_exit_policy(
        exit_state_keys,
        _apply_l3_exit_calibrator(exit_model.predict(X_lgb[tune_idx]).astype(np.float64), exit_calibrator),
        _l3_value_predict_hurdle(
            X_lgb[tune_idx],
            value_model,
            value_nonzero_model,
            prob_power=value_hurdle_prob_power,
        ) if not value_disabled else np.zeros(int(tune_idx.size), dtype=np.float64),
        y_exit[tune_idx],
        y_value_true=y_value[tune_idx],
        value_policy_mode=value_policy_mode,
        value_tie_margin=value_tie_margin,
        sample_weight=tune_weights,
        hold_bars=X[tune_idx, hold_idx].astype(np.float64),
        report_guardrail=report_guardrail,
    )
    _ip = _l3_exit_infer_params(None)
    print(
        f"  [L3] live exit smoothing check: smooth={_l3_exit_infer_enabled(None)}  "
        f"ema_alpha={_ip['ema_alpha']:.2f}  "
        f"hysteresis=({_ip['enter_thr']:.2f},{_ip['leave_thr']:.2f})  "
        f"min_hold={int(round(_ip['min_hold']))}",
        flush=True,
    )
    if use_hybrid and not value_disabled and os.environ.get("L3_GRU_ABLATION", "1").strip().lower() in {"1", "true", "yes"}:
        print("  [L3] GRU ablation: fitting static-only comparators...", flush=True)
        static_exit_model, static_value_model = _fit_l3_static_ablation(
            X,
            y_exit,
            y_value_fit,
            train_mask,
            val_mask,
            rows_entry,
            static_cols,
            rounds=rounds_exit,
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
        v_hybrid = _l3_value_predict_hurdle(
            X_lgb[val_report_mask],
            value_model,
            value_nonzero_model,
            prob_power=value_hurdle_prob_power,
        )
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
        print(
            f"  [L3] gain importance share — exit: static={st_sh:.1%}  traj_emb={em_sh:.1%}",
            flush=True,
        )
        if value_model is not None:
            st_v, em_v = l3_trajectory_embed_importance_ratio(value_model, n_static, traj_cfg.embed_dim)
            print(f"  [L3] gain importance share — value: static={st_v:.1%}  traj_emb={em_v:.1%}", flush=True)
    eval_summary = _log_l3_val_extended(
        X_lgb,
        y_exit,
        y_value,
        t_state,
        feature_cols,
        val_report_mask,
        test_mask,
        exit_model,
        value_model,
        value_nonzero_model=value_nonzero_model,
        value_hurdle_prob_power=value_hurdle_prob_power,
        exit_calibrator=exit_calibrator,
        value_policy_mode=value_policy_mode,
        value_tie_margin=value_tie_margin,
        exit_policy_summary=exit_policy,
    )
    release_hold_recall_min = _env_float_clipped("L3_RELEASE_MIN_HOLD_RECALL", 0.25, lo=0.0, hi=1.0)
    release_exit_rate_max = _env_float_clipped("L3_RELEASE_MAX_EXIT_RATE", 0.85, lo=0.0, hi=1.0)
    release_holdout_gap_max = _env_float_clipped("L3_RELEASE_MAX_HOLD_RECALL_GAP", 0.05, lo=0.0, hi=1.0)
    hold_recall_release_flag = False
    hold_recall_release_reasons: list[str] = []
    val_hold_recall = float(eval_summary.get("val_hold_recall", float("nan")))
    val_exit_rate = float(eval_summary.get("val_exit_rate", float("nan")))
    holdout_hold_recall = float(eval_summary.get("holdout_hold_recall", float("nan")))
    if np.isfinite(val_hold_recall) and np.isfinite(val_exit_rate):
        if val_hold_recall < release_hold_recall_min and val_exit_rate > release_exit_rate_max:
            hold_recall_release_flag = True
            hold_recall_release_reasons.append(
                f"val_hold_recall<{release_hold_recall_min:.2f}&val_exit_rate>{release_exit_rate_max:.2f}"
            )
    if np.isfinite(val_hold_recall) and np.isfinite(holdout_hold_recall):
        if holdout_hold_recall < (val_hold_recall - release_holdout_gap_max):
            hold_recall_release_flag = True
            hold_recall_release_reasons.append(
                f"holdout_hold_recall_drop>{release_holdout_gap_max:.2f}"
            )
    if hold_recall_release_flag:
        print(
            "  [L3][WARN][RELEASE_GUARD] hold-recall acceptance failed: "
            + ", ".join(hold_recall_release_reasons),
            flush=True,
        )
    os.makedirs(MODEL_DIR, exist_ok=True)
    exit_model.save_model(os.path.join(MODEL_DIR, L3_EXIT_FILE))
    model_files: dict[str, str] = {"exit": L3_EXIT_FILE}
    if not value_disabled and value_model is not None:
        value_model.save_model(os.path.join(MODEL_DIR, L3_VALUE_FILE))
        model_files["value"] = L3_VALUE_FILE
    if not value_disabled and value_nonzero_model is not None:
        value_nz_file = "l3_value_nonzero_model.txt"
        value_nonzero_model.save_model(os.path.join(MODEL_DIR, value_nz_file))
        model_files["value_nonzero"] = value_nz_file
    if exit_calibrator is not None:
        exit_calib_file = "l3_exit_calibrator.pkl"
        with open(os.path.join(MODEL_DIR, exit_calib_file), "wb") as f:
            pickle.dump(exit_calibrator, f)
        model_files["exit_calibrator"] = exit_calib_file
    cox_path = os.path.join(MODEL_DIR, L3_COX_FILE)
    if cox_bundle.get("l3_cox_fitted") and cox_bundle.get("fitter") is not None:
        with open(cox_path, "wb") as f:
            pickle.dump(
                {
                    "fitter": cox_bundle["fitter"],
                    "cov_names": cox_bundle["cov_names"],
                    "l3_cox_fitted": True,
                },
                f,
            )
        print(f"  [L3] Cox artifact saved -> {cox_path}", flush=True)
    elif os.path.exists(cox_path):
        try:
            os.remove(cox_path)
        except OSError:
            pass
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
            "l3_drawdown_from_peak_atr",
            "l3_price_velocity_3bar_atr",
            "l3_feature_momentum_regdiv_3bar",
            "l3_vol_surprise_accel",
            "l3_regime_stability_3bar",
            *PA_STATE_FEATURES,
            "l3_signal_conf_decay",
            "l3_signal_direction_agree",
            "l3_regime_changed",
            "l3_l2_gate_current",
            "l3_l2_gate_decay",
            "l3_would_enter_now",
            "l3_regret_ratio",
            "l3_bars_since_peak",
            "l3_at_new_high",
            "l3_regret_velocity",
            "l3_trade_quality_bayes",
            "l3_cox_log_partial_hazard",
            "l3_cox_baseline_cumhaz_at_stop",
        ],
        "l3_cox_fitted": bool(cox_bundle.get("l3_cox_fitted")),
        "l3_cox_artifact_file": L3_COX_FILE,
        "l3_cox_fit_reason": str(cox_bundle.get("reason", "")),
        "l3_cox_dropped_covariates": dict(cox_bundle.get("dropped_covariates", {})),
        "l3_cox_drawdown_stabilization": dict(cox_bundle.get("drawdown_stabilization", {})),
        "l3_value_disabled": bool(value_disabled),
        "l3_value_mode": "disabled" if value_disabled else "full",
        "l3_value_head_type": "hurdle_two_stage" if value_nonzero_model is not None else ("single_regression" if not value_disabled else "disabled"),
        "l3_value_hurdle_enabled": bool(value_nonzero_model is not None),
        "l3_value_hurdle_nonzero_epsilon": float(value_hurdle_eps),
        "l3_value_hurdle_prob_power": float(value_hurdle_prob_power),
        "l3_exit_state_granularity": os.environ.get("L3_EXIT_STATE_GRANULARITY", "coarse"),
        "l3_exit_boost_rounds": int(rounds_exit),
        "l3_exit_infer_smooth": os.environ.get("L3_EXIT_INFER_SMOOTH", "1").strip().lower() in {"1", "true", "yes"},
        "l3_exit_ema_alpha": float(os.environ.get("L3_EXIT_EMA_ALPHA", "0.3")),
        "l3_exit_hyst_enter": float(os.environ.get("L3_EXIT_HYST_ENTER", "0.55")),
        "l3_exit_hyst_leave": float(os.environ.get("L3_EXIT_HYST_LEAVE", "0.35")),
        "l3_min_hold_bars": int(max(0, round(float(os.environ.get("L3_MIN_HOLD_BARS", "3"))))),
        "l3_hybrid": bool(use_hybrid),
        "l3_split_config": {
            "oot_start": str(oot_start),
            "holdout_start": str(holdout_start),
            "strict_trade_level_split": os.environ.get("L3_STRICT_TRADE_LEVEL_SPLIT", "1").strip().lower() in {"1", "true", "yes"},
        },
        "l3_val_tune_frac": val_tune_frac,
        "l3_allow_truth_fallback": os.environ.get("L3_ALLOW_TRUTH_FALLBACK", "0").strip().lower() in {"1", "true", "yes"},
        "l3_entry_min_confidence": entry_min_confidence,
        "l3_entry_min_size": entry_min_size,
        "policy_state_vol_quantiles": dataset_policy.get("policy_state_vol_quantiles", []),
        "l3_entry_policy_by_state": dataset_policy.get("l3_entry_policy_by_state", {}),
        "l3_target_horizon_bars": int(dataset_policy.get("l3_target_horizon_bars", _l3_target_horizon_bars(30))),
        "l3_target_horizon_bars_by_state": dataset_policy.get("l3_target_horizon_bars_by_state", {}),
        "l3_target_horizon_default_source": "env_or_default_5bar_policy",
        "l3_exit_epsilon_atr": _l3_exit_epsilon_atr(),
        "l3_exit_loss_buffer_atr": _l3_exit_loss_buffer_atr(),
        "l3_exit_live_edge_floor": _l3_exit_live_edge_floor(),
        "l3_exit_prob_threshold": float(exit_policy["exit_prob_threshold"]),
        "l3_exit_prob_threshold_early": float(exit_policy.get("exit_prob_threshold_early", exit_policy["exit_prob_threshold"])),
        "l3_exit_prob_threshold_late": float(exit_policy.get("exit_prob_threshold_late", exit_policy["exit_prob_threshold"])),
        "l3_policy_early_hold_split_bar": int(exit_policy.get("early_hold_split_bar", 3)),
        "l3_policy_early_prob_threshold_delta": float(exit_policy.get("early_prob_threshold_delta", 0.0)),
        "l3_value_left_threshold": float(exit_policy["value_left_threshold"]),
        "l3_value_policy_mode": str(exit_policy.get("value_policy_mode", value_policy_mode)),
        "l3_value_tie_margin": float(exit_policy.get("value_tie_margin", value_tie_margin)),
        "l3_policy_recalib_mode": str(policy_recalib_mode),
        "l3_policy_recalib_weighting": dict(weight_meta),
        "l3_value_mode_selector": {
            "selected_mode": value_policy_mode,
            "tie_margin": float(value_tie_margin),
            "selection_rule": "tie_break iff corr>=0.03 and pred_std>=0.02 and r2>=-0.05; else prob_only (policy search objective uses utility on y_value_true)",
        },
        "l3_exit_policy_search": exit_policy,
        "l3_exit_policy_by_state": exit_policy_by_state,
        "l3_release_guard": {
            "enabled": True,
            "flagged": bool(hold_recall_release_flag),
            "reasons": list(hold_recall_release_reasons),
            "thresholds": {
                "min_hold_recall": float(release_hold_recall_min),
                "max_exit_rate": float(release_exit_rate_max),
                "max_holdout_drop": float(release_holdout_gap_max),
            },
            "metrics": {
                "val_hold_recall": float(val_hold_recall),
                "val_exit_rate": float(val_exit_rate),
                "holdout_hold_recall": float(holdout_hold_recall),
                "val_auc": float(eval_summary.get("val_auc", float("nan"))),
                "holdout_auc": float(eval_summary.get("holdout_auc", float("nan"))),
            },
        },
        "pa_state_features": list(PA_STATE_FEATURES),
        "pa_policy_semantics": "PA buckets expand L3 entry/exit/horizon state policies and tighten holds in fragile price-action contexts",
        "pa_target_semantics": str(dataset_policy.get("pa_target_semantics", "PA-aware continuation thresholds and target horizon scaling are applied inside L3 supervision")),
        "l3_target_semantics": "fixed_deadline_continuation_with_meaningful_hold_class",
        "l3_gru_default_enabled": False,
        "l3_exit_class_weights": {
            "hold": float(hold_neg_w),
            "exit": float(exit_pos_w),
        },
        "l3_value_training": value_prep,
        "l3_boost_rounds": int(rounds_value),
        "l3_early_stopping_rounds": int(es_rounds),
        "l3_exit_lgb_config": exit_cfg,
        "l3_value_lgb_config": value_cfg,
        "l3_feature_gain_selection": dict(l3_gain_sel_meta),
        "l3_gru_min_rows": {
            "train": int(gru_min_train_rows),
            "val": int(gru_min_val_rows),
        },
        "rollback_criteria": {
            "p1a_abstain": {"rollback_if_l2_gate_auc_drop_gt": 0.01},
            "p1c_hurdle_after_expansion": {"rollback_if_exit_auc_drop_gt": 0.02},
            "p2a_deterministic_prune": {"rollback_if_l2_sign_acc_drop_gt": 0.02},
        },
    }
    meta = attach_threshold_registry(
        meta,
        "l3",
        [
            threshold_entry(
                "l3_exit_prob_threshold",
                float(exit_policy["exit_prob_threshold"]),
                category="adaptive_candidate",
                role="state-conditioned exit probability cutoff",
                adaptive_hint="utility search on val_tune with recency weights",
            ),
            threshold_entry(
                "l3_exit_prob_threshold_early",
                float(exit_policy.get("exit_prob_threshold_early", exit_policy["exit_prob_threshold"])),
                category="adaptive_candidate",
                role="early-hold exit cutoff",
                adaptive_hint="segmented policy for hold<split_bar",
            ),
            threshold_entry(
                "l3_exit_prob_threshold_late",
                float(exit_policy.get("exit_prob_threshold_late", exit_policy["exit_prob_threshold"])),
                category="adaptive_candidate",
                role="post-early-hold exit cutoff",
                adaptive_hint="segmented policy for hold>=split_bar",
            ),
            threshold_entry(
                "l3_value_left_threshold",
                float(exit_policy["value_left_threshold"]),
                category="adaptive_candidate",
                role="value tie-break threshold",
                adaptive_hint="bias-adjusted utility search",
            ),
            threshold_entry(
                "L3_POLICY_ROLLING_FRACTION",
                float(weight_meta.get("half_life_rows", float("nan")) / max(tune_idx.size, 1)),
                category="adaptive_candidate",
                role="effective data fraction under exponential decay weighting",
                adaptive_hint="derived from AR(1) half-life rather than fixed cut",
                statistical_principle="autocorrelation_half_life",
            ),
            threshold_entry(
                "L3_POLICY_ROLLING_MIN_ROWS",
                int(policy_roll_min_rows),
                category="data_guardrail",
                role="minimum rows required to activate rolling window",
            ),
            threshold_entry(
                "L3_POLICY_RECALIB_MODE",
                str(policy_recalib_mode),
                category="adaptive_candidate",
                role="policy recalibration weighting mode",
                statistical_principle="exponential_decay_weighting",
                method_selected=str(weight_meta.get("mode", policy_recalib_mode)),
                fallback_reason=str(weight_meta.get("reason", "")),
            ),
            threshold_entry(
                "L3_MIN_HOLD_BARS",
                int(max(0, round(float(os.environ.get("L3_MIN_HOLD_BARS", "3"))))),
                category="safety_constraint",
                role="live churn control min hold",
            ),
            threshold_entry(
                "L3_EXIT_LOSS_BUFFER_ATR",
                float(_l3_exit_loss_buffer_atr()),
                category="safety_constraint",
                role="loss buffer safety floor",
            ),
        ],
    )
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
    if value_nonzero_model is not None:
        bundle_models["value_nonzero"] = value_nonzero_model
    if exit_calibrator is not None:
        bundle_models["exit_calibrator"] = exit_calibrator
    train_finished_at = datetime.now().astimezone()
    elapsed_sec = max(0.0, time.perf_counter() - train_started_perf)
    print(
        f"  [L3] training finished at {train_finished_at.strftime('%Y-%m-%d %H:%M:%S %z')}  "
        f"elapsed={elapsed_sec:.1f}s",
        flush=True,
    )
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
    models: dict[str, Any] = {}
    for name, fname in model_files.items():
        if name == "exit_calibrator" or not fname:
            continue
        path = os.path.join(MODEL_DIR, str(fname))
        if name == "value" and not os.path.exists(path):
            continue
        models[name] = lgb.Booster(model_file=path)
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

    from core.trainers.l3.trajectory import L3TrajectoryEncoder

    cfg_d = meta.get("l3_traj_cfg")
    if not cfg_d:
        return None, None
    cfg = L3TrajectoryConfig(**cfg_d)
    enc = L3TrajectoryEncoder(cfg)
    path = os.path.join(MODEL_DIR, meta.get("l3_traj_encoder_file", L3_TRAJECTORY_ENCODER_FILE))
    enc.load_state_dict(torch.load(path, map_location="cpu"))
    enc.eval()
    return enc, cfg
