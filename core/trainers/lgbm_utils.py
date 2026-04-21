from __future__ import annotations

import gc
import os
import pickle
import sys
import warnings
from typing import Any

import lightgbm as lgb
import optuna
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, confusion_matrix
from tqdm.auto import tqdm

# Default tqdm stream (stderr). When ``train_pipeline.Logger`` tees stdout, use ``_tqdm_stream()``
# so bars go to the same real console as ``Logger.terminal`` (some terminals only show stdout).
TQDM_FILE = getattr(sys, "__stderr__", None) or sys.stderr


def _tqdm_stream():
    """Text stream for tqdm: ``Logger.terminal`` if stdout is a train_pipeline Logger, else stderr."""
    out = sys.stdout
    term = getattr(out, "terminal", None)
    if term is not None:
        return term
    return TQDM_FILE


from core.indicators import atr as compute_atr
from core.pa_rules import add_pa_features
from core.tcn_pa_state import PAStateTCN, FocalLoss

from core.trainers.constants import *

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _tcn_bottleneck_dim_from_meta() -> int:
    meta_path = os.path.join(MODEL_DIR, "tcn_meta.pkl")
    if not os.path.isfile(meta_path):
        return TCN_BOTTLENECK_DIM
    import pickle

    with open(meta_path, "rb") as f:
        m = pickle.load(f)
    return int(m.get("bottleneck_dim", TCN_BOTTLENECK_DIM))

def _mamba_bottleneck_dim_from_meta() -> int:
    meta_path = os.path.join(MODEL_DIR, "mamba_meta.pkl")
    if not os.path.isfile(meta_path):
        return TCN_BOTTLENECK_DIM
    import pickle

    with open(meta_path, "rb") as f:
        m = pickle.load(f)
    return int(m.get("bottleneck_dim", TCN_BOTTLENECK_DIM))

def _tcn_derived_feature_names(bottleneck_dim: int | None = None) -> list[str]:
    bd = int(bottleneck_dim) if bottleneck_dim is not None else _tcn_bottleneck_dim_from_meta()
    emb = [f"tcn_emb_{i}" for i in range(bd)]
    return TCN_REGIME_FUT_PROB_COLS + [TCN_TRANSITION_PROB_COL, "tcn_regime_fut_entropy", TCN_BARRIER_DIR_DIFF_COL, "tcn_is_warm"] + emb

def _mamba_derived_feature_names(bottleneck_dim: int | None = None) -> list[str]:
    bd = int(bottleneck_dim) if bottleneck_dim is not None else _mamba_bottleneck_dim_from_meta()
    emb = [f"mamba_emb_{i}" for i in range(bd)]
    return MAMBA_REGIME_FUT_PROB_COLS + ["mamba_regime_fut_entropy", "mamba_is_warm"] + emb


def configure_compute_threads() -> None:
    """
    Conservative defaults to reduce CPU oversubscription (PyTorch + LightGBM OpenMP + BLAS).

    Env:
      TORCH_CPU_THREADS — PyTorch intra-op threads (default: ceil(n_cpu/2), capped at 8).
      LGBM_N_JOBS — passed to LightGBM train() (default: same cap rule, max 16).
    """
    n_cpu = max(1, os.cpu_count() or 4)
    default_cap = max(1, min(8, (n_cpu + 1) // 2))
    n = int(os.environ.get("TORCH_CPU_THREADS", str(default_cap)))
    n = max(1, min(n, n_cpu))
    try:
        torch.set_num_threads(n)
        inter = max(1, min(2, max(1, n // 4)))
        torch.set_num_interop_threads(inter)
    except RuntimeError:
        pass


def configure_cuda_training_speedups() -> None:
    """Best-effort CUDA throughput toggles (no-ops if no CUDA).

    Opt out: TORCH_CUDNN_BENCHMARK=0, TORCH_ALLOW_TF32=0.
    Precision: TORCH_MATMUL_PRECISION=highest|high|medium (default high).
    """
    if not torch.cuda.is_available():
        return
    if os.environ.get("TORCH_CUDNN_BENCHMARK", "1").strip().lower() not in {"0", "false", "no", "off"}:
        torch.backends.cudnn.benchmark = True
    if os.environ.get("TORCH_ALLOW_TF32", "1").strip().lower() not in {"0", "false", "no", "off"}:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    prec = (os.environ.get("TORCH_MATMUL_PRECISION", "high") or "high").strip().lower()
    if prec not in {"highest", "high", "medium"}:
        prec = "high"
    try:
        torch.set_float32_matmul_precision(prec)
    except Exception:
        pass


def configure_training_runtime() -> None:
    """CPU thread budget + CUDA matmul/cudnn defaults for training scripts."""
    configure_compute_threads()
    configure_cuda_training_speedups()


def _lgbm_n_jobs() -> int:
    n_cpu = max(1, os.cpu_count() or 4)
    default = max(1, min(16, (n_cpu + 1) // 2))
    return int(os.environ.get("LGBM_N_JOBS", str(default)))


def _tcn_inference_device() -> torch.device:
    forced = os.environ.get("TORCH_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda", "mps"):
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_lgbm_string_tag_col(name: str) -> bool:
    if name in LGBM_EXCLUDE_PA_STRING_COLS:
        return True
    if name.startswith("pa_htf_") and name.endswith("_state"):
        return True
    return False


def _numeric_feature_cols_for_matrix(df: pd.DataFrame, names: list[str]) -> list[str]:
    """Keep only columns present in df and numeric/bool dtypes (defensive for float32 matrices)."""
    keep: list[str] = []
    dropped: list[str] = []
    for c in names:
        if c not in df.columns:
            dropped.append(f"{c}<missing>")
            continue
        if _is_lgbm_string_tag_col(c):
            dropped.append(c)
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            keep.append(c)
        else:
            dropped.append(f"{c}<{s.dtype}>")
    if dropped:
        preview = dropped[:24]
        more = " …" if len(dropped) > 24 else ""
        print(
            f"  [LGBM] Excluded {len(dropped)} non-numeric / tag columns from matrix: {preview}{more}",
            flush=True,
        )
    return keep


def _unique_cols(cols: list[str]) -> list[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _require_lgb_matrix_matches_names(
    X: np.ndarray, feature_names: list[str], context: str
) -> None:
    """LightGBM requires len(feature_name) == num_feature; fail fast with a clear error."""
    n_x = int(X.shape[1]) if X.ndim == 2 else 0
    n_n = len(feature_names)
    if n_x != n_n:
        raise ValueError(
            f"{context}: matrix has {n_x} columns but feature_name has {n_n} entries."
        )
    seen: set[str] = set()
    dup: list[str] = []
    for c in feature_names:
        if c in seen:
            dup.append(c)
        else:
            seen.add(c)
    if dup:
        raise ValueError(
            f"{context}: duplicate feature names: {sorted(set(dup))!r}"
        )


def _lgbm_booster_feature_names(model: lgb.Booster) -> list[str]:
    """Columns in training order; required for chunked predict to match Layer 2a."""
    names = [str(x) for x in model.feature_name()]
    if not names:
        raise ValueError("LightGBM booster returned no feature names.")
    if names[0].startswith("Column_"):
        raise ValueError(
            "Regime booster has generic Column_* names; retrain Layer 2a with named features."
        )
    return names


def _split_feature_groups(
    feat_cols: list[str],
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], list[str]]:
    hmm = [c for c in feat_cols if c.startswith("pa_hmm_")]
    garch = [c for c in feat_cols if c.startswith("pa_garch_")]
    hsmm = [c for c in feat_cols if c.startswith("pa_hsmm_")]
    egarch = [c for c in feat_cols if c.startswith("pa_egarch_")]
    tcn = [c for c in feat_cols if c.startswith("tcn_")]
    mamba = [c for c in feat_cols if c.startswith("mamba_")]
    base = [c for c in feat_cols if c not in set(hmm + garch + hsmm + egarch + tcn + mamba)]
    return base, hmm, garch, hsmm, egarch, tcn, mamba


def _regime_lgbm_feature_cols(feat_cols: list[str]) -> list[str]:
    """PA + volatility stats only for regime head (no TCN/Mamba/HMM-style state columns)."""
    return [
        c
        for c in feat_cols
        if not c.startswith("tcn_")
        and not c.startswith("mamba_")
        and not c.startswith("pa_hmm_")
        and not c.startswith("pa_hsmm_")
    ]


def _tq(it, **kwargs):
    """Progress bar. DISABLE_TQDM=1 forces off; FORCE_TQDM=1 forces on even when logging to a file."""
    d = os.environ.get("DISABLE_TQDM", "").strip().lower()
    if d in {"1", "true", "yes"}:
        return it
    tf = _tqdm_stream()
    if not tf.isatty():
        if os.environ.get("FORCE_TQDM", "").strip().lower() not in {"1", "true", "yes"}:
            return it
    kw = {"file": tf, **kwargs}
    return tqdm(it, **kw)


def _lgb_log_eval_period() -> int:
    """Valid-set log interval for LightGBM; 0 = omit log_evaluation. Override: LGBM_LOG_EVAL_PERIOD."""
    raw = os.environ.get("LGBM_LOG_EVAL_PERIOD", "").strip()
    if raw:
        return max(0, int(raw))
    return 100 if _tqdm_stream().isatty() else 0


def _lgb_train_callbacks(early_stopping_rounds: int, *, first_metric_only: bool = True) -> list:
    p = _lgb_log_eval_period()
    out = []
    if p > 0:
        out.append(lgb.log_evaluation(p))
    out.append(lgb.early_stopping(early_stopping_rounds, first_metric_only=first_metric_only))
    return out


def _options_target_config() -> dict[str, float]:
    decision_horizon = max(1, int(os.environ.get("OPTION_DECISION_HORIZON_BARS", "20")))
    theta_start = max(0, int(os.environ.get("OPTION_THETA_START_BARS", "15")))
    theta_decay = max(float(os.environ.get("OPTION_THETA_DECAY_BARS", "5.0")), 1e-6)
    adverse_penalty = max(float(os.environ.get("OPTION_ADVERSE_PENALTY", "1.0")), 0.0)
    max_hold = max(decision_horizon, int(os.environ.get("OPTION_MAX_HOLD_BARS", "30")))
    return {
        "decision_horizon_bars": decision_horizon,
        "theta_start_bars": theta_start,
        "theta_decay_bars": theta_decay,
        "adverse_penalty": adverse_penalty,
        "max_hold_bars": max_hold,
    }


def _theta_decay_from_bars(bars: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    cfg = _options_target_config()
    hold_time = np.maximum(np.asarray(bars, dtype=float), 0.0)
    return np.exp(-np.maximum(hold_time - cfg["theta_start_bars"], 0.0) / cfg["theta_decay_bars"])


def _decision_edge_atr_array(df: pd.DataFrame) -> np.ndarray:
    if "decision_net_edge_atr" in df.columns:
        return df["decision_net_edge_atr"].fillna(0.0).values.astype(np.float64)
    if {"decision_mfe_atr", "decision_mae_atr", "decision_theta_decay"}.issubset(df.columns):
        mfe = np.clip(df["decision_mfe_atr"].fillna(0.0).values.astype(np.float64), 0.0, 5.0)
        mae = np.clip(df["decision_mae_atr"].fillna(0.0).values.astype(np.float64), 0.0, 4.0)
        theta_decay = np.clip(df["decision_theta_decay"].fillna(1.0).values.astype(np.float64), 0.0, 1.0)
        cfg = _options_target_config()
        return mfe * theta_decay - cfg["adverse_penalty"] * mae
    if {"decision_mfe_atr", "decision_mae_atr", "decision_peak_bar"}.issubset(df.columns):
        mfe = np.clip(df["decision_mfe_atr"].fillna(0.0).values.astype(np.float64), 0.0, 5.0)
        mae = np.clip(df["decision_mae_atr"].fillna(0.0).values.astype(np.float64), 0.0, 4.0)
        peak_bar = df["decision_peak_bar"].fillna(0).values.astype(float)
        cfg = _options_target_config()
        return mfe * _theta_decay_from_bars(peak_bar) - cfg["adverse_penalty"] * mae
    raise RuntimeError(
        "Missing decision-window label columns for L2/L3 targets. "
        "Expected `decision_net_edge_atr` or (`decision_mfe_atr`, `decision_mae_atr`, `decision_theta_decay`/`decision_peak_bar`). "
        "Regenerate labels with `data_tools/label_v2.py` before training."
    )


def _optimal_exit_target_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = _options_target_config()
    if {"optimal_tp_atr", "optimal_sl_atr", "optimal_exit_bar"}.issubset(df.columns):
        tp = df["optimal_tp_atr"].fillna(0.0).values.astype(np.float64)
        sl = df["optimal_sl_atr"].fillna(0.0).values.astype(np.float64)
        tm = df["optimal_exit_bar"].fillna(0.0).values.astype(np.float64)
        return tp, sl, np.clip(tm, 1.0, float(cfg["max_hold_bars"]))
    lbl_atr = df["lbl_atr"].values
    safe_atr = np.where(lbl_atr > 1e-3, lbl_atr, 1e-3)
    mfe = np.clip(df["max_favorable"].values / safe_atr, 0.0, 6.0)
    mae = np.clip(df["max_adverse"].values / safe_atr, 0.0, 4.0)
    exit_bar = df["exit_bar"].fillna(0).values.astype(float) if "exit_bar" in df.columns else np.zeros(len(df), dtype=float)
    return mfe * _theta_decay_from_bars(exit_bar), mae, np.clip(exit_bar, 1.0, float(cfg["max_hold_bars"]))


L4_POLICY_DYNAMIC_FEATURES = [
    "l4_hold_bars",
    "l4_hold_frac",
    "l4_side",
    "l4_unreal_pnl_atr",
    "l4_mfe_atr_live",
    "l4_mae_atr_live",
    "l4_net_edge_atr_live",
    "l4_theta_decay_live",
    "l4_setup_aligned_now",
    "l4_setup_retention",
    "l4_structure_veto_now",
    "l4_premise_break_now",
    "l4_opposing_setup_now",
    "l4_follow_through_gap",
]


def _net_edge_atr_from_state(
    mfe_atr: np.ndarray | float,
    mae_atr: np.ndarray | float,
    hold_bars: np.ndarray | float,
) -> np.ndarray:
    cfg = _options_target_config()
    mfe = np.asarray(mfe_atr, dtype=float)
    mae = np.asarray(mae_atr, dtype=float)
    hold = np.asarray(hold_bars, dtype=float)
    return mfe * _theta_decay_from_bars(hold) - cfg["adverse_penalty"] * mae


def _live_trade_state_from_bar(
    *,
    side: float,
    entry_price: float,
    atr: float,
    high_price: float,
    low_price: float,
    close_price: float,
) -> tuple[float, float, float]:
    safe_atr = max(float(atr), 1e-6)
    if float(side) > 0.0:
        fav = max(0.0, (float(high_price) - float(entry_price)) / safe_atr)
        adv = max(0.0, (float(entry_price) - float(low_price)) / safe_atr)
        unreal = (float(close_price) - float(entry_price)) / safe_atr
    else:
        fav = max(0.0, (float(entry_price) - float(low_price)) / safe_atr)
        adv = max(0.0, (float(high_price) - float(entry_price)) / safe_atr)
        unreal = (float(entry_price) - float(close_price)) / safe_atr
    return float(fav), float(adv), float(unreal)


def _layer4_policy_feature_names(base_feature_cols: list[str]) -> list[str]:
    return list(base_feature_cols) + list(L4_POLICY_DYNAMIC_FEATURES)


def _layer4_policy_state_vector(
    base_row: np.ndarray,
    *,
    hold_bars: float,
    max_hold_bars: float,
    side: float,
    unreal_pnl_atr: float,
    mfe_atr_live: float,
    mae_atr_live: float,
    setup_aligned_now: float = 0.0,
    setup_retention: float = 0.0,
    structure_veto_now: float = 0.0,
    premise_break_now: float = 0.0,
    opposing_setup_now: float = 0.0,
    follow_through_gap: float = 0.0,
) -> np.ndarray:
    hold_arr = np.asarray([hold_bars], dtype=np.float32)
    dyn = np.array(
        [
            float(hold_bars),
            float(hold_bars / max(max_hold_bars, 1.0)),
            float(side),
            float(unreal_pnl_atr),
            float(mfe_atr_live),
            float(mae_atr_live),
            float(_net_edge_atr_from_state(mfe_atr_live, mae_atr_live, hold_bars)),
            float(_theta_decay_from_bars(hold_arr)[0]),
            float(setup_aligned_now),
            float(setup_retention),
            float(structure_veto_now),
            float(premise_break_now),
            float(opposing_setup_now),
            float(follow_through_gap),
        ],
        dtype=np.float32,
    )
    return np.concatenate([np.asarray(base_row, dtype=np.float32), dyn], axis=0)


def _apply_structure_veto_to_gates(
    work: pd.DataFrame,
    p_long_gate: np.ndarray,
    p_short_gate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    long_setup = work["pa_ctx_setup_long"].to_numpy(dtype=np.float32, copy=False) if "pa_ctx_setup_long" in work.columns else np.zeros(len(work), dtype=np.float32)
    short_setup = work["pa_ctx_setup_short"].to_numpy(dtype=np.float32, copy=False) if "pa_ctx_setup_short" in work.columns else np.zeros(len(work), dtype=np.float32)
    range_pressure = work["pa_ctx_range_pressure"].to_numpy(dtype=np.float32, copy=False) if "pa_ctx_range_pressure" in work.columns else np.zeros(len(work), dtype=np.float32)
    premise_break_long = work["pa_ctx_premise_break_long"].to_numpy(dtype=np.float32, copy=False) if "pa_ctx_premise_break_long" in work.columns else np.zeros(len(work), dtype=np.float32)
    premise_break_short = work["pa_ctx_premise_break_short"].to_numpy(dtype=np.float32, copy=False) if "pa_ctx_premise_break_short" in work.columns else np.zeros(len(work), dtype=np.float32)
    structure_veto = work["pa_ctx_structure_veto"].to_numpy(dtype=np.float32, copy=False) if "pa_ctx_structure_veto" in work.columns else np.zeros(len(work), dtype=np.float32)

    long_allow = np.clip(0.20 + 0.95 * long_setup - 0.40 * range_pressure - 0.35 * premise_break_long - 0.20 * structure_veto, 0.0, 1.0)
    short_allow = np.clip(0.20 + 0.95 * short_setup - 0.40 * range_pressure - 0.35 * premise_break_short - 0.20 * structure_veto, 0.0, 1.0)
    return (
        p_long_gate.astype(np.float32, copy=False) * long_allow.astype(np.float32, copy=False),
        p_short_gate.astype(np.float32, copy=False) * short_allow.astype(np.float32, copy=False),
    )


def _lgb_round_tqdm_enabled() -> bool:
    if os.environ.get("DISABLE_TQDM", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("LGBM_DISABLE_ROUND_TQDM", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if not _tqdm_stream().isatty():
        if os.environ.get("FORCE_TQDM", "").strip().lower() not in {"1", "true", "yes"}:
            return False
    return True


def _lgb_train_callbacks_with_round_tqdm(
    early_stopping_rounds: int,
    num_boost_round: int,
    tqdm_desc: str,
    *,
    first_metric_only: bool = True,
) -> tuple[list, list]:
    """LightGBM callbacks plus per-boosting-round tqdm. Returns (callbacks, cleanup_fns)."""
    base = _lgb_train_callbacks(early_stopping_rounds, first_metric_only=first_metric_only)
    if not _lgb_round_tqdm_enabled() or num_boost_round <= 0:
        return base, []

    bar = tqdm(
        total=num_boost_round,
        desc=tqdm_desc,
        unit="round",
        leave=False,
        mininterval=0.2,
        file=_tqdm_stream(),
    )

    def _round_cb(env) -> None:
        bar.update(1)
        er = getattr(env, "evaluation_result_list", None) or []
        if er:
            parts: list[str] = []
            for tup in er:
                if len(tup) >= 3:
                    parts.append(f"{tup[1]}={float(tup[2]):.4f}")
            if parts:
                bar.set_postfix_str(" ".join(parts[:3]), refresh=False)

    def _cleanup() -> None:
        bar.close()

    return [_round_cb] + base, [_cleanup]


def _compute_sample_weights(y: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Class-balanced weights with temporal recency boost."""
    n = len(y)
    class_frequencies = np.bincount(y, minlength=NUM_REGIME_CLASSES).astype(float)
    class_w = np.zeros(NUM_REGIME_CLASSES, dtype=float)
    for c in range(NUM_REGIME_CLASSES):
        if class_frequencies[c] <= 0:
            continue
        freq = class_frequencies[c] / n
        class_w[c] = 1.0 / max(freq * NUM_REGIME_CLASSES, 1e-6)
    active = class_w > 0
    if active.any():
        class_w[active] /= class_w[active].mean()
    else:
        class_w[:] = 1.0

    weights = np.array([class_w[label] for label in y], dtype=float)

    ts = pd.to_datetime(timestamps)
    days_from_end = (ts.max() - ts).total_seconds() / 86400
    max_days = days_from_end.max()
    recency = 0.7 + 0.3 * (1.0 - days_from_end / max(max_days, 1))
    weights *= recency.values

    return weights


def _optuna_search_params(
    X_train, y_train, w_train,
    X_cal, y_cal,
    feat_cols: list[str],
    base_params: dict,
    n_trials: int = 30,
) -> dict:
    """Restricted Optuna search for sensitive hyperparameters to prevent overfitting."""
    mode_name = "FAST" if FAST_TRAIN_MODE else "FULL"
    print(f"\n  Optuna search over {n_trials} trials ({mode_name} mode):")
    _require_lgb_matrix_matches_names(X_train, feat_cols, "Layer 2a Optuna search")

    # Limit search space for faster processing when FAST_TRAIN_MODE is enabled
    if FAST_TRAIN_MODE:
        print("  → FAST_TRAIN_MODE is enabled. Skipping Optuna search and using default values.")
        best_defaults = {"num_leaves": 31, "learning_rate": 0.05}
        return {**base_params, **best_defaults}

    actual_trials = min(10, n_trials) if FAST_TRAIN_MODE else n_trials

    # We evaluate on Cal, so we want to be very careful not to overfit it
    # Use early stopping but fewer rounds to speed up the search
    optuna_rounds = 800 if FAST_TRAIN_MODE else 2000
    optuna_es = 40 if FAST_TRAIN_MODE else 60

    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train,
                             feature_name=feat_cols, free_raw_data=False)
    valid_data = lgb.Dataset(X_cal, label=y_cal,
                             feature_name=feat_cols, free_raw_data=False)

    def objective(trial):
        p = base_params.copy()
        p["num_leaves"] = trial.suggest_int("num_leaves", 15, 63)
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        p["feature_fraction"] = trial.suggest_float("feature_fraction", 0.5, 0.9)
        # Lock all other parameters to base_params to prevent small-data noise

        model = lgb.train(
            p, train_data, num_boost_round=optuna_rounds, valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(optuna_es, verbose=False),
            ],
        )
        if model.best_iteration < 10:
            raise optuna.TrialPruned()
        return model.best_score["valid_0"]["multi_logloss"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=actual_trials, show_progress_bar=True)

    best = study.best_params
    print(f"  → Best Optuna params (n={actual_trials}): num_leaves={best['num_leaves']}  "
          f"learning_rate={best['learning_rate']:.4f}  feature_fraction={best['feature_fraction']:.4f}  "
          f"logloss={study.best_value:.5f}")

    return {**base_params, **best}


def _mfe_mae_atr_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Decision-window MFE/MAE in ATR units from label_v2 outputs."""
    if {"decision_mfe_atr", "decision_mae_atr"}.issubset(df.columns):
        mfe = np.clip(df["decision_mfe_atr"].fillna(0.0).values.astype(np.float64), 0.0, 5.0)
        mae = np.clip(df["decision_mae_atr"].fillna(0.0).values.astype(np.float64), 0.0, 4.0)
        return mfe, mae
    raise RuntimeError(
        "Missing `decision_mfe_atr` / `decision_mae_atr` required for L2 regression heads. "
        "Regenerate labels with `data_tools/label_v2.py` before training."
    )


def _opp_regression_sample_weights(mfe_tgt: np.ndarray, regime_name: str) -> np.ndarray:
    """
    Differentiated percentile weighting to force the model to focus on high-MFE (breakout) samples.
    """
    weights = np.ones(len(mfe_tgt), dtype=np.float64)
    
    # Mask out the zeros (non-breakouts) to calculate meaningful percentiles on actual breakouts
    active_mask = mfe_tgt > 0.05
    if active_mask.sum() > 10:
        active_series = pd.Series(mfe_tgt[active_mask])
        pct = active_series.rank(pct=True).values
        
        weights[active_mask] = np.where(
            pct > 0.90, 5.0,          # top 10% of active breakouts
            np.where(pct > 0.70, 2.5,  # 70~90% of active breakouts
                     np.where(pct > 0.50, 1.5,  # 50~70% of active breakouts
                              1.0))
        )
    
    # Keep the existing domain-specific logic as a multiplier
    if regime_name.startswith("range"):
        weights = np.where(mfe_tgt < 0.2, weights * 0.3, weights)
        
    return weights


def _l2b_reg_objective_params() -> dict:
    obj = os.environ.get("L2B_REG_OBJECTIVE", "tweedie").strip().lower()
    if obj == "huber":
        return {"objective": "huber", "alpha": 0.9}
    if obj == "regression":
        return {"objective": "regression", "metric": "mae"}
    tweedie_power = float(os.environ.get("L2B_TWEEDIE_POWER", "1.5"))
    return {
        "objective": "tweedie",
        "tweedie_variance_power": tweedie_power,
        "metric": "tweedie"
    }


def _layer3_chunk_rows() -> int:
    """Row chunk size for Layer-3 batched LGBM predict (RAM). Override with LAYER3_CHUNK."""
    return max(4096, int(os.environ.get("LAYER3_CHUNK", "65536")))


def _stacking_calibration_front_fraction() -> float:
    """Front fraction of the calibration window reserved for upstream calibration/tuning."""
    raw = os.environ.get("STACKING_CAL_FRONT_FRACTION", "").strip()
    if not raw:
        return 0.60
    return float(np.clip(float(raw), 0.05, 1.0))


# `from module import *` skips leading-underscore names unless listed here.
__all__ = [
    "configure_compute_threads",
    "configure_cuda_training_speedups",
    "configure_training_runtime",
    "_compute_sample_weights",
    "_decision_edge_atr_array",
    "_is_lgbm_string_tag_col",
    "_l2b_reg_objective_params",
    "_layer3_chunk_rows",
    "_stacking_calibration_front_fraction",
    "_layer4_policy_feature_names",
    "_layer4_policy_state_vector",
    "L4_POLICY_DYNAMIC_FEATURES",
    "_apply_structure_veto_to_gates",
    "_lgb_log_eval_period",
    "_lgb_round_tqdm_enabled",
    "_lgb_train_callbacks",
    "_lgb_train_callbacks_with_round_tqdm",
    "TQDM_FILE",
    "_tqdm_stream",
    "_lgbm_booster_feature_names",
    "_lgbm_n_jobs",
    "_mfe_mae_atr_arrays",
    "_live_trade_state_from_bar",
    "_net_edge_atr_from_state",
    "_numeric_feature_cols_for_matrix",
    "_opp_regression_sample_weights",
    "_optimal_exit_target_arrays",
    "_options_target_config",
    "_optuna_search_params",
    "_regime_lgbm_feature_cols",
    "_require_lgb_matrix_matches_names",
    "_split_feature_groups",
    "_tcn_bottleneck_dim_from_meta",
    "_tcn_derived_feature_names",
    "_mamba_bottleneck_dim_from_meta",
    "_mamba_derived_feature_names",
    "_tcn_inference_device",
    "_theta_decay_from_bars",
    "_tq",
    "_unique_cols",
]
