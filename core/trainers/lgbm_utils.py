from __future__ import annotations

import gc
import os
import pickle
import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from core.indicators import atr as compute_atr
from core.pa_rules import add_pa_features
from core.tcn_pa_state import PAStateTCN, FocalLoss

from core.trainers.constants import *

def _tcn_bottleneck_dim_from_meta() -> int:
    meta_path = os.path.join(MODEL_DIR, "tcn_meta.pkl")
    if not os.path.isfile(meta_path):
        return TCN_BOTTLENECK_DIM
    import pickle

    with open(meta_path, "rb") as f:
        m = pickle.load(f)
    return int(m.get("bottleneck_dim", TCN_BOTTLENECK_DIM))


def _tcn_derived_feature_names(bottleneck_dim: int | None = None) -> list[str]:
    bd = int(bottleneck_dim) if bottleneck_dim is not None else _tcn_bottleneck_dim_from_meta()
    emb = [f"tcn_emb_{i}" for i in range(bd)]
    return TCN_REGIME_FUT_PROB_COLS + ["tcn_regime_fut_entropy"] + emb


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
    if name in _LGBM_EXCLUDE_PA_STRING_COLS:
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


def _split_feature_groups(feat_cols: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    hmm = [c for c in feat_cols if c.startswith("pa_hmm_")]
    garch = [c for c in feat_cols if c.startswith("pa_garch_")]
    tcn = [c for c in feat_cols if c.startswith("tcn_")]
    base = [c for c in feat_cols if c not in set(hmm + garch + tcn)]
    return base, hmm, garch, tcn


def _tq(it, **kwargs):
    """Progress bar. DISABLE_TQDM=1 forces off; FORCE_TQDM=1 forces on even when logging to a file."""
    d = os.environ.get("DISABLE_TQDM", "").strip().lower()
    if d in {"1", "true", "yes"}:
        return it
    if not sys.stderr.isatty():
        if os.environ.get("FORCE_TQDM", "").strip().lower() not in {"1", "true", "yes"}:
            return it
    return tqdm(it, **kwargs)


def _lgb_log_eval_period() -> int:
    """Valid-set log interval for LightGBM; 0 = omit log_evaluation. Override: LGBM_LOG_EVAL_PERIOD."""
    raw = os.environ.get("LGBM_LOG_EVAL_PERIOD", "").strip()
    if raw:
        return max(0, int(raw))
    return 100 if sys.stderr.isatty() else 0


def _lgb_train_callbacks(early_stopping_rounds: int) -> list:
    p = _lgb_log_eval_period()
    out = []
    if p > 0:
        out.append(lgb.log_evaluation(p))
    out.append(lgb.early_stopping(early_stopping_rounds))
    return out


def _lgb_round_tqdm_enabled() -> bool:
    if os.environ.get("DISABLE_TQDM", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("LGBM_DISABLE_ROUND_TQDM", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if not sys.stderr.isatty():
        if os.environ.get("FORCE_TQDM", "").strip().lower() not in {"1", "true", "yes"}:
            return False
    return True


def _lgb_train_callbacks_with_round_tqdm(
    early_stopping_rounds: int,
    num_boost_round: int,
    tqdm_desc: str,
) -> tuple[list, list]:
    """LightGBM callbacks plus per-boosting-round tqdm. Returns (callbacks, cleanup_fns)."""
    base = _lgb_train_callbacks(early_stopping_rounds)
    if not _lgb_round_tqdm_enabled() or num_boost_round <= 0:
        return base, []

    bar = tqdm(
        total=num_boost_round,
        desc=tqdm_desc,
        unit="round",
        leave=False,
        mininterval=0.2,
        file=sys.stderr,
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
        p["learning_rate"] = trial.suggest_float("learning_rate", 0.03, 0.1, log=True)
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
    study.optimize(objective, n_trials=actual_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  → Best Optuna params (n={actual_trials}): num_leaves={best['num_leaves']}  "
          f"learning_rate={best['learning_rate']:.4f}  logloss={study.best_value:.5f}")

    return {**base_params, **best}


def _mfe_mae_atr_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Same scaling as trade-quality KMeans targets (MFE/MAE in ATR units)."""
    lbl_atr = df["lbl_atr"].values
    safe_atr = np.where(lbl_atr > 1e-3, lbl_atr, 1e-3)
    mfe = np.clip(df["max_favorable"].values / safe_atr, 0.0, 5.0)
    mae = np.clip(df["max_adverse"].values / safe_atr, 0.0, 4.0)
    return mfe.astype(np.float64), mae.astype(np.float64)


def _opp_regression_sample_weights(mfe_tgt: np.ndarray, regime_name: str) -> np.ndarray:
    w = np.ones(len(mfe_tgt), dtype=np.float64)
    w[mfe_tgt > 1.0] = 5.0
    w[mfe_tgt > 2.0] = 15.0
    if regime_name.startswith("range"):
        w[mfe_tgt < 0.2] = 0.3
    return w


def _l2b_reg_objective_params() -> dict:
    obj = os.environ.get("L2B_REG_OBJECTIVE", "regression").strip().lower()
    if obj == "huber":
        return {"objective": "huber", "alpha": 0.9}
    return {"objective": "regression", "metric": "mae"}


def _layer3_chunk_rows() -> int:
    """Row chunk size for Layer-3 batched LGBM predict (RAM). Override with LAYER3_CHUNK."""
    return max(4096, int(os.environ.get("LAYER3_CHUNK", "65536")))


