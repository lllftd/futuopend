"""
Inference helpers for the layered PA pipeline (TCN + L2a + L2b + L3).

Used by OOS backtest; does not import training entrypoints (avoids circular imports).
"""
from __future__ import annotations

import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
import torch
from tqdm.auto import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.tcn_pa_state import PAStateTCN

OOS_PRED_CHUNK = max(4096, int(os.environ.get("OOS_PRED_CHUNK", "65536")))

OOS_START = os.environ.get("OOS_START", "2025-01-01")
OOS_END = os.environ.get("OOS_END", "2026-01-01")

MODEL_DIR = os.path.join(_REPO_ROOT, "lgbm_models")
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.environ.get("OOS_RESULTS_DIR", os.path.join(_REPO_ROOT, "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)


def _tq(it, **kwargs):
    if os.environ.get("DISABLE_TQDM", "").strip() in {"1", "true", "yes"}:
        return it
    return tqdm(it, **kwargs)


# Order MUST match train_lgbm_pa_state.{REGIMES_6, STATE_NAMES, REGIME_NOW_PROB_COLS}:
# L2a class index k, calibrator k, prob column k, argmax → REGIMES_6[k].
REGIMES_6 = (
    "bull_conv",
    "bull_div",
    "bear_conv",
    "bear_div",
    "range_conv",
    "range_div",
)


def _compute_opportunity_triplet_infer(
    X: np.ndarray,
    regime_probs: np.ndarray,
    models: dict[str, dict[str, lgb.Booster]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # st[i] = L2a argmax k → models[REGIMES_6[k]]["mfe"|"mae"]
    st = np.argmax(regime_probs, axis=1).astype(np.int64, copy=False)
    n = len(X)
    opp = np.zeros(n, dtype=np.float64)
    mfe_p = np.zeros(n, dtype=np.float64)
    mae_p = np.zeros(n, dtype=np.float64)
    for argmax_idx, predicted_regime in enumerate(REGIMES_6):
        if predicted_regime not in models:
            continue
        m = st == argmax_idx
        if not m.any():
            continue
        mf = models[predicted_regime]["mfe"].predict(X[m])
        ma = models[predicted_regime]["mae"].predict(X[m])
        mf = np.clip(mf, 0.0, None)
        ma = np.clip(ma, 0.01, None)
        opp[m] = mf / (ma + 0.1)
        mfe_p[m] = mf
        mae_p[m] = ma
    return opp, mfe_p, mae_p


def _compute_opportunity_scores_infer(
    X: np.ndarray,
    regime_probs: np.ndarray,
    models: dict[str, dict[str, lgb.Booster]],
) -> np.ndarray:
    o, _, _ = _compute_opportunity_triplet_infer(X, regime_probs, models)
    return o


def _l2b_triplet_from_trade_prob(p_trade: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pt = np.clip(p_trade.astype(np.float64), 0.0, 1.0)
    mf = np.clip(2.0 * pt, 0.0, 5.0)
    ma = np.clip(0.5 + 0.5 * (1.0 - pt), 0.01, 4.0)
    opp = mf / (ma + 0.1)
    return opp.astype(np.float32), mf.astype(np.float32), ma.astype(np.float32)


def materialize_layer3_features_v2(
    df,
    pipeline: dict,
    l2b_x: np.ndarray,
    regime_prob_cols: list[str],
    chunk: int,
    desc: str = "",
) -> None:
    """Add ``l3_meta['feature_cols']`` prerequisites: L2b triplet + ``l2b_opp_x_<regime>`` (6 cols, opp×p_k)."""
    meta = pipeline["l3_meta"]
    if int(meta.get("l3_schema", 1)) < 2:
        return
    n = len(df)
    if n == 0:
        return
    rp = df[regime_prob_cols].values.astype(np.float32)
    models = pipeline.get("l2b_opp")
    if models:
        o = np.empty(n, dtype=np.float32)
        mf = np.empty(n, dtype=np.float32)
        ma = np.empty(n, dtype=np.float32)
        n_chunk = (n + chunk - 1) // chunk
        for start in _tq(range(0, n, chunk), desc=desc or "L3 L2b triplet", unit="chunk", total=n_chunk):
            end = min(start + chunk, n)
            oc, mfc, mac = _compute_opportunity_triplet_infer(l2b_x[start:end], rp[start:end], models)
            o[start:end] = oc.astype(np.float32)
            mf[start:end] = mfc.astype(np.float32)
            ma[start:end] = mac.astype(np.float32)
    else:
        o, mf, ma = _l2b_triplet_from_trade_prob(df["tq_p_trade"].values.astype(np.float32))

    df["l2b_opportunity_score"] = o
    df["l2b_pred_mfe"] = mf
    df["l2b_pred_mae"] = ma
    rp64 = rp.astype(np.float64)
    o64 = o.astype(np.float64)
    for k, regime in enumerate(REGIMES_6):
        df[f"l2b_opp_x_{regime}"] = o64 * rp64[:, k]

    for c in meta.get("garch_cols", []) + meta.get("pa_key_cols", []) + meta.get("tcn_prob_cols", []):
        if c not in df.columns:
            df[c] = 0.0


def _opp_to_synthetic_p_trade(opp: np.ndarray, thr_row: np.ndarray, kappa: float = 4.0) -> np.ndarray:
    z = np.clip(kappa * (opp - thr_row), -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _apply_cp_skip(regime_probs: np.ndarray, p_trade: np.ndarray, thr_cp: float, tcn_transition_prob: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """Apply Conformal Prediction prediction sets and TCN transition signal to filter out uncertain/OOS trades."""
    y_set = regime_probs >= thr_cp
    set_size = y_set.sum(axis=1)
    contains_bull = y_set[:, 0] | y_set[:, 1]
    contains_bear = y_set[:, 2] | y_set[:, 3]
    is_conflicting = contains_bull & contains_bear
    skip_cp = (set_size >= 3) | is_conflicting | (set_size == 0)
    
    if tcn_transition_prob is not None:
        high_transition_risk = tcn_transition_prob > 0.70
        skip_cp = skip_cp | high_transition_risk
        
    p_trade_adj = p_trade.copy()
    p_trade_adj[skip_cp] = 0.0
    return p_trade_adj, skip_cp


def _chunked_booster_predict(model: lgb.Booster, X: np.ndarray, chunk: int, desc: str) -> np.ndarray:
    n = len(X)
    if n <= chunk:
        return model.predict(X)
    head = model.predict(X[:chunk])
    if head.ndim == 1:
        out = np.empty(n, dtype=np.float64)
    else:
        out = np.empty((n, head.shape[1]), dtype=np.float64)
    out[:chunk] = head
    tail_ranges = list(range(chunk, n, chunk))
    for start in _tq(tail_ranges, desc=desc, unit="chunk", total=len(tail_ranges)):
        end = min(start + chunk, n)
        out[start:end] = model.predict(X[start:end])
    return out


def load_layered_pa_pipeline():
    """Load TCN + L2a regime + L2b trade stack + L3 execution models from ``lgbm_models/``."""
    print("  Loading checkpoints…")
    with open(os.path.join(MODEL_DIR, "tcn_meta.pkl"), "rb") as f:
        tcn_meta = pickle.load(f)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tcn_bd = int(tcn_meta.get("bottleneck_dim", 8))
    n_cls = int(tcn_meta.get("num_regime_classes", tcn_meta.get("num_classes", 6)))
    tcn_model = PAStateTCN(
        input_size=tcn_meta["input_size"],
        num_channels=tcn_meta["num_channels"],
        kernel_size=tcn_meta["kernel_size"],
        dropout=0.0,
        bottleneck_dim=tcn_bd,
        num_classes=n_cls,
    ).to(device)
    tcn_model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "tcn_state_classifier.pt"), map_location=device, weights_only=True)
    )
    tcn_model.eval()

    l2a_model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "state_classifier_6c.txt"))
    with open(os.path.join(MODEL_DIR, "state_calibrators.pkl"), "rb") as f:
        l2a_cals = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "trade_quality_meta.pkl"), "rb") as f:
        tq_meta = pickle.load(f)
    rg = tq_meta["regression_gate"]
    mfiles = rg["model_files"]
    tv = rg["thr_vec"]
    if tuple(rg["groups"]) != REGIMES_6:
        raise ValueError(
            "regression_gate.groups must match REGIMES_6 order/name list in pa_pipeline_infer.py; "
            f"got {rg['groups']!r}",
        )
    if len(tv) != len(REGIMES_6):
        raise ValueError(
            f"regression_gate.thr_vec must have length {len(REGIMES_6)}, got {len(tv)}",
        )
    l2b_opp_thr_vec = np.asarray(tv, dtype=np.float64)
    l2b_opp: dict[str, dict[str, lgb.Booster]] = {}
    for regime in REGIMES_6:
        pm = mfiles.get(f"{regime}_mfe")
        pa = mfiles.get(f"{regime}_mae")
        if not pm or not pa:
            continue
        l2b_opp[regime] = {
            "mfe": lgb.Booster(model_file=os.path.join(MODEL_DIR, pm)),
            "mae": lgb.Booster(model_file=os.path.join(MODEL_DIR, pa)),
        }
    if not l2b_opp:
        raise FileNotFoundError(
            "L2b regression Step1: no opp models loaded; check regression_gate.model_files.",
        )
    l2b_step1 = lgb.Booster(model_file=os.path.join(MODEL_DIR, "trade_gate_step1.txt"))
    l2b_step2 = lgb.Booster(model_file=os.path.join(MODEL_DIR, "trade_dir_step2.txt"))
    l2b_step3 = lgb.Booster(model_file=os.path.join(MODEL_DIR, "trade_grade_step3.txt"))

    with open(os.path.join(MODEL_DIR, "execution_sizer_meta.pkl"), "rb") as f:
        l3_meta = pickle.load(f)
    l3_schema = int(l3_meta.get("l3_schema", 1))
    if l3_schema >= 2:
        mf_paths = l3_meta.get("model_files") or {}
        gate_name = mf_paths.get("gate", "execution_sizer_gate.txt")
        size_name = mf_paths.get("size", "execution_sizer_size.txt")
        
        l3_gate = lgb.Booster(model_file=os.path.join(MODEL_DIR, gate_name))
        l3_size = lgb.Booster(model_file=os.path.join(MODEL_DIR, size_name))
        l3_model = None
    else:
        l3_gate = None
        l3_size = None
        l3_model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "execution_sizer_v1.txt"))

    # Layer 4: Exit Manager (TP / SL Quantile Regressors)
    try:
        with open(os.path.join(MODEL_DIR, "exit_manager_meta.pkl"), "rb") as f:
            l4_meta = pickle.load(f)
        l4_tp = lgb.Booster(model_file=os.path.join(MODEL_DIR, l4_meta["model_files"]["tp"]))
        l4_sl = lgb.Booster(model_file=os.path.join(MODEL_DIR, l4_meta["model_files"]["sl"]))
    except Exception:
        l4_meta = None
        l4_tp = None
        l4_sl = None

    return {
        "tcn": tcn_model,
        "tcn_meta": tcn_meta,
        "device": device,
        "l2a_model": l2a_model,
        "l2a_cals": l2a_cals,
        "tq_meta": tq_meta,
        "l2b_opp": l2b_opp,
        "l2b_opp_thr_vec": l2b_opp_thr_vec,
        "l2b_s1": l2b_step1,
        "l2b_s2": l2b_step2,
        "l2b_s3": l2b_step3,
        "l3_model": l3_model,
        "l3_gate": l3_gate,
        "l3_size": l3_size,
        "l3_meta": l3_meta,
        "l4_tp": l4_tp,
        "l4_sl": l4_sl,
        "l4_meta": l4_meta,
    }
