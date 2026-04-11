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
from core.mamba_pa_state import PAStateMamba
from core.trainers.tcn_constants import STATE_CLASSIFIER_FILE as TCN_STATE_DICT_BASENAME
from core.trainers.layer1b_mamba import MAMBA_STATE_CLASSIFIER_FILE
from core.trainers.l4_sequence import L4ExitSequenceModel
from core.trainers.constants import (
    L2B_META_GATE_ALL_COLS,
    TCN_BARRIER_DIR_DIFF_COL,
    TCN_FEATURES_FOR_L2B,
    TCN_REGIME_FUT_PROB_COLS,
    TCN_TRANSITION_PROB_COL,
)
from core.trainers.layer2b_quality import (
    _build_l2b_gate_meta_features,
    _build_l3_interaction_features,
    _build_l2b_directional_feature_matrix,
    _regime_context_from_probs,
    _tcn_transition_risk_profile,
)

OOS_PRED_CHUNK = max(4096, int(os.environ.get("OOS_PRED_CHUNK", "65536")))

OOS_START = os.environ.get("OOS_START", "2025-01-01")
OOS_END = os.environ.get("OOS_END", "2026-01-01")

MODEL_DIR = os.path.join(_REPO_ROOT, "lgbm_models")
DATA_DIR = os.path.join(_REPO_ROOT, "data")
RESULTS_DIR = os.environ.get("OOS_RESULTS_DIR", os.path.join(_REPO_ROOT, "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)


def _experimental_mamba_enabled() -> bool:
    return os.environ.get("ENABLE_EXPERIMENTAL_MAMBA", "").strip().lower() in {"1", "true", "yes"}


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


def _ensure_tcn_summary_features_infer(df) -> None:
    required = list(TCN_REGIME_FUT_PROB_COLS)
    if not all(c in df.columns for c in required):
        return
    if TCN_TRANSITION_PROB_COL not in df.columns:
        chop = np.asarray(df["tcn_barrier_chop"], dtype=np.float32)
        df[TCN_TRANSITION_PROB_COL] = np.clip(1.0 - chop, 0.0, 1.0)
    if TCN_BARRIER_DIR_DIFF_COL not in df.columns:
        up = np.asarray(df["tcn_barrier_hit_up"], dtype=np.float32)
        dn = np.asarray(df["tcn_barrier_hit_dn"], dtype=np.float32)
        df[TCN_BARRIER_DIR_DIFF_COL] = up - dn


def _compute_opportunity_triplet_infer(
    X: np.ndarray,
    regime_probs: np.ndarray,
    models: dict[str, dict[str, lgb.Booster]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    opp, mfe_p, mae_p, _ = _compute_opportunity_triplet_with_regime_opp_infer(X, regime_probs, models)
    return opp, mfe_p, mae_p


def _compute_opportunity_triplet_with_regime_opp_infer(
    X: np.ndarray,
    regime_probs: np.ndarray,
    models: dict[str, dict[str, lgb.Booster]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    opp = np.zeros(n, dtype=np.float64)
    mfe_p = np.zeros(n, dtype=np.float64)
    mae_p = np.zeros(n, dtype=np.float64)
    opp_regime = np.zeros((n, len(REGIMES_6)), dtype=np.float64)
    if n == 0 or not models:
        return opp, mfe_p, mae_p, opp_regime

    available = [idx for idx, regime in enumerate(REGIMES_6) if regime in models]
    if not available:
        return opp, mfe_p, mae_p, opp_regime

    mfe_stack: list[np.ndarray] = []
    mae_stack: list[np.ndarray] = []
    for idx in available:
        regime = REGIMES_6[idx]
        mf = np.clip(models[regime]["mfe"].predict(X), 0.0, None)
        ma = np.clip(models[regime]["mae"].predict(X), 0.01, None)
        mfe_stack.append(mf)
        mae_stack.append(ma)

    weights = np.clip(regime_probs[:, available].astype(np.float64, copy=False), 0.0, None)
    denom = weights.sum(axis=1, keepdims=True)
    weights = np.divide(
        weights,
        denom,
        out=np.full_like(weights, 1.0 / weights.shape[1]),
        where=denom > 1e-12,
    )
    mfe_mat = np.column_stack(mfe_stack)
    mae_mat = np.column_stack(mae_stack)
    opp_mat = np.log1p(mfe_mat) - np.log1p(mae_mat)
    opp_regime[:, available] = opp_mat
    mfe_p = np.sum(weights * mfe_mat, axis=1)
    mae_p = np.sum(weights * mae_mat, axis=1)
    opp = np.log1p(mfe_p) - np.log1p(mae_p)
    return opp, mfe_p, mae_p, opp_regime


def _l2b_triplet_from_trade_prob(p_trade: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pt = np.clip(p_trade.astype(np.float64), 0.0, 1.0)
    mf = np.clip(2.0 * pt, 0.0, 5.0)
    ma = np.clip(0.5 + 0.5 * (1.0 - pt), 0.01, 4.0)
    opp = np.log1p(mf) - np.log1p(ma)
    return opp.astype(np.float32), mf.astype(np.float32), ma.astype(np.float32)


def materialize_layer3_features_v2(
    df,
    pipeline: dict,
    l2b_x: np.ndarray,
    regime_prob_cols: list[str],
    chunk: int,
    desc: str = "",
) -> None:
    """Add ``l3_meta['feature_cols']`` prerequisites: soft L2b triplet + ``l2b_opp_x_<regime>``."""
    meta = pipeline["l3_meta"]
    if int(meta.get("l3_schema", 1)) < 2:
        return
    n = len(df)
    if n == 0:
        return
    _ensure_tcn_summary_features_infer(df)
    rp = df[regime_prob_cols].values.astype(np.float32)
    raw_route_cols = [f"{c}_raw" for c in regime_prob_cols]
    route_probs = (
        df[raw_route_cols].values.astype(np.float32)
        if all(c in df.columns for c in raw_route_cols)
        else rp
    )
    models = pipeline.get("l2b_opp")
    if models:
        o = np.empty(n, dtype=np.float32)
        mf = np.empty(n, dtype=np.float32)
        ma = np.empty(n, dtype=np.float32)
        opp_regime = np.empty((n, len(REGIMES_6)), dtype=np.float32)
        n_chunk = (n + chunk - 1) // chunk
        for start in _tq(range(0, n, chunk), desc=desc or "L3 L2b triplet", unit="chunk", total=n_chunk):
            end = min(start + chunk, n)
            oc, mfc, mac, orc = _compute_opportunity_triplet_with_regime_opp_infer(
                l2b_x[start:end], route_probs[start:end], models
            )
            o[start:end] = oc.astype(np.float32)
            mf[start:end] = mfc.astype(np.float32)
            ma[start:end] = mac.astype(np.float32)
            opp_regime[start:end] = orc.astype(np.float32)
    else:
        o, mf, ma = _l2b_triplet_from_trade_prob(df["tq_p_trade"].values.astype(np.float32))
        opp_regime = np.repeat(o.reshape(-1, 1), len(REGIMES_6), axis=1).astype(np.float32)

    tcn_prob = df[TCN_TRANSITION_PROB_COL].values.astype(np.float32) if TCN_TRANSITION_PROB_COL in df.columns else None
    l2b_dir_blk, l2b_dir_cols = _build_l2b_directional_feature_matrix(rp, o, mf, ma, opp_regime, tcn_prob)
    for idx, col in enumerate(l2b_dir_cols):
        df[col] = l2b_dir_blk[:, idx]
    if "tq_p_long" in df.columns and "tq_p_short" in df.columns:
        gate_blk, gate_cols = _build_l2b_gate_meta_features(
            df["tq_p_long"].values.astype(np.float32),
            df["tq_p_short"].values.astype(np.float32),
        )
        for idx, col in enumerate(gate_cols):
            df[col] = gate_blk[:, idx]
        interaction_blk, interaction_cols = _build_l3_interaction_features(
            o,
            mf,
            ma,
            df["tq_p_long"].values.astype(np.float32),
            df["tq_p_short"].values.astype(np.float32),
        )
        for idx, col in enumerate(interaction_cols):
            df[col] = interaction_blk[:, idx]

    tcn_feature_cols = meta.get("tcn_feature_cols", meta.get("tcn_prob_cols", TCN_FEATURES_FOR_L2B))
    interaction_cols = meta.get("interaction_feature_cols", list(L3_INTERACTION_FEATURE_COLS))
    for c in meta.get("garch_cols", []) + meta.get("pa_key_cols", []) + tcn_feature_cols + interaction_cols + list(L2B_META_GATE_ALL_COLS):
        if c not in df.columns:
            df[c] = 0.0
def _apply_cp_skip(
    regime_probs: np.ndarray,
    p_trade: np.ndarray,
    thr_cp: float,
    tcn_transition_prob: np.ndarray = None,
    *,
    side: str = "trade",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply direction-aware CP skip plus continuous TCN risk shrink."""
    rp = np.clip(np.asarray(regime_probs, dtype=np.float64), 0.0, None)
    denom = rp.sum(axis=1, keepdims=True)
    rp = np.divide(
        rp,
        denom,
        out=np.full_like(rp, 1.0 / max(rp.shape[1], 1)),
        where=denom > 1e-12,
    )
    y_set = rp >= thr_cp
    set_size = y_set.sum(axis=1)
    contains_bull = y_set[:, 0] | y_set[:, 1]
    contains_bear = y_set[:, 2] | y_set[:, 3]
    is_conflicting = contains_bull & contains_bear
    ctx = _regime_context_from_probs(rp)
    bull_mass = ctx["bull_mass"]
    bear_mass = ctx["bear_mass"]
    range_mass = ctx["range_mass"]
    conf = ctx["conf"]
    margin = ctx["margin"]
    entropy = ctx["entropy"]
    entropy_skip_thr = float(os.environ.get("L2B_CP_ENTROPY_SKIP_THR", "1.45"))
    margin_skip_thr = float(os.environ.get("L2B_CP_MARGIN_SKIP_THR", "0.08"))
    conf_skip_thr = float(os.environ.get("L2B_CP_CONF_SKIP_THR", "0.34"))
    range_skip_thr = float(os.environ.get("L2B_CP_RANGE_SKIP_THR", "0.60"))
    opp_mass_buffer = float(os.environ.get("L2B_CP_OPPOSITE_BUFFER", "0.03"))
    ambiguous = (entropy >= entropy_skip_thr) | (margin <= margin_skip_thr) | (conf <= conf_skip_thr)
    skip_cp = (set_size >= 3) | is_conflicting | (set_size == 0) | (ambiguous & (range_mass >= range_skip_thr))
    if side == "long":
        opposite_hit = contains_bear | (bear_mass >= bull_mass - opp_mass_buffer)
        skip_cp = skip_cp | (opposite_hit & ambiguous)
    elif side == "short":
        opposite_hit = contains_bull | (bull_mass >= bear_mass - opp_mass_buffer)
        skip_cp = skip_cp | (opposite_hit & ambiguous)
    else:
        skip_cp = skip_cp | (ambiguous & (np.abs(bull_mass - bear_mass) <= 0.05))

    p_trade_adj = p_trade.copy()
    prof = _tcn_transition_risk_profile(tcn_transition_prob)
    if prof is not None:
        p_trade_adj = p_trade_adj * prof["prob_scale"]
        skip_cp = skip_cp | prof["is_high"]
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
        readout_type=str(tcn_meta.get("readout_type", "last_timestep")),
        min_attention_seq_len=int(tcn_meta.get("min_attention_seq_len", 4)),
    ).to(device)
    tcn_model.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, TCN_STATE_DICT_BASENAME),
            map_location=device,
            weights_only=True,
        )
    )
    tcn_model.eval()

    # Mamba
    mamba_model = None
    mamba_meta = None
    mamba_meta_path = os.path.join(MODEL_DIR, "mamba_meta.pkl")
    if _experimental_mamba_enabled() and os.path.exists(mamba_meta_path):
        with open(mamba_meta_path, "rb") as f:
            mamba_meta = pickle.load(f)
        n_cls_mamba = int(mamba_meta.get("num_regime_classes", mamba_meta.get("num_classes", 2)))
        mamba_model = PAStateMamba(
            input_size=mamba_meta["input_size"],
            d_model=int(mamba_meta.get("d_model", 64)),
            n_layers=int(mamba_meta.get("n_layers", 4)),
            dropout=0.0,
            bottleneck_dim=int(mamba_meta.get("bottleneck_dim", 8)),
            num_classes=n_cls_mamba,
        ).to(device)
        mamba_model.load_state_dict(
            torch.load(
                os.path.join(MODEL_DIR, MAMBA_STATE_CLASSIFIER_FILE),
                map_location=device,
                weights_only=True,
            )
        )
        mamba_model.eval()
    elif os.path.exists(mamba_meta_path):
        print("  Mamba artifacts found but skipped (experimental module; set ENABLE_EXPERIMENTAL_MAMBA=1 to enable).")

    l2a_model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "state_classifier_6c.txt"))
    with open(os.path.join(MODEL_DIR, "state_calibrators.pkl"), "rb") as f:
        l2a_cals_obj = pickle.load(f)
        if isinstance(l2a_cals_obj, dict):
            l2a_cals = l2a_cals_obj["calibrators"]
        else:
            l2a_cals = l2a_cals_obj

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
    l2b_soft_opp_thr = float(rg.get("soft_opp_threshold", 0.0))
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
    tq_model_files = tq_meta.get("model_files", {})
    l2b_step1_trade = None
    l2b_step2_dir = None
    l2b_step1_long = None
    l2b_step1_short = None
    if "step1_has_trade" in tq_model_files and "step2_direction" in tq_model_files:
        l2b_step1_trade = lgb.Booster(model_file=os.path.join(MODEL_DIR, tq_model_files["step1_has_trade"]))
        l2b_step2_dir = lgb.Booster(model_file=os.path.join(MODEL_DIR, tq_model_files["step2_direction"]))
    else:
        l2b_step1_long = lgb.Booster(model_file=os.path.join(MODEL_DIR, tq_model_files.get("step1_long", "trade_gate_long.txt")))
        l2b_step1_short = lgb.Booster(model_file=os.path.join(MODEL_DIR, tq_model_files.get("step1_short", "trade_gate_short.txt")))
    l2b_step3_path = os.path.join(MODEL_DIR, "trade_grade_step3.txt")
    l2b_step3 = lgb.Booster(model_file=l2b_step3_path) if os.path.exists(l2b_step3_path) else None

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

    # Layer 4: Exit Manager (Ordinal EVT / Multi-class Binning + Survival Proxy)
    try:
        with open(os.path.join(MODEL_DIR, "exit_manager_meta.pkl"), "rb") as f:
            l4_meta = pickle.load(f)
        l4_seq = None
        
        if l4_meta.get("l4_schema", 1) >= 4:
            l4_tp = None
            l4_sl = None
            l4_time = None
            l4_exit = lgb.Booster(model_file=os.path.join(MODEL_DIR, l4_meta["model_files"]["exit"]))
            l4_value = lgb.Booster(model_file=os.path.join(MODEL_DIR, l4_meta["model_files"]["value"]))
            seq_name = (l4_meta.get("model_files") or {}).get("seq")
            if seq_name:
                seq_path = os.path.join(MODEL_DIR, seq_name)
                if os.path.exists(seq_path):
                    l4_seq = L4ExitSequenceModel(
                        input_size=len(l4_meta["feature_cols"]),
                        hidden_size=int(l4_meta.get("seq_hidden_size", 48)),
                        dropout=float(l4_meta.get("seq_dropout", 0.0)),
                    ).to(device)
                    l4_seq.load_state_dict(
                        torch.load(seq_path, map_location=device, weights_only=True)
                    )
                    l4_seq.eval()
        elif l4_meta.get("l4_schema", 1) >= 3:
            l4_tp = [lgb.Booster(model_file=os.path.join(MODEL_DIR, fn)) for fn in l4_meta["model_files"]["tp_ordinal"]]
            l4_sl = [lgb.Booster(model_file=os.path.join(MODEL_DIR, fn)) for fn in l4_meta["model_files"]["sl_ordinal"]]
            l4_exit = None
            l4_value = None
            l4_seq = None
        else:
            l4_tp = lgb.Booster(model_file=os.path.join(MODEL_DIR, l4_meta["model_files"]["tp"]))
            l4_sl = lgb.Booster(model_file=os.path.join(MODEL_DIR, l4_meta["model_files"]["sl"]))
            l4_exit = None
            l4_value = None
            l4_seq = None
            
        if l4_meta.get("l4_schema", 1) < 4:
            l4_time = lgb.Booster(model_file=os.path.join(MODEL_DIR, l4_meta["model_files"]["time"]))
    except Exception:
        l4_meta = None
        l4_tp = None
        l4_sl = None
        l4_time = None
        l4_exit = None
        l4_value = None
        l4_seq = None

    return {
        "tcn": tcn_model,
        "tcn_meta": tcn_meta,
        "mamba": mamba_model,
        "mamba_meta": mamba_meta,
        "device": device,
        "l2a_model": l2a_model,
        "l2a_cals": l2a_cals,
        "tq_meta": tq_meta,
        "l2b_opp": l2b_opp,
        "l2b_opp_thr_vec": l2b_opp_thr_vec,
        "l2b_soft_opp_thr": l2b_soft_opp_thr,
        "l2b_s1_trade": l2b_step1_trade,
        "l2b_s2_dir": l2b_step2_dir,
        "l2b_s1_long": l2b_step1_long,
        "l2b_s1_short": l2b_step1_short,
        "l2b_s3": l2b_step3,
        "l3_model": l3_model,
        "l3_gate": l3_gate,
        "l3_size": l3_size,
        "l3_meta": l3_meta,
        "l4_tp": l4_tp,
        "l4_sl": l4_sl,
        "l4_time": l4_time,
        "l4_exit": l4_exit,
        "l4_value": l4_value,
        "l4_seq": l4_seq,
        "l4_meta": l4_meta,
    }
