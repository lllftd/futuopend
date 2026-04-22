"""
Out-of-sample backtest for the dual-view stack: L1a -> L1b -> L2 -> L3.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.indicators import atr as compute_atr
from core.pa_rules import add_pa_features
from core.trainers.constants import (
    L1A_META_FILE,
    L1A_REGIME_COLS,
    L1B_META_FILE,
    L2_META_FILE,
    L3_META_FILE,
    MODEL_DIR,
    PA_STATE_FEATURES,
    PA_TIMEFRAME,
    TEST_END,
)
from core.trainers.l1a import infer_l1a_market_encoder, load_l1a_market_encoder
from core.trainers.l1b import infer_l1b_market_descriptor, load_l1b_market_descriptor
from core.trainers.l1b.l1a_bridge import (
    attach_l1a_outputs_to_df,
    l1a_feature_cols_from_l1b_meta,
    meta_expects_l1a_features,
)
from core.trainers.stack_v2_common import load_output_cache

# Lazy-loaded stack modules. Some import orders can trigger native crashes on macOS
# when L1a artifacts are loaded after L2/L3 modules are imported.
infer_l2_trade_decision = None
load_l2_trade_decision = None
L3ExitInferenceState = None
L3TrajRollingState = None
l3_entry_policy_params = None
l3_entry_side_from_l2 = None
l3_exit_decision_live = None
l3_exit_policy_params = None
l3_infer_cox_features = None
l3_load_cox_bundle = None
load_l3_exit_manager = None
load_l3_trajectory_encoder_for_infer = None
l3_single_trajectory_embedding = None
build_straddle_features = None
build_base_iv_series = None
StraddleSimulator = None
_live_trade_state_from_bar = None
_net_edge_atr_from_state = None
_apply_l3_exit_calibrator = None
_l3_exit_infer_params = None
_l3_straddle_sim_mode_enabled = None
_l3_soft_exit_hysteresis_thresholds = None
TORCH_DEVICE = None


def _lazy_import_stack_modules() -> None:
    global infer_l2_trade_decision, load_l2_trade_decision
    global L3ExitInferenceState, L3TrajRollingState, l3_entry_policy_params, l3_entry_side_from_l2
    global l3_exit_decision_live, l3_exit_policy_params, l3_infer_cox_features, l3_load_cox_bundle
    global load_l3_exit_manager, load_l3_trajectory_encoder_for_infer, l3_single_trajectory_embedding
    global build_straddle_features, build_base_iv_series, StraddleSimulator
    global _live_trade_state_from_bar, _net_edge_atr_from_state
    global _apply_l3_exit_calibrator, _l3_exit_infer_params, _l3_straddle_sim_mode_enabled, _l3_soft_exit_hysteresis_thresholds
    global TORCH_DEVICE
    if infer_l2_trade_decision is not None:
        return
    from core.trainers.l2 import infer_l2_trade_decision as _infer_l2_trade_decision, load_l2_trade_decision as _load_l2_trade_decision
    from core.trainers.l3 import (
        L3ExitInferenceState as _L3ExitInferenceState,
        L3TrajRollingState as _L3TrajRollingState,
        l3_entry_policy_params as _l3_entry_policy_params,
        l3_entry_side_from_l2 as _l3_entry_side_from_l2,
        l3_exit_decision_live as _l3_exit_decision_live,
        l3_exit_policy_params as _l3_exit_policy_params,
        l3_infer_cox_features as _l3_infer_cox_features,
        l3_load_cox_bundle as _l3_load_cox_bundle,
        load_l3_exit_manager as _load_l3_exit_manager,
        load_l3_trajectory_encoder_for_infer as _load_l3_trajectory_encoder_for_infer,
        l3_single_trajectory_embedding as _l3_single_trajectory_embedding,
    )
    from core.trainers.l3.feature_engineering import build_straddle_features as _build_straddle_features
    from core.trainers.l3.iv_models import build_base_iv_series as _build_base_iv_series
    from core.trainers.l3.straddle_simulator import StraddleSimulator as _StraddleSimulator
    from core.trainers.lgbm_utils import _live_trade_state_from_bar as _live_state, _net_edge_atr_from_state as _net_edge
    from core.trainers.l3.train import (
        _apply_l3_exit_calibrator as _apply_exit_cal,
        _l3_exit_infer_params as _exit_infer_params,
        _l3_straddle_sim_mode_enabled as _straddle_mode_enabled,
        _l3_soft_exit_hysteresis_thresholds as _soft_exit_thresholds,
    )
    from core.trainers.tcn_constants import DEVICE as _TORCH_DEVICE

    infer_l2_trade_decision = _infer_l2_trade_decision
    load_l2_trade_decision = _load_l2_trade_decision
    L3ExitInferenceState = _L3ExitInferenceState
    L3TrajRollingState = _L3TrajRollingState
    l3_entry_policy_params = _l3_entry_policy_params
    l3_entry_side_from_l2 = _l3_entry_side_from_l2
    l3_exit_decision_live = _l3_exit_decision_live
    l3_exit_policy_params = _l3_exit_policy_params
    l3_infer_cox_features = _l3_infer_cox_features
    l3_load_cox_bundle = _l3_load_cox_bundle
    load_l3_exit_manager = _load_l3_exit_manager
    load_l3_trajectory_encoder_for_infer = _load_l3_trajectory_encoder_for_infer
    l3_single_trajectory_embedding = _l3_single_trajectory_embedding
    build_straddle_features = _build_straddle_features
    build_base_iv_series = _build_base_iv_series
    StraddleSimulator = _StraddleSimulator
    _live_trade_state_from_bar = _live_state
    _net_edge_atr_from_state = _net_edge
    _apply_l3_exit_calibrator = _apply_exit_cal
    _l3_exit_infer_params = _exit_infer_params
    _l3_straddle_sim_mode_enabled = _straddle_mode_enabled
    _l3_soft_exit_hysteresis_thresholds = _soft_exit_thresholds
    TORCH_DEVICE = _TORCH_DEVICE

DATA_DIR = _REPO_ROOT / "data"
RESULTS_DIR = Path(os.environ.get("OOS_RESULTS_DIR", str(_REPO_ROOT / "results" / "modeloos")))
# Default OOS = bars never used in any layer fit (same as data_prep "Holdout" / L3 holdout: time >= TEST_END).
# L3 OOT train/val lives in [CAL_END, TEST_END); L2 "holdout" slice is [CAL_END, TEST_END) for metrics only (no fit).
# OOS_END default is far in the future so CSVs through "today" stay included; narrow with env if needed.
OOS_START = os.environ.get("OOS_START", str(TEST_END))
OOS_END = os.environ.get("OOS_END", "2035-01-01")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _resolve_symbol_csv_path(symbol: str) -> Path:
    candidates = [
        DATA_DIR / f"{symbol}.csv",
        DATA_DIR / f"{symbol}_labeled_v2.csv",
        DATA_DIR / f"{symbol.lower()}.csv",
        DATA_DIR / f"{symbol.lower()}_labeled_v2.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing CSV for {symbol}; tried: {', '.join(str(p) for p in candidates)}")


def _oos_progress_enabled() -> bool:
    """Progress bars for OOS (stderr). Off: DISABLE_TQDM=1 or OOS_DISABLE_TQDM=1."""
    if os.environ.get("DISABLE_TQDM", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("OOS_DISABLE_TQDM", "").strip().lower() in {"1", "true", "yes"}:
        return False
    return True


def _oos_tqdm_pbar(*args: Any, **kwargs: Any) -> tqdm:
    kw = {"file": sys.stderr, "mininterval": 0.35, "dynamic_ncols": True}
    kw.update(kwargs)
    return tqdm(*args, **kw)


def _oos_cost_fracs() -> tuple[float, float]:
    """Slippage + commission as fraction of price per leg (see OOS_SLIPPAGE_BPS, OOS_COMMISSION_BPS_PER_SIDE)."""
    slip_bps = float((os.environ.get("OOS_SLIPPAGE_BPS", "0") or "0").strip() or "0")
    comm_bps = float((os.environ.get("OOS_COMMISSION_BPS_PER_SIDE", "0") or "0").strip() or "0")
    return slip_bps * 1e-4, comm_bps * 1e-4


def _oos_l3_exit_mode() -> str:
    """L3: learned exit. atr_trailing: ATR drawdown-from-peak stop (ablation vs L3). OOS_L3_EXIT_MODE."""
    v = (os.environ.get("OOS_L3_EXIT_MODE") or "l3").strip().lower()
    if v in {"l3", "learned", "model"}:
        return "l3"
    if v in {"atr", "atr_trail", "atr_trailing", "trailing", "atrx"}:
        return "atr_trailing"
    raise ValueError("OOS_L3_EXIT_MODE must be 'l3' or 'atr_trailing'")


def _oos_atr_trailing_params() -> tuple[float, int]:
    """(mult, min_hold_bars). Straddle: exit when drawdown_from_peak >= mult * (ATR/close). Directional: mult in ATR units of unreal."""
    mult = float(np.clip(float(os.environ.get("OOS_ATR_TRAIL_MULT", "1.2")), 0.1, 20.0))
    min_h = int(np.clip(int(os.environ.get("OOS_ATR_TRAIL_MIN_HOLD", "1")), 1, 10000))
    return mult, min_h


def _oos_block_entry_l1a_regime_ids() -> set[int]:
    """Comma list of L1a argmax-regime ids to skip at entry (e.g. '1' or '0,1'). OOS_BLOCK_ENTRY_L1A_REGIMES."""
    raw = (os.environ.get("OOS_BLOCK_ENTRY_L1A_REGIMES") or "").strip()
    if not raw:
        return set()
    out: set[int] = set()
    for p in raw.split(","):
        p = p.strip()
        if p == "":
            continue
        out.add(int(p))
    return out


def _oos_figure_title(symbol: str) -> str:
    """Chart title. Override: OOS_CHART_TITLE with format keys {symbol}, {oos_start}, {oos_end}, {test_end}."""
    tpl = (os.environ.get("OOS_CHART_TITLE") or "").strip()
    if tpl:
        try:
            return tpl.format(
                symbol=symbol,
                oos_start=OOS_START,
                oos_end=OOS_END,
                test_end=str(TEST_END),
            )
        except (KeyError, ValueError):
            return tpl
    return f"OOS {symbol}  |  [{OOS_START}, {OOS_END})"


def _oos_adjust_fill(open_px: float, *, is_buy: bool, slip_f: float, comm_f: float) -> float:
    adj = slip_f + comm_f
    if is_buy:
        return float(open_px * (1.0 + adj))
    return float(open_px * (1.0 - adj))


def _oos_default_straddle_dte_days(l3_meta: dict[str, Any]) -> int:
    vals = [int(x) for x in (l3_meta.get("l3_straddle_dte_grid") or []) if int(x) > 0]
    if not vals:
        return int(os.environ.get("L3_STRADDLE_DEFAULT_DTE_DAYS", "14"))
    raw = (os.environ.get("L3_STRADDLE_DEFAULT_DTE_DAYS") or "").strip()
    if raw:
        return max(1, int(raw))
    vals = sorted(vals)
    return int(vals[len(vals) // 2])


def _prepare_symbol_df(symbol: str) -> pd.DataFrame:
    from core.trainers.data_prep import ensure_breakout_features, ensure_structure_context_features

    show = _oos_progress_enabled()
    prep = _oos_tqdm_pbar(
        total=5,
        desc=f"[{symbol}] prep",
        unit="step",
        leave=True,
        mininterval=0.25,
    ) if show else None
    try:
        print(f"  [{symbol}] loading CSV...", flush=True)
        raw_path = _resolve_symbol_csv_path(symbol)
        raw = pd.read_csv(raw_path)
        raw["time_key"] = pd.to_datetime(raw["time_key"])
        raw = raw[
            (raw["time_key"] >= pd.Timestamp(OOS_START) - pd.Timedelta(days=10))
            & (raw["time_key"] < pd.Timestamp(OOS_END))
        ].reset_index(drop=True)
        if prep is not None:
            prep.set_postfix_str(f"rows={len(raw):,}")
            prep.update(1)
        print(f"  [{symbol}] rows(after time filter)={len(raw):,}  ATR + PA...", flush=True)
        atr_1m = compute_atr(raw, length=14)
        if prep is not None:
            prep.set_postfix_str("PA rules (slow)")
        df = add_pa_features(raw, atr_1m, timeframe=PA_TIMEFRAME)
        if prep is not None:
            prep.update(1)
        print(f"  [{symbol}] breakout + structure context...", flush=True)
        if prep is not None:
            prep.set_postfix_str("breakout")
        df = ensure_breakout_features(df)
        if prep is not None:
            prep.update(1)
            prep.set_postfix_str("structure")
        df = ensure_structure_context_features(df)
        if prep is not None:
            prep.update(1)
        if "lbl_atr" not in df.columns:
            if "atr_5m" in df.columns:
                df["lbl_atr"] = pd.to_numeric(df["atr_5m"], errors="coerce").fillna(method="ffill").fillna(0.25)
            elif "atr_1m" in df.columns:
                df["lbl_atr"] = pd.to_numeric(df["atr_1m"], errors="coerce").fillna(method="ffill").fillna(0.25)
            else:
                df["lbl_atr"] = (df["high"] - df["low"]).ewm(span=14, min_periods=1).mean().clip(lower=1e-3)
        df["symbol"] = symbol
        out = df[df["time_key"] >= pd.Timestamp(OOS_START)].reset_index(drop=True)
        if prep is not None:
            prep.set_postfix_str(f"OOS n={len(out):,}")
            prep.update(1)
        print(f"  [{symbol}] feature prep done OOS rows={len(out):,}", flush=True)
        return out
    finally:
        if prep is not None:
            prep.close()


def _build_l3_feature_vector(
    idx: int,
    in_pos: int,
    hold: int,
    entry_idx: int,
    entry_price: float,
    entry_atr: float,
    df: pd.DataFrame,
    l1a_out: pd.DataFrame,
    l2_out: pd.DataFrame,
    peak_unreal: float,
    l3_aux: dict,
    l3_meta: dict,
    cox_bundle: dict | None,
) -> tuple[np.ndarray, float, float]:
    if _l3_straddle_sim_mode_enabled(l3_meta):
        sim = l3_aux.get("_straddle_sim")
        if not isinstance(sim, StraddleSimulator):
            sim = StraddleSimulator(risk_free_rate=float(l3_aux.get("risk_free_rate", 0.04)))
            l3_aux["_straddle_sim"] = sim
        strike = float(l3_aux.get("strike", df["close"].iloc[entry_idx]))
        entry_iv = float(l3_aux.get("entry_iv", df.loc[entry_idx, "l3_base_iv"] if "l3_base_iv" in df.columns else 0.25))
        entry_underlying = float(l3_aux.get("entry_underlying", df["close"].iloc[entry_idx]))
        dte_days = int(l3_aux.get("dte_days", _oos_default_straddle_dte_days(l3_meta)))
        current_iv_base = float(df.loc[idx, "l3_base_iv"]) if "l3_base_iv" in df.columns else entry_iv
        current_iv = float(np.clip(0.65 * entry_iv + 0.35 * current_iv_base, 0.05, 5.0))
        t_remaining = max(dte_days / 252.0 - hold / (390.0 * 252.0), 1e-9)
        greeks = sim.straddle_greeks(float(df["close"].iloc[idx]), strike, t_remaining, current_iv)
        pnl_pct = float((greeks.price - entry_price) / max(entry_price, 1e-9))
        peak_in = float(peak_unreal)
        peak_unreal = max(peak_in, pnl_pct)
        drawdown_from_peak = float(peak_unreal - pnl_pct)
        running_mfe = float(max(float(l3_aux.get("running_mfe", 0.0)), max(pnl_pct, 0.0)))
        running_mae = float(max(float(l3_aux.get("running_mae", 0.0)), max(-pnl_pct, 0.0)))
        l3_aux["running_mfe"] = running_mfe
        l3_aux["running_mae"] = running_mae
        live_edge = float(running_mfe - running_mae)
        entry_regime = l2_out.loc[entry_idx, [f"l2_entry_regime_{i}" for i in range(6)]].to_numpy(dtype=np.float32)
        current_regime = l1a_out.loc[idx, L1A_REGIME_COLS].to_numpy(dtype=np.float32)
        p = np.clip(entry_regime, 1e-6, 1.0)
        q = np.clip(current_regime, 1e-6, 1.0)
        p = p / p.sum()
        q = q / q.sum()
        regime_div = float(np.sum(p * (np.log(p) - np.log(q))))
        entry_vol = float(l2_out.loc[entry_idx, "l2_entry_vol"])
        current_vol = float(l1a_out.loc[idx, "l1a_vol_forecast"])
        vol_surprise = float(current_vol / max(entry_vol, 1e-3))
        log_h = float(np.log1p(hold))
        h_sq = float(hold * hold) / 100.0
        h_bkt = float(np.searchsorted(np.array([3, 8, 15, 30, 999], dtype=np.int64), int(hold), side="right"))
        j_vel = max(entry_idx + 1, idx - 3)
        vel3 = float((df["close"].iloc[idx] - df["close"].iloc[j_vel]) / max(entry_atr, 1e-6))
        j_rd = max(entry_idx + 1, idx - 3)
        q_past = np.clip(l1a_out.loc[j_rd, L1A_REGIME_COLS].to_numpy(dtype=np.float32), 1e-6, 1.0)
        q_past = q_past / q_past.sum()
        reg_div_past = float(np.sum(p * (np.log(p) - np.log(q_past))))
        mom_rd = float(regime_div - reg_div_past)
        if idx >= entry_idx + 2:
            vs0 = float(l1a_out.loc[idx, "l1a_vol_forecast"] / max(entry_vol, 1e-3))
            vs1 = float(l1a_out.loc[idx - 1, "l1a_vol_forecast"] / max(entry_vol, 1e-3))
            vs2 = float(l1a_out.loc[idx - 2, "l1a_vol_forecast"] / max(entry_vol, 1e-3))
            vs_acc = float(vs0 - 2.0 * vs1 + vs2)
        else:
            vs_acc = 0.0
        rids: list[int] = []
        for j in range(max(entry_idx + 1, idx - 2), idx + 1):
            rids.append(int(np.argmax(l1a_out.loc[j, L1A_REGIME_COLS].to_numpy(dtype=np.float32))))
        stab = float(np.mean(np.array(rids) == rids[-1])) if rids else 1.0
        pa_state = (
            df.loc[idx, PA_STATE_FEATURES].to_numpy(dtype=np.float32)
            if all(col in df.columns for col in PA_STATE_FEATURES)
            else np.zeros(len(PA_STATE_FEATURES), dtype=np.float32)
        )
        dec_conf_e = float(l2_out.loc[entry_idx, "l2_decision_confidence"])
        signal_conf_decay = float(l2_out.loc[idx, "l2_decision_confidence"]) - dec_conf_e
        so_e = float(l2_out.loc[entry_idx].get("l2_straddle_on", 0))
        so_c = float(l2_out.loc[idx].get("l2_straddle_on", 0))
        signal_direction_agree = float((so_e > 0.5) and (so_c > 0.5))
        rid_e = int(np.argmax(entry_regime.astype(np.float64)))
        regime_changed = float(int(np.argmax(current_regime.astype(np.float64)) != rid_e))
        gate_e = float(np.clip(float(l2_out.loc[entry_idx, "l2_gate_prob"]), 0.0, 1.0)) if "l2_gate_prob" in l2_out.columns else 0.0
        gate_curr = float(np.clip(float(l2_out.loc[idx, "l2_gate_prob"]), 0.0, 1.0)) if "l2_gate_prob" in l2_out.columns else 0.0
        gate_decay = float(gate_curr - gate_e)
        regime_probs_i = l2_out.loc[idx, [f"l2_entry_regime_{k}" for k in range(len(L1A_REGIME_COLS))]].to_numpy(dtype=np.float32)
        entry_vol_i = float(l2_out.loc[idx, "l2_entry_vol"])
        pa_row = df.loc[idx, PA_STATE_FEATURES] if all(c in df.columns for c in PA_STATE_FEATURES) else None
        min_c, min_sz, _, _ = l3_entry_policy_params(regime_probs_i, entry_vol_i, l3_meta, pa_state=pa_row)
        cls_i = 0 if float(l2_out.loc[idx].get("l2_straddle_on", 0)) > 0.5 else 1
        conf_i = float(l2_out.loc[idx, "l2_decision_confidence"])
        sz_i = float(l2_out.loc[idx, "l2_size"])
        would_enter_now = float(l3_entry_side_from_l2(cls_i, conf_i, sz_i, min_confidence=min_c, min_size=min_sz) != 0.0)
        if pnl_pct >= peak_in - 1e-9:
            l3_aux["bars_since_peak"] = 0
        else:
            l3_aux["bars_since_peak"] = int(l3_aux.get("bars_since_peak", 0)) + 1
        bars_since_peak = float(l3_aux["bars_since_peak"])
        at_new_high = float(abs(pnl_pct - peak_unreal) < 1e-9)
        regret_ratio = float(max(0.0, (peak_unreal - pnl_pct) / max(abs(peak_unreal), 1e-6))) if abs(peak_unreal) > 1e-6 else 0.0
        regret_velocity = float(drawdown_from_peak / bars_since_peak) if bars_since_peak > 0.5 else 0.0
        w_fav = float(os.environ.get("L3_BAYES_LLR_FAV", "0.28"))
        w_adv = float(os.environ.get("L3_BAYES_LLR_ADV", "-0.35"))
        w_reg = float(os.environ.get("L3_BAYES_LLR_REGIME", "-0.45"))
        w_gate = float(os.environ.get("L3_BAYES_LLR_GATE", "-0.18"))
        w_gthr = float(os.environ.get("L3_BAYES_GATE_DECAY_THR", "-0.12"))
        prev_u = float(l3_aux.get("prev_unreal", 0.0))
        du = pnl_pct - prev_u
        l3_aux["prev_unreal"] = pnl_pct
        favorable = du > 0.0
        llr = w_fav if favorable else w_adv
        if regime_changed > 0.5:
            llr += w_reg
        if gate_decay < w_gthr:
            llr += w_gate
        lo = float(l3_aux["log_odds"]) + llr
        l3_aux["log_odds"] = lo
        trade_quality_bayes = float(1.0 / (1.0 + np.exp(-lo)))
        vals: dict[str, float] = {
            "l2_straddle_on": float(l2_out.loc[entry_idx].get("l2_straddle_on", 0.0)),
            "l2_range_pred": float(l2_out.loc[entry_idx, "l2_range_pred"]),
            "l2_gate_prob": float(l2_out.loc[entry_idx, "l2_gate_prob"]),
            "l2_decision_confidence": float(l2_out.loc[entry_idx, "l2_decision_confidence"]),
            "l2_size": float(l2_out.loc[entry_idx, "l2_size"]),
            "l2_pred_mfe": float(l2_out.loc[entry_idx, "l2_pred_mfe"]),
            "l2_pred_mae": float(l2_out.loc[entry_idx, "l2_pred_mae"]),
            "l2_predicted_profit": float(l2_out.loc[entry_idx, "l2_predicted_profit"]) if "l2_predicted_profit" in l2_out.columns else 0.0,
            "l3_l2_vol_regime_id": float(
                {"low_vol_stable": 0, "low_vol_rising": 1, "mid_vol": 2, "high_vol_stable": 3, "high_vol_falling": 4}.get(
                    str(l2_out.loc[entry_idx, "l2_vol_regime"]), -1
                )
            ) if "l2_vol_regime" in l2_out.columns else -1.0,
            "l2_regime_size_mult": float(l2_out.loc[entry_idx, "l2_regime_size_mult"]) if "l2_regime_size_mult" in l2_out.columns else 1.0,
            **{f"l2_entry_regime_{i}": float(entry_regime[i]) for i in range(len(entry_regime))},
            "l2_entry_vol": float(entry_vol),
            **{c: float(current_regime[j]) for j, c in enumerate(L1A_REGIME_COLS)},
            "l1a_vol_forecast": float(current_vol),
            "l3_regime_divergence": float(regime_div),
            "l3_vol_surprise": float(vol_surprise),
            "l3_hold_bars": float(hold),
            "l3_unreal_pnl_atr": float(pnl_pct),
            "l3_live_mfe": float(running_mfe),
            "l3_live_mae": float(running_mae),
            "l3_live_edge": float(live_edge),
            "l3_side": 1.0,
            "l3_log_hold_bars": float(log_h),
            "l3_hold_bars_sq": float(h_sq),
            "l3_hold_bucket": float(h_bkt),
            "l3_drawdown_from_peak_atr": float(drawdown_from_peak),
            "l3_price_velocity_3bar_atr": float(vel3),
            "l3_feature_momentum_regdiv_3bar": float(mom_rd),
            "l3_vol_surprise_accel": float(vs_acc),
            "l3_regime_stability_3bar": float(stab),
            **{PA_STATE_FEATURES[k]: float(pa_state[k]) for k in range(len(PA_STATE_FEATURES))},
            "l3_signal_conf_decay": float(signal_conf_decay),
            "l3_signal_direction_agree": float(signal_direction_agree),
            "l3_regime_changed": float(regime_changed),
            "l3_l2_gate_current": float(gate_curr),
            "l3_l2_gate_decay": float(gate_decay),
            "l3_would_enter_now": float(would_enter_now),
            "l3_regret_ratio": float(regret_ratio),
            "l3_bars_since_peak": float(bars_since_peak),
            "l3_at_new_high": float(at_new_high),
            "l3_regret_velocity": float(regret_velocity),
            "l3_trade_quality_bayes": float(trade_quality_bayes),
            "l3_straddle_value_rel": float(greeks.price / max(entry_price, 1e-9)),
            "l3_straddle_pnl_pct": float(pnl_pct),
            "l3_straddle_theta": float(greeks.theta),
            "l3_straddle_vega": float(greeks.vega),
            "l3_straddle_gamma": float(greeks.gamma),
            "l3_straddle_iv": float(current_iv),
            "l3_straddle_entry_iv": float(entry_iv),
            "l3_straddle_t_remaining": float(t_remaining),
            "l3_underlying_abs_move": float(abs(float(df["close"].iloc[idx]) / max(entry_underlying, 1e-9) - 1.0)),
            "l3_underlying_gap_abs": float(abs(float(df["open"].iloc[idx]) / max(float(df["close"].iloc[idx - 1]), 1e-9) - 1.0)),
        }
        Trem = float(t_remaining)
        dte_d = float(l3_aux.get("dte_days", _oos_default_straddle_dte_days(l3_meta)))
        T_init = max(dte_d / 252.0, 1e-9)
        if "straddle_T0" not in l3_aux:
            l3_aux["straddle_T0"] = T_init
        T0 = float(l3_aux["straddle_T0"])
        vals["l3_theta_burn_rate"] = float(abs(greeks.theta) / max(Trem, 1e-8))
        rv60 = float(df.loc[idx, "rv_60"]) if "rv_60" in df.columns else 0.25
        vals["l3_iv_rv_spread"] = float(current_iv - rv60)
        vals["l3_remaining_dte_ratio"] = float(Trem / max(T0, 1e-9))
        vcol = "vixy_level_ma60_ratio"
        if vcol in df.columns:
            vc = float(pd.to_numeric(df.loc[idx, vcol], errors="coerce") or 1.0)
            ve = float(l3_aux.get("vixy_entry", vc))
            if "vixy_entry" not in l3_aux:
                l3_aux["vixy_entry"] = vc
            l3_aux["vixy_max"] = max(float(l3_aux.get("vixy_max", vc)), vc)
            vals["l3_vixy_max_since_entry"] = float(l3_aux["vixy_max"] / max(ve, 1e-6))
            vals["l3_vixy_rel_entry"] = float(vc / max(ve, 1e-6))
        else:
            vals["l3_vixy_max_since_entry"] = 1.0
            vals["l3_vixy_rel_entry"] = 1.0
        hist = l3_aux.get("pnl_hist")
        if not isinstance(hist, list):
            hist = []
        hist.append(float(pnl_pct))
        if len(hist) > 64:
            hist = hist[-64:]
        l3_aux["pnl_hist"] = hist
        vals["l3_roll_pnl_vol_5"] = float(np.std(hist[-5:])) if len(hist) >= 2 else 0.0
        vals["l3_pnl_path_curvature"] = (
            float(abs(hist[-1] - 2.0 * hist[-2] + hist[-3])) if len(hist) >= 3 else 0.0
        )
        feature_cols = list(l3_meta["feature_cols"])
        for c in feature_cols:
            if c not in vals and c in df.columns:
                vals[c] = float(pd.to_numeric(pd.Series([df.loc[idx, c]]), errors="coerce").fillna(0.0).iloc[0])
        static_names = [c for c in feature_cols if not c.startswith("l3_traj_emb_")]
        cox_names = {"l3_cox_log_partial_hazard", "l3_cox_baseline_cumhaz_at_stop"}
        static_wo_cox = [c for c in static_names if c not in cox_names]
        feat_base = np.asarray([float(vals.get(c, 0.0)) for c in static_wo_cox], dtype=np.float32)
        cox_part = l3_infer_cox_features(cox_bundle, feat_base, static_wo_cox)
        feat = np.concatenate([feat_base, cox_part], dtype=np.float32)
        return feat, peak_unreal, drawdown_from_peak

    live_mfe, live_mae, unreal = _live_trade_state_from_bar(
        side=float(in_pos),
        entry_price=entry_price,
        atr=entry_atr,
        high_price=float(df["high"].iloc[idx]),
        low_price=float(df["low"].iloc[idx]),
        close_price=float(df["close"].iloc[idx]),
    )
    live_edge = float(_net_edge_atr_from_state(live_mfe, live_mae, hold))
    entry_regime = l2_out.loc[entry_idx, [f"l2_entry_regime_{i}" for i in range(6)]].to_numpy(dtype=np.float32)
    current_regime = l1a_out.loc[idx, L1A_REGIME_COLS].to_numpy(dtype=np.float32)
    p = np.clip(entry_regime, 1e-6, 1.0)
    q = np.clip(current_regime, 1e-6, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    regime_div = float(np.sum(p * (np.log(p) - np.log(q))))
    entry_vol = float(l2_out.loc[entry_idx, "l2_entry_vol"])
    current_vol = float(l1a_out.loc[idx, "l1a_vol_forecast"])
    vol_surprise = float(current_vol / max(entry_vol, 1e-3))
    log_h = float(np.log1p(hold))
    h_sq = float(hold * hold) / 100.0
    h_bkt = float(np.searchsorted(np.array([3, 8, 15, 30, 999], dtype=np.int64), int(hold), side="right"))
    u = float(unreal)
    peak_in = float(peak_unreal)
    peak_unreal = max(peak_in, u)
    drawdown_from_peak = float(peak_unreal - u)
    j_vel = max(entry_idx + 1, idx - 3)
    vel3 = float((df["close"].iloc[idx] - df["close"].iloc[j_vel]) / max(entry_atr, 1e-6))
    j_rd = max(entry_idx + 1, idx - 3)
    q_past = np.clip(l1a_out.loc[j_rd, L1A_REGIME_COLS].to_numpy(dtype=np.float32), 1e-6, 1.0)
    q_past = q_past / q_past.sum()
    reg_div_past = float(np.sum(p * (np.log(p) - np.log(q_past))))
    mom_rd = float(regime_div - reg_div_past)
    if idx >= entry_idx + 2:
        vs0 = float(l1a_out.loc[idx, "l1a_vol_forecast"] / max(entry_vol, 1e-3))
        vs1 = float(l1a_out.loc[idx - 1, "l1a_vol_forecast"] / max(entry_vol, 1e-3))
        vs2 = float(l1a_out.loc[idx - 2, "l1a_vol_forecast"] / max(entry_vol, 1e-3))
        vs_acc = float(vs0 - 2.0 * vs1 + vs2)
    else:
        vs_acc = 0.0
    rids: list[int] = []
    for j in range(max(entry_idx + 1, idx - 2), idx + 1):
        rids.append(int(np.argmax(l1a_out.loc[j, L1A_REGIME_COLS].to_numpy(dtype=np.float32))))
    stab = float(np.mean(np.array(rids) == rids[-1])) if rids else 1.0
    pa_state = (
        df.loc[idx, PA_STATE_FEATURES].to_numpy(dtype=np.float32)
        if all(col in df.columns for col in PA_STATE_FEATURES)
        else np.zeros(len(PA_STATE_FEATURES), dtype=np.float32)
    )
    dec_conf_e = float(l2_out.loc[entry_idx, "l2_decision_confidence"])
    signal_conf_decay = float(l2_out.loc[idx, "l2_decision_confidence"]) - dec_conf_e
    so_e = float(l2_out.loc[entry_idx].get("l2_straddle_on", 0))
    so_c = float(l2_out.loc[idx].get("l2_straddle_on", 0))
    cls_e = 0 if so_e > 0.5 else 1
    cls_c = 0 if so_c > 0.5 else 1
    dir_e = 1.0 if cls_e == 0 else (-1.0 if cls_e == 2 else 0.0)
    dir_c = 1.0 if cls_c == 0 else (-1.0 if cls_c == 2 else 0.0)
    signal_direction_agree = float(dir_c == dir_e and dir_e != 0.0)
    rid_e = int(np.argmax(entry_regime.astype(np.float64)))
    regime_changed = float(int(np.argmax(current_regime.astype(np.float64)) != rid_e))
    if "l2_gate_prob" in l2_out.columns:
        neut_e = float(1.0 - np.clip(float(l2_out.loc[entry_idx, "l2_gate_prob"]), 0.0, 1.0))
        neut_c = float(1.0 - np.clip(float(l2_out.loc[idx, "l2_gate_prob"]), 0.0, 1.0))
    elif "l2_decision_neutral" in l2_out.columns:
        neut_e = float(l2_out.loc[entry_idx, "l2_decision_neutral"])
        neut_c = float(l2_out.loc[idx, "l2_decision_neutral"])
    else:
        neut_e = 1.0 if cls_e == 1 else 0.25
        neut_c = 1.0 if cls_c == 1 else 0.25
    gate_e = float(1.0 - neut_e)
    gate_curr = float(1.0 - neut_c)
    gate_decay = float(gate_curr - gate_e)
    regime_probs_i = l2_out.loc[idx, [f"l2_entry_regime_{k}" for k in range(len(L1A_REGIME_COLS))]].to_numpy(
        dtype=np.float32
    )
    entry_vol_i = float(l2_out.loc[idx, "l2_entry_vol"])
    pa_row = df.loc[idx, PA_STATE_FEATURES] if all(c in df.columns for c in PA_STATE_FEATURES) else None
    min_c, min_sz, _, _ = l3_entry_policy_params(regime_probs_i, entry_vol_i, l3_meta, pa_state=pa_row)
    cls_i = 0 if float(l2_out.loc[idx].get("l2_straddle_on", 0)) > 0.5 else 1
    conf_i = float(l2_out.loc[idx, "l2_decision_confidence"])
    sz_i = float(l2_out.loc[idx, "l2_size"])
    would_enter_now = float(l3_entry_side_from_l2(cls_i, conf_i, sz_i, min_confidence=min_c, min_size=min_sz) != 0.0)
    if u >= peak_in - 1e-9:
        l3_aux["bars_since_peak"] = 0
    else:
        l3_aux["bars_since_peak"] = int(l3_aux.get("bars_since_peak", 0)) + 1
    bars_since_peak = float(l3_aux["bars_since_peak"])
    at_new_high = float(abs(u - peak_unreal) < 1e-9)
    if peak_unreal > 1e-6:
        regret_ratio = float(max(0.0, (peak_unreal - u) / peak_unreal))
    else:
        regret_ratio = 0.0
    regret_velocity = float(drawdown_from_peak / bars_since_peak) if bars_since_peak > 0.5 else 0.0

    w_fav = float(os.environ.get("L3_BAYES_LLR_FAV", "0.28"))
    w_adv = float(os.environ.get("L3_BAYES_LLR_ADV", "-0.35"))
    w_reg = float(os.environ.get("L3_BAYES_LLR_REGIME", "-0.45"))
    w_gate = float(os.environ.get("L3_BAYES_LLR_GATE", "-0.18"))
    w_gthr = float(os.environ.get("L3_BAYES_GATE_DECAY_THR", "-0.12"))
    prev_u = float(l3_aux.get("prev_unreal", 0.0))
    du = u - prev_u
    l3_aux["prev_unreal"] = u
    sgn = 1.0 if float(in_pos) > 0 else -1.0
    favorable = sgn * du > 0.0
    llr = w_fav if favorable else w_adv
    if regime_changed > 0.5:
        llr += w_reg
    if gate_decay < w_gthr:
        llr += w_gate
    lo = float(l3_aux["log_odds"]) + llr
    l3_aux["log_odds"] = lo
    trade_quality_bayes = float(1.0 / (1.0 + np.exp(-lo)))

    vals: dict[str, float] = {
        "l2_decision_confidence": float(l2_out.loc[entry_idx, "l2_decision_confidence"]),
        "l2_size": float(l2_out.loc[entry_idx, "l2_size"]),
        "l2_pred_mfe": float(l2_out.loc[entry_idx, "l2_pred_mfe"]),
        "l2_pred_mae": float(l2_out.loc[entry_idx, "l2_pred_mae"]),
        **{f"l2_entry_regime_{i}": float(entry_regime[i]) for i in range(len(entry_regime))},
        "l2_entry_vol": float(entry_vol),
        **{c: float(current_regime[j]) for j, c in enumerate(L1A_REGIME_COLS)},
        "l1a_vol_forecast": float(current_vol),
        "l3_regime_divergence": float(regime_div),
        "l3_vol_surprise": float(vol_surprise),
        "l3_hold_bars": float(hold),
        "l3_unreal_pnl_atr": float(unreal),
        "l3_live_mfe": float(live_mfe),
        "l3_live_mae": float(live_mae),
        "l3_live_edge": float(live_edge),
        "l3_side": float(in_pos),
        "l3_log_hold_bars": float(log_h),
        "l3_hold_bars_sq": float(h_sq),
        "l3_hold_bucket": float(h_bkt),
        "l3_drawdown_from_peak_atr": float(drawdown_from_peak),
        "l3_price_velocity_3bar_atr": float(vel3),
        "l3_feature_momentum_regdiv_3bar": float(mom_rd),
        "l3_vol_surprise_accel": float(vs_acc),
        "l3_regime_stability_3bar": float(stab),
        **{PA_STATE_FEATURES[k]: float(pa_state[k]) for k in range(len(PA_STATE_FEATURES))},
        "l3_signal_conf_decay": float(signal_conf_decay),
        "l3_signal_direction_agree": float(signal_direction_agree),
        "l3_regime_changed": float(regime_changed),
        "l3_l2_gate_current": float(gate_curr),
        "l3_l2_gate_decay": float(gate_decay),
        "l3_would_enter_now": float(would_enter_now),
        "l3_regret_ratio": float(regret_ratio),
        "l3_bars_since_peak": float(bars_since_peak),
        "l3_at_new_high": float(at_new_high),
        "l3_regret_velocity": float(regret_velocity),
        "l3_trade_quality_bayes": float(trade_quality_bayes),
    }
    feature_cols = list(l3_meta["feature_cols"])
    static_names = [c for c in feature_cols if not c.startswith("l3_traj_emb_")]
    cox_names = {"l3_cox_log_partial_hazard", "l3_cox_baseline_cumhaz_at_stop"}
    static_wo_cox = [c for c in static_names if c not in cox_names]
    feat_base = np.asarray([vals[c] for c in static_wo_cox], dtype=np.float32)
    cox_part = l3_infer_cox_features(cox_bundle, feat_base, static_wo_cox)
    feat = np.concatenate([feat_base, cox_part], dtype=np.float32)
    return feat, peak_unreal, drawdown_from_peak


def _ensure_backtest_artifacts_exist() -> None:
    required = [
        L1A_META_FILE,
        L1B_META_FILE,
        L2_META_FILE,
        L3_META_FILE,
    ]
    missing = [name for name in required if not (Path(MODEL_DIR) / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing trained stack artifacts under {Path(MODEL_DIR).resolve()}: {missing_list}. "
            "Train the new stack first, e.g. `./scripts/run_train.sh layer1`."
        )


def run_single_symbol(
    symbol: str,
    l1a_model,
    l1a_meta: dict,
    l1b_models: dict,
    l1b_meta: dict,
    l2_models: dict,
    l2_meta: dict,
    l3_models: dict,
    l3_meta: dict,
    l3_traj_enc=None,
    l3_traj_cfg=None,
    torch_device=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Returns (trades, oos_price_df, runtime_diag) for charting + diagnostics."""
    print(f"\n[{symbol}] preparing features...", flush=True)
    df = _prepare_symbol_df(symbol)
    if df.empty:
        return pd.DataFrame(), df, {}
    if _l3_straddle_sim_mode_enabled(l3_meta):
        df = build_straddle_features(df, timestamp_col="time_key")
        df["l3_base_iv"] = build_base_iv_series(df, timestamp_col="time_key", close_col="close")
    show_pb = _oos_progress_enabled()
    infer_bar = (
        _oos_tqdm_pbar(total=3, desc=f"[{symbol}] infer", unit="stage", leave=True, mininterval=0.25)
        if show_pb
        else None
    )
    try:
        if infer_bar is not None:
            infer_bar.set_postfix_str("L1a")
        l1a_out = infer_l1a_market_encoder(l1a_model, l1a_meta, df.copy())
        if infer_bar is not None:
            infer_bar.update(1)
            infer_bar.set_postfix_str("L1b")
        df_l1b = df.copy()
        if meta_expects_l1a_features(l1b_meta):
            l1_cols = l1a_feature_cols_from_l1b_meta(l1b_meta)
            df_l1b = attach_l1a_outputs_to_df(df_l1b, l1a_out, cols=l1_cols if l1_cols else None)
        l1b_out = infer_l1b_market_descriptor(
            l1b_models, l1b_meta, df_l1b, infer_stage_pbar=infer_bar
        )
        if infer_bar is not None:
            infer_bar.update(1)
            infer_bar.set_postfix_str("L2")
        l2_out = infer_l2_trade_decision(l2_models, l2_meta, df.copy(), l1a_out, l1b_out)
        if infer_bar is not None:
            infer_bar.update(1)
    finally:
        if infer_bar is not None:
            infer_bar.close()

    feature_cols = list(l3_meta["feature_cols"])
    static_l3_names = [c for c in feature_cols if not c.startswith("l3_traj_emb_")]
    _sidx = {c: i for i, c in enumerate(static_l3_names)}
    exit_model = l3_models["exit"]
    value_model = l3_models.get("value")
    value_nonzero_model = l3_models.get("value_nonzero")
    value_hurdle_power = float(l3_meta.get("l3_value_hurdle_prob_power", 1.0))
    if l3_meta.get("l3_value_mode") == "disabled" or l3_meta.get("l3_value_disabled"):
        value_model = None
        value_nonzero_model = None
    exit_calibrator = l3_models.get("exit_calibrator")
    dev = torch_device if torch_device is not None else TORCH_DEVICE
    hybrid = l3_traj_enc is not None and l3_traj_cfg is not None
    traj_buf: L3TrajRollingState | None = None
    max_hold = int(
        l3_meta.get("l3_straddle_max_hold_minutes", l3_meta.get("l3_target_horizon_bars", 30))
        if _l3_straddle_sim_mode_enabled(l3_meta)
        else l3_meta.get("l3_target_horizon_bars", 30)
    )
    l2_abstain_margin = float((l2_meta.get("two_stage_policy") or l2_meta).get("direction_abstain_margin", 0.0))
    slip_f, comm_f = _oos_cost_fracs()
    l3_exit_mode = _oos_l3_exit_mode()
    atr_trail_mult, atr_trail_min_hold = _oos_atr_trailing_params()
    block_entry_l1a_regimes = _oos_block_entry_l1a_regime_ids()
    print(
        f"  [{symbol}] exit={l3_exit_mode}  ATR_trail_mult={atr_trail_mult}  ATR_min_hold={atr_trail_min_hold}  "
        f"block_L1a_regimes={sorted(block_entry_l1a_regimes) if block_entry_l1a_regimes else '—'}",
        flush=True,
    )
    trades: list[dict[str, object]] = []
    exit_infer_state = L3ExitInferenceState()
    peak_unreal_atr = float("-inf")
    cox_bundle = l3_load_cox_bundle(l3_meta)
    l3_aux_state: dict[str, float | int] = {}
    infer_cfg = _l3_exit_infer_params(l3_meta)
    runtime_diag: dict[str, Any] = {
        "symbol": symbol,
        "l3_exit_mode": l3_exit_mode,
        "entry_blocked_l1a_regime": 0,
        "entries": 0,
        "policy_exit_count": 0,
        "flip_exit_count": 0,
        "atr_trail_exit_count": 0,
        "soft_enter_hits_early": 0,
        "soft_enter_hits_late": 0,
        "steps_early": 0,
        "steps_late": 0,
        "soft_enter_thr_early_sum": 0.0,
        "soft_enter_thr_late_sum": 0.0,
    }

    in_pos = 0
    entry_idx = -1
    entry_price = 0.0
    entry_atr = 1e-3
    entry_time = None
    hold = 0
    n_bars = len(df) - 1
    bar_iter: range | tqdm = range(n_bars)
    if show_pb and n_bars > 0:
        bar_iter = _oos_tqdm_pbar(
            range(n_bars),
            total=n_bars,
            desc=f"[{symbol}] bar sim",
            unit="bar",
            mininterval=1.0,
            leave=True,
        )
    for i in bar_iter:
        if in_pos == 0:
            straddle_on = float(l2_out.loc[i].get("l2_straddle_on", 0)) > 0.5
            cls = 0 if straddle_on else 1
            conf = float(l2_out.loc[i, "l2_decision_confidence"])
            size = float(l2_out.loc[i, "l2_size"])
            regime_probs = l2_out.loc[i, [f"l2_entry_regime_{idx}" for idx in range(len(L1A_REGIME_COLS))]].to_numpy(dtype=np.float32)
            entry_vol = float(l2_out.loc[i, "l2_entry_vol"])
            pa_state = df.loc[i, PA_STATE_FEATURES] if all(col in df.columns for col in PA_STATE_FEATURES) else None
            entry_min_conf, entry_min_size, max_hold, _ = l3_entry_policy_params(regime_probs, entry_vol, l3_meta, pa_state=pa_state)
            side = l3_entry_side_from_l2(
                cls,
                conf,
                size,
                min_confidence=entry_min_conf,
                min_size=entry_min_size,
            )
            l1a_entry_regime = int(
                np.argmax(l1a_out.loc[i, L1A_REGIME_COLS].to_numpy(dtype=np.float32))
            )
            if side != 0.0 and l1a_entry_regime in block_entry_l1a_regimes:
                runtime_diag["entry_blocked_l1a_regime"] = int(runtime_diag["entry_blocked_l1a_regime"]) + 1
                side = 0.0
            if side != 0.0:
                runtime_diag["entries"] = int(runtime_diag["entries"]) + 1
                straddle_sim_mode = _l3_straddle_sim_mode_enabled(l3_meta)
                in_pos = 1 if straddle_sim_mode else int(side)
                entry_idx = i
                entry_open = float(df["open"].iloc[i + 1])
                if straddle_sim_mode:
                    dte_days = _oos_default_straddle_dte_days(l3_meta)
                    entry_iv = float(df["l3_base_iv"].iloc[i + 1]) if "l3_base_iv" in df.columns else max(entry_vol, 0.05)
                    strike = float(entry_open)
                    sim = StraddleSimulator(risk_free_rate=float(l3_meta.get("l3_straddle_risk_free_rate", 0.04) or 0.04))
                    raw_premium = sim.straddle_price(entry_open, strike, dte_days / 252.0, entry_iv)
                    entry_price = float(raw_premium * (1.0 + slip_f + comm_f))
                    l3_aux_state["strike"] = strike
                    l3_aux_state["entry_iv"] = entry_iv
                    l3_aux_state["entry_underlying"] = float(entry_open)
                    l3_aux_state["dte_days"] = int(dte_days)
                    l3_aux_state["risk_free_rate"] = float(l3_meta.get("l3_straddle_risk_free_rate", 0.04) or 0.04)
                else:
                    entry_price = _oos_adjust_fill(entry_open, is_buy=(in_pos == 1), slip_f=slip_f, comm_f=comm_f)
                entry_atr = max(float(df["lbl_atr"].iloc[i]), 1e-3)
                entry_time = df["time_key"].iloc[i + 1]
                hold = 0
                peak_unreal_atr = float("-inf")
                exit_infer_state.reset()
                p0 = float(np.clip(conf, 0.05, 0.95))
                l3_aux_state.clear()
                l3_aux_state["log_odds"] = float(np.log(p0 / (1.0 - p0)))
                l3_aux_state["bars_since_peak"] = 0
                l3_aux_state["prev_unreal"] = 0.0
                l3_aux_state["running_mfe"] = 0.0
                l3_aux_state["running_mae"] = 0.0
                if hybrid and l3_exit_mode == "l3":
                    ref = max(int(l3_traj_cfg.max_seq_len), max_hold)
                    traj_buf = L3TrajRollingState(
                        max_seq_len=int(l3_traj_cfg.max_seq_len),
                        max_seq_ref=ref,
                        seq_feat_dim=int(l3_traj_cfg.seq_feat_dim),
                        mfe_norm_scale=float(getattr(l3_traj_cfg, "mfe_norm_scale", 5.0)),
                        mae_norm_scale=float(getattr(l3_traj_cfg, "mae_norm_scale", 5.0)),
                    )
        else:
            hold += 1
            static, peak_unreal_atr, dd_from_peak = _build_l3_feature_vector(
                i,
                in_pos,
                hold,
                entry_idx,
                entry_price,
                entry_atr,
                df,
                l1a_out,
                l2_out,
                peak_unreal_atr,
                l3_aux_state,
                l3_meta,
                cox_bundle,
            )
            static = static.ravel()
            if "l2_straddle_on" in l2_out.columns:
                flip = 0 if float(l2_out.loc[i, "l2_straddle_on"]) > 0.5 else 1
                abstain_like = flip == 1
                flip_against = False
            else:
                flip = int(l2_out.loc[i, "l2_decision_class"])
                p_long = float(l2_out.loc[i, "l2_decision_long"]) if "l2_decision_long" in l2_out.columns else 0.0
                p_short = float(l2_out.loc[i, "l2_decision_short"]) if "l2_decision_short" in l2_out.columns else 0.0
                directional_mass = max(p_long + p_short, 1e-6)
                directional_conf = abs(p_long - p_short) / directional_mass
                abstain_like = (flip == 1) or (directional_conf < l2_abstain_margin)
                flip_against = (not abstain_like) and ((in_pos == 1 and flip == 2) or (in_pos == -1 and flip == 0))
            if l3_exit_mode == "atr_trailing":
                straddle = _l3_straddle_sim_mode_enabled(l3_meta)
                dd = float(dd_from_peak)
                close_px = max(float(df["close"].iloc[i]), 1e-9)
                atr_b = max(float(df["lbl_atr"].iloc[i]), 1e-9)
                if straddle:
                    trail_th = float(atr_trail_mult * (atr_b / close_px))
                else:
                    trail_th = float(atr_trail_mult)
                policy_exit = (hold >= atr_trail_min_hold) and (dd + 1e-12 >= trail_th)
                value_left = 0.0
                exit_prob = 0.0
                ent_soft = float(trail_th)
                lev_soft = float(dd)
                state_hold_ref = 0
                if policy_exit:
                    runtime_diag["atr_trail_exit_count"] = int(runtime_diag["atr_trail_exit_count"]) + 1
            else:
                if hybrid and traj_buf is not None and l3_exit_mode == "l3":
                    ts = np.datetime64(df["time_key"].iloc[i])
                    if _l3_straddle_sim_mode_enabled(l3_meta):
                        price_rel_prev = float(
                            df["close"].iloc[max(i - 1, 0)]
                            / max(float(l3_aux_state.get("entry_underlying", df["close"].iloc[max(i - 1, 0)])), 1e-9)
                        )
                        price_rel_now = float(
                            df["close"].iloc[i] / max(float(l3_aux_state.get("entry_underlying", df["close"].iloc[i])), 1e-9)
                        )
                        traj_buf.append_step_straddle(
                            float(static[_sidx["l3_straddle_pnl_pct"]]),
                            int(hold),
                            ts,
                            price_rel_prev,
                            price_rel_now,
                            float(static[_sidx["l3_underlying_abs_move"]]),
                            float(static[_sidx["l3_straddle_iv"]]),
                            float(static[_sidx["l3_vol_surprise"]]),
                            float(static[_sidx["l3_regime_divergence"]]),
                            float(static[_sidx["l3_straddle_vega"]]),
                            float(abs(static[_sidx["l3_straddle_theta"]])),
                        )
                    else:
                        close_prev = float(df["close"].iloc[i - 1])
                        traj_buf.append_step(
                            float(static[_sidx["l3_unreal_pnl_atr"]]),
                            int(hold),
                            ts,
                            close_prev,
                            float(df["close"].iloc[i]),
                            float(df["high"].iloc[i]),
                            float(df["low"].iloc[i]),
                            entry_atr,
                            float(static[_sidx["l3_vol_surprise"]]),
                            float(static[_sidx["l3_regime_divergence"]]),
                            float(static[_sidx["l3_live_mfe"]]),
                            float(static[_sidx["l3_live_mae"]]),
                        )
                    seq, sl = traj_buf.padded_sequence()
                    emb = l3_single_trajectory_embedding(l3_traj_enc, seq, sl, dev)
                    feat_vec = np.concatenate([static, emb], dtype=np.float32).reshape(1, -1)
                else:
                    feat_vec = static.reshape(1, -1)
                if feat_vec.shape[1] != len(feature_cols):
                    raise RuntimeError(f"L3 feature width mismatch: {feat_vec.shape[1]} vs {len(feature_cols)}")
                exit_raw = float(exit_model.predict(feat_vec)[0])
                raw_arr = np.asarray([exit_raw], dtype=np.float64)
                if exit_calibrator is not None:
                    exit_prob = float(np.clip(_apply_l3_exit_calibrator(raw_arr, exit_calibrator)[0], 0.0, 1.0))
                else:
                    exit_prob = float(np.clip(exit_raw, 0.0, 1.0))
                if value_model is None:
                    value_left = 0.0
                elif value_nonzero_model is None:
                    value_left = float(value_model.predict(feat_vec)[0])
                else:
                    mu = float(value_model.predict(feat_vec)[0])
                    p_nz = float(np.clip(value_nonzero_model.predict(feat_vec)[0], 0.0, 1.0))
                    value_left = float(mu * (p_nz ** float(np.clip(value_hurdle_power, 0.5, 2.0))))
                exit_state_probs = l1a_out.loc[i, L1A_REGIME_COLS].to_numpy(dtype=np.float32)
                exit_state_vol = float(l1a_out.loc[i, "l1a_vol_forecast"])
                pa_state = df.loc[i, PA_STATE_FEATURES] if all(col in df.columns for col in PA_STATE_FEATURES) else None
                exit_prob_threshold, value_left_threshold, state_hold_ref, _, value_policy_mode, value_tie_margin = l3_exit_policy_params(
                    exit_state_probs,
                    exit_state_vol,
                    hold,
                    l3_meta,
                    pa_state=pa_state,
                )
                ent_soft, lev_soft = _l3_soft_exit_hysteresis_thresholds(
                    hold,
                    enter_thr=float(exit_prob_threshold),
                    leave_thr=max(float(value_tie_margin), float(exit_prob_threshold) - 0.20),
                    hold_ref_bars=int(state_hold_ref),
                    infer_cfg=infer_cfg,
                )
                if hold < int(state_hold_ref):
                    runtime_diag["steps_early"] = int(runtime_diag["steps_early"]) + 1
                    runtime_diag["soft_enter_thr_early_sum"] = float(runtime_diag["soft_enter_thr_early_sum"]) + float(
                        ent_soft
                    )
                    if exit_prob >= ent_soft:
                        runtime_diag["soft_enter_hits_early"] = int(runtime_diag["soft_enter_hits_early"]) + 1
                else:
                    runtime_diag["steps_late"] = int(runtime_diag["steps_late"]) + 1
                    runtime_diag["soft_enter_thr_late_sum"] = float(runtime_diag["soft_enter_thr_late_sum"]) + float(
                        ent_soft
                    )
                    if exit_prob >= ent_soft:
                        runtime_diag["soft_enter_hits_late"] = int(runtime_diag["soft_enter_hits_late"]) + 1
                policy_exit, exit_infer_state = l3_exit_decision_live(
                    exit_prob,
                    value_left,
                    exit_infer_state,
                    hold,
                    exit_prob_threshold=exit_prob_threshold,
                    value_left_threshold=value_left_threshold,
                    value_policy_mode=value_policy_mode,
                    value_tie_margin=value_tie_margin,
                    hold_ref_bars=int(state_hold_ref),
                    meta=l3_meta,
                )
            if policy_exit or flip_against:
                if policy_exit:
                    runtime_diag["policy_exit_count"] = int(runtime_diag["policy_exit_count"]) + 1
                elif flip_against:
                    runtime_diag["flip_exit_count"] = int(runtime_diag["flip_exit_count"]) + 1
                exit_open = float(df["open"].iloc[i + 1])
                if _l3_straddle_sim_mode_enabled(l3_meta):
                    dte_days = int(l3_aux_state.get("dte_days", _oos_default_straddle_dte_days(l3_meta)))
                    strike = float(l3_aux_state.get("strike", exit_open))
                    entry_iv = float(l3_aux_state.get("entry_iv", df["l3_base_iv"].iloc[max(i, 0)] if "l3_base_iv" in df.columns else 0.25))
                    current_iv = float(0.65 * entry_iv + 0.35 * (float(df["l3_base_iv"].iloc[i + 1]) if "l3_base_iv" in df.columns else entry_iv))
                    t_remaining = max(dte_days / 252.0 - (hold + 1) / (390.0 * 252.0), 1e-9)
                    sim = StraddleSimulator(risk_free_rate=float(l3_aux_state.get("risk_free_rate", 0.04)))
                    raw_exit_value = sim.straddle_price(exit_open, strike, t_remaining, current_iv)
                    exit_price = float(raw_exit_value * (1.0 - slip_f - comm_f))
                    ret = (exit_price / entry_price) - 1.0
                else:
                    exit_price = _oos_adjust_fill(exit_open, is_buy=(in_pos == -1), slip_f=slip_f, comm_f=comm_f)
                    ret = (exit_price / entry_price - 1.0) * in_pos
                if l3_exit_mode == "atr_trailing" and policy_exit:
                    exit_reason = "ATR_Trail_Exit"
                elif policy_exit:
                    exit_reason = "Policy_Exit"
                else:
                    exit_reason = "Signal_Flip"
                trades.append(
                    {
                        "symbol": symbol,
                        "entry_time": entry_time,
                        "exit_time": df["time_key"].iloc[i + 1],
                        "direction": "STRADDLE" if _l3_straddle_sim_mode_enabled(l3_meta) else ("LONG" if in_pos == 1 else "SHORT"),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return": ret,
                        "holding_bars": hold,
                        "entry_regime_id": int(
                            np.argmax(l1a_out.loc[entry_idx, L1A_REGIME_COLS].to_numpy(dtype=np.float32))
                        ) if entry_idx >= 0 else -1,
                        "exit_soft_enter_thr": float(ent_soft),
                        "exit_soft_leave_thr": float(lev_soft),
                        "exit_hold_ref_bars": int(state_hold_ref),
                        "exit_prob": float(exit_prob),
                        "exit_value_left": float(value_left),
                        "exit_reason": exit_reason,
                        **(
                            {
                                "entry_iv": float(l3_aux_state.get("entry_iv", np.nan)),
                                "exit_iv": float(current_iv),
                                "strike": float(l3_aux_state.get("strike", np.nan)),
                                "dte_days": int(l3_aux_state.get("dte_days", 0)),
                                "entry_underlying": float(l3_aux_state.get("entry_underlying", np.nan)),
                                "exit_underlying": float(exit_open),
                            }
                            if _l3_straddle_sim_mode_enabled(l3_meta)
                            else {}
                        ),
                    }
                )
                in_pos = 0
                entry_idx = -1
                traj_buf = None
                peak_unreal_atr = float("-inf")
                l3_aux_state.clear()
                exit_infer_state.reset()
    return pd.DataFrame(trades), df, runtime_diag


def _equity_curve_on_timeline(time_key: pd.Series, trades: pd.DataFrame) -> np.ndarray:
    """Piecewise-constant cumulative equity after each closed trade (aligned to bar timestamps)."""
    n = len(time_key)
    if trades is None or trades.empty or n == 0:
        return np.ones(n, dtype=np.float64)
    t = trades.sort_values("exit_time").reset_index(drop=True)
    r = t["return"].to_numpy(dtype=np.float64)
    eq_at_exit = np.cumprod(1.0 + r)
    ts_bar = pd.to_datetime(time_key).to_numpy()
    ts_exit = pd.to_datetime(t["exit_time"]).to_numpy()
    out = np.ones(n, dtype=np.float64)
    j = 0
    for i in range(n):
        while j < len(ts_exit) and ts_exit[j] <= ts_bar[i]:
            j += 1
        if j > 0:
            out[i] = eq_at_exit[j - 1]
    return out


def _format_metric_legend(m: dict[str, Any]) -> str:
    if int(m.get("n_trades", 0)) == 0:
        return "无成交"
    sh = m.get("sharpe_trade_annualized", float("nan"))
    so = m.get("sortino_trade_annualized", float("nan"))
    sh_s = f"{sh:.2f}" if np.isfinite(sh) else "—"
    so_s = f"{so:.2f}" if np.isfinite(so) else "—"
    pf = m.get("profit_factor", float("nan"))
    pf_s = f"{pf:.2f}" if np.isfinite(pf) else "—"
    cm = m.get("calmar", float("nan"))
    cm_s = f"{cm:.2f}" if np.isfinite(cm) else "—"
    cagr = m.get("cagr_pct", float("nan"))
    cagr_s = f"{cagr:.1f}%" if np.isfinite(cagr) else "—"
    return (
        f"夏普 Sharpe: {sh_s}\n"
        f"索提诺 Sortino: {so_s}\n"
        f"最大回撤 Max DD: {m['max_drawdown_pct']:.2f}%\n"
        f"胜率 Win rate: {m['win_rate']:.2%}\n"
        f"总收益 Total: {m['total_return_pct']:.2f}%\n"
        f"CAGR: {cagr_s}\n"
        f"盈亏比 Profit factor: {pf_s}\n"
        f"卡玛 Calmar: {cm_s}\n"
        f"笔数 Trades: {m['n_trades']}"
    )


def plot_oos_price_and_equity(
    symbol: str,
    df_price: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: dict[str, Any],
    out_path: Path,
) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if df_price is None or df_price.empty or "time_key" not in df_price.columns:
        print(f"  [OOS plot] skip {symbol}: empty price frame", flush=True)
        return
    price_col = "close" if "close" in df_price.columns else None
    if price_col is None:
        print(f"  [OOS plot] skip {symbol}: no close column", flush=True)
        return

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti TC", "Noto Sans CJK SC", "DejaVu Sans"]

    tk = pd.to_datetime(df_price["time_key"])
    px = pd.to_numeric(df_price[price_col], errors="coerce").to_numpy(dtype=np.float64)
    eq = _equity_curve_on_timeline(tk, trades)
    cum_ret_pct = (eq - 1.0) * 100.0

    fig, ax1 = plt.subplots(figsize=(14, 6), layout="constrained")
    c_price = "#1f77b4"
    c_eq = "#ff7f0e"
    (ln_price,) = ax1.plot(tk, px, color=c_price, linewidth=1.0, label="价格 Close")
    ax1.set_xlabel("时间")
    ax1.set_ylabel("价格", color=c_price)
    ax1.tick_params(axis="y", labelcolor=c_price)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()

    ax2 = ax1.twinx()
    (ln_eq,) = ax2.plot(tk, cum_ret_pct, color=c_eq, linewidth=1.2, linestyle="-", label="累计收益 %")
    ax2.set_ylabel("累计收益 (%)", color=c_eq)
    ax2.tick_params(axis="y", labelcolor=c_eq)
    ax2.axhline(0.0, color=c_eq, linewidth=0.6, alpha=0.35, zorder=0)

    leg_lines = [ln_price, ln_eq]
    leg_labels = [ln.get_label() for ln in leg_lines]
    stats_txt = _format_metric_legend(metrics)
    ax1.text(
        0.01,
        0.99,
        stats_txt,
        transform=ax1.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.9, edgecolor="0.45"),
    )
    ax1.legend(
        leg_lines,
        leg_labels,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.72),
        framealpha=0.92,
        fontsize=9,
    )

    ax1.set_title(_oos_figure_title(symbol))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OOS plot] saved -> {out_path}", flush=True)


def summarize_trade_returns(trades: pd.DataFrame, *, label: str) -> dict[str, Any]:
    """Sequential compounding on closed trades sorted by exit_time (one position redeployed per trade)."""
    if trades is None or trades.empty:
        return {"label": label, "n_trades": 0}
    t = trades.sort_values("exit_time").reset_index(drop=True)
    r = t["return"].to_numpy(dtype=np.float64)
    n = int(len(r))
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    max_dd_frac = float(np.min(dd))
    total_return_frac = float(equity[-1] - 1.0)
    t0 = pd.Timestamp(t["exit_time"].iloc[0])
    t1 = pd.Timestamp(t["exit_time"].iloc[-1])
    span_days = max((t1 - t0).total_seconds() / 86400.0, 1e-6)
    years = float(span_days / 365.25)
    cagr_frac = float((max(equity[-1], 1e-12) ** (1.0 / max(years, 1e-6))) - 1.0) if years > 0 else float("nan")
    win_rate = float(np.mean(r > 0.0))
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1)) if n > 1 else 0.0
    trades_per_year = float(n / max(years, 1e-6))
    sharpe = float((mu / sd) * np.sqrt(trades_per_year)) if sd > 1e-12 else float("nan")
    neg = r[r < 0.0]
    dsd = float(np.std(neg, ddof=1)) if neg.size > 1 else float("nan")
    sortino = float((mu / dsd) * np.sqrt(trades_per_year)) if np.isfinite(dsd) and dsd > 1e-12 else float("nan")
    gross_win = float(np.sum(r[r > 0.0]))
    gross_loss = float(np.sum(r[r < 0.0]))
    profit_factor = float(gross_win / abs(gross_loss)) if gross_loss < -1e-12 else float("nan")
    calmar = float(cagr_frac / abs(max_dd_frac)) if max_dd_frac < -1e-8 and np.isfinite(cagr_frac) else float("nan")
    return {
        "label": label,
        "n_trades": n,
        "first_exit": str(t0),
        "last_exit": str(t1),
        "span_calendar_days": float(span_days),
        "total_return_frac": total_return_frac,
        "total_return_pct": float(100.0 * total_return_frac),
        "cagr_frac": cagr_frac,
        "cagr_pct": float(100.0 * cagr_frac) if np.isfinite(cagr_frac) else float("nan"),
        "max_drawdown_frac": max_dd_frac,
        "max_drawdown_pct": float(100.0 * max_dd_frac),
        "win_rate": win_rate,
        "mean_return_per_trade_frac": mu,
        "std_return_per_trade_frac": sd,
        "sharpe_trade_annualized": sharpe,
        "sortino_trade_annualized": sortino,
        "profit_factor": profit_factor,
        "calmar": calmar,
        "avg_holding_bars": float(t["holding_bars"].mean()) if "holding_bars" in t.columns else float("nan"),
    }


def summarize_exit_path_diagnostics(trades: pd.DataFrame, runtime_diag: dict[str, Any], *, label: str) -> dict[str, Any]:
    t = trades if trades is not None else pd.DataFrame()
    total = int(len(t))
    policy_n = int((t["exit_reason"] == "Policy_Exit").sum()) if total > 0 and "exit_reason" in t.columns else 0
    flip_n = int((t["exit_reason"] == "Signal_Flip").sum()) if total > 0 and "exit_reason" in t.columns else 0
    by_reason_hold: dict[str, dict[str, float]] = {}
    if total > 0 and {"exit_reason", "holding_bars"}.issubset(t.columns):
        for reason, grp in t.groupby("exit_reason"):
            g = pd.to_numeric(grp["holding_bars"], errors="coerce")
            by_reason_hold[str(reason)] = {
                "n": int(len(grp)),
                "share": float(len(grp) / max(total, 1)),
                "avg_hold_bars": float(np.nanmean(g)),
                "p50_hold_bars": float(np.nanmedian(g)),
                "p90_hold_bars": float(np.nanquantile(g, 0.90)),
            }
    hold_hist: dict[str, int] = {}
    if total > 0 and "holding_bars" in t.columns:
        vc = pd.to_numeric(t["holding_bars"], errors="coerce").dropna().astype(int).value_counts().sort_index()
        hold_hist = {str(int(k)): int(v) for k, v in vc.items()}
    early_steps = int(runtime_diag.get("steps_early", 0))
    late_steps = int(runtime_diag.get("steps_late", 0))
    thr_early_sum = float(runtime_diag.get("soft_enter_thr_early_sum", 0.0))
    thr_late_sum = float(runtime_diag.get("soft_enter_thr_late_sum", 0.0))
    return {
        "label": label,
        "n_trades": total,
        "policy_exit_share": float(policy_n / max(total, 1)),
        "flip_exit_share": float(flip_n / max(total, 1)),
        "runtime_policy_exit_count": int(runtime_diag.get("policy_exit_count", 0)),
        "runtime_flip_exit_count": int(runtime_diag.get("flip_exit_count", 0)),
        "entry_count_runtime": int(runtime_diag.get("entries", 0)),
        "soft_enter_hit_rate_early": float(runtime_diag.get("soft_enter_hits_early", 0) / max(early_steps, 1)),
        "soft_enter_hit_rate_late": float(runtime_diag.get("soft_enter_hits_late", 0) / max(late_steps, 1)),
        "soft_enter_thr_early_mean": float(thr_early_sum / max(early_steps, 1)),
        "soft_enter_thr_late_mean": float(thr_late_sum / max(late_steps, 1)),
        "holding_bars_histogram": hold_hist,
        "hold_by_exit_reason": by_reason_hold,
    }


def _oos_json_float(x: float) -> float | None:
    """Finite floats for JSON; None if missing or non-finite (avoids NaN in summaries)."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if np.isfinite(xf) else None


def summarize_regime_diagnostics(trades: pd.DataFrame, *, label: str) -> dict[str, Any]:
    t = trades if trades is not None else pd.DataFrame()
    out = {"label": label, "entry_regime_stats": []}
    if t.empty or "entry_regime_id" not in t.columns:
        return out
    arr = pd.to_numeric(t["entry_regime_id"], errors="coerce")
    t2 = t.loc[arr.notna()].copy()
    if t2.empty:
        return out
    t2["entry_regime_id"] = arr[arr.notna()].astype(int)
    stats = []
    for rid, grp in t2.groupby("entry_regime_id"):
        ret = pd.to_numeric(grp["return"], errors="coerce").to_numpy(dtype=np.float64)
        ret = ret[np.isfinite(ret)]
        hold = pd.to_numeric(grp.get("holding_bars"), errors="coerce")
        wins = ret[ret > 0.0]
        losses = ret[ret < 0.0]
        avg_win = float(np.mean(wins)) if wins.size else float("nan")
        avg_loss = float(np.mean(losses)) if losses.size else float("nan")
        if wins.size and losses.size and avg_loss < 0.0:
            avg_pl_ratio = avg_win / (-avg_loss)
        else:
            avg_pl_ratio = float("nan")
        loss_sum = float(np.sum(losses))
        if loss_sum < 0.0:
            profit_factor = float(np.sum(wins) / (-loss_sum)) if wins.size or losses.size else float("nan")
        else:
            profit_factor = float("nan")
        stats.append(
            {
                "regime_id": int(rid),
                "n": int(len(grp)),
                "share": float(len(grp) / max(len(t2), 1)),
                "mean_return_frac": _oos_json_float(float(np.mean(ret))) if ret.size else None,
                "win_rate": float(np.mean(ret > 0.0)) if ret.size else None,
                "avg_holding_bars": float(np.nanmean(hold)),
                "avg_win_return_frac": _oos_json_float(avg_win),
                "avg_loss_return_frac": _oos_json_float(avg_loss),
                "avg_profit_loss_ratio": _oos_json_float(avg_pl_ratio),
                "profit_factor": _oos_json_float(profit_factor),
            }
        )
    out["entry_regime_stats"] = sorted(stats, key=lambda x: x["regime_id"])
    return out


def evaluate_symbol_asymmetry_guardrail(summary_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = {str(row.get("label")): row for row in summary_blocks}
    qqq = by_label.get("QQQ")
    spy = by_label.get("SPY")
    out = {"enabled": True, "flagged": False, "reasons": [], "metrics": {}}
    if not qqq or not spy:
        return out
    ret_gap = float(qqq.get("mean_return_per_trade_frac", 0.0) - spy.get("mean_return_per_trade_frac", 0.0))
    sharpe_gap = float(qqq.get("sharpe_trade_annualized", 0.0) - spy.get("sharpe_trade_annualized", 0.0))
    win_gap = float(qqq.get("win_rate", 0.0) - spy.get("win_rate", 0.0))
    ret_gap_thr = float(os.environ.get("L3_OOS_SYMBOL_GUARD_MAX_RET_GAP", "0.00012"))
    sharpe_gap_thr = float(os.environ.get("L3_OOS_SYMBOL_GUARD_MAX_SHARPE_GAP", "0.80"))
    win_gap_thr = float(os.environ.get("L3_OOS_SYMBOL_GUARD_MAX_WIN_GAP", "0.08"))
    out["metrics"] = {
        "mean_return_per_trade_gap": ret_gap,
        "sharpe_gap": sharpe_gap,
        "win_rate_gap": win_gap,
        "thresholds": {
            "max_ret_gap": ret_gap_thr,
            "max_sharpe_gap": sharpe_gap_thr,
            "max_win_gap": win_gap_thr,
        },
    }
    if abs(ret_gap) > ret_gap_thr:
        out["flagged"] = True
        out["reasons"].append("mean_return_per_trade_gap_exceeds_threshold")
    if abs(sharpe_gap) > sharpe_gap_thr:
        out["flagged"] = True
        out["reasons"].append("sharpe_gap_exceeds_threshold")
    if abs(win_gap) > win_gap_thr:
        out["flagged"] = True
        out["reasons"].append("win_rate_gap_exceeds_threshold")
    return out


def _print_oos_summary(blocks: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 70)
    print("  OOS performance (sequential compound on closed trades by exit_time)")
    print("  Note: COMBINED chains QQQ+SPY exits in one curve — not a capital-budgeted portfolio.")
    print("=" * 70)
    for row in blocks:
        if int(row.get("n_trades", 0)) == 0:
            print(f"\n  [{row.get('label')}] no trades")
            continue
        print(f"\n  [{row.get('label')}]  trades={row['n_trades']}  {row.get('first_exit', '')} .. {row.get('last_exit', '')}")
        cagr_pct = row.get("cagr_pct", float("nan"))
        cagr_s = f"{cagr_pct:.2f}%" if np.isfinite(cagr_pct) else "nan"
        print(
            f"    total_return={row['total_return_pct']:.2f}%  CAGR={cagr_s}  "
            f"max_dd={row['max_drawdown_pct']:.2f}%",
            flush=True,
        )
        print(
            f"    win_rate={row['win_rate']:.2%}  Sharpe~{row['sharpe_trade_annualized']:.3f}  "
            f"Sortino~{row['sortino_trade_annualized']:.3f}  profit_factor={row['profit_factor']:.3f}  Calmar~{row['calmar']:.3f}",
            flush=True,
        )
        print(
            f"    mean/trade={100.0 * row['mean_return_per_trade_frac']:.4f}%  "
            f"avg_hold_bars={row['avg_holding_bars']:.1f}",
            flush=True,
        )


def _print_exit_path_diagnostics(blocks: list[dict[str, Any]]) -> None:
    if not blocks:
        return
    print("\n" + "=" * 70)
    print("  Exit-path diagnostics (runtime + closed trades)")
    print("=" * 70)
    for row in blocks:
        n = int(row.get("n_trades", 0))
        if n <= 0:
            continue
        print(
            f"\n  [{row.get('label')}] trades={n}  policy_exit={100.0 * float(row.get('policy_exit_share', 0.0)):.1f}%  "
            f"flip_exit={100.0 * float(row.get('flip_exit_share', 0.0)):.1f}%",
            flush=True,
        )
        print(
            f"    soft_enter_hit_rate early={100.0 * float(row.get('soft_enter_hit_rate_early', 0.0)):.1f}%  "
            f"late={100.0 * float(row.get('soft_enter_hit_rate_late', 0.0)):.1f}%  "
            f"thr_mean early={float(row.get('soft_enter_thr_early_mean', float('nan'))):.3f}  "
            f"late={float(row.get('soft_enter_thr_late_mean', float('nan'))):.3f}",
            flush=True,
        )
        hold_hist = row.get("holding_bars_histogram", {}) or {}
        if hold_hist:
            items = sorted(((int(k), int(v)) for k, v in hold_hist.items()), key=lambda x: x[0])
            top = ", ".join([f"{k}:{v}" for k, v in items[:8]])
            print(f"    holding_bars histogram(top)={top}", flush=True)


def _print_regime_diagnostics(blocks: list[dict[str, Any]]) -> None:
    if not blocks:
        return
    print("\n" + "=" * 70)
    print("  Entry-regime diagnostics")
    print("=" * 70)
    for row in blocks:
        stats = row.get("entry_regime_stats") or []
        if not stats:
            continue
        print(f"\n  [{row.get('label')}] entry-regime decomposition:", flush=True)
        print(
            "    (avg_profit_loss_ratio = mean(winning returns) / mean(|losing returns|); "
            "profit_factor = sum(wins)/sum(|losses|))",
            flush=True,
        )
        for it in stats:
            mr = it.get("mean_return_frac")
            mr_s = f"{100.0*float(mr):.4f}%" if mr is not None else "—"
            wr = it.get("win_rate")
            wr_s = f"{100.0*float(wr):.1f}%" if wr is not None else "—"
            plr = it.get("avg_profit_loss_ratio")
            plr_s = f"{float(plr):.3f}" if plr is not None else "—"
            pf = it.get("profit_factor")
            pf_s = f"{float(pf):.3f}" if pf is not None else "—"
            print(
                f"    regime={int(it['regime_id'])}  n={int(it['n'])}  share={100.0*float(it['share']):.1f}%  "
                f"mean_ret={mr_s}  win_rate={wr_s}  "
                f"avg_pl_ratio={plr_s}  profit_factor={pf_s}  "
                f"avg_hold={float(it['avg_holding_bars']):.2f}",
                flush=True,
            )


def main():
    print("=" * 70, flush=True)
    print("  Running OOS Dual-View Pipeline (L1a -> L1b -> L2 -> L3)", flush=True)
    _oem = (os.environ.get("OOS_L3_EXIT_MODE") or "l3").strip() or "l3"
    _reg_b = (os.environ.get("OOS_BLOCK_ENTRY_L1A_REGIMES") or "").strip() or "—"
    print(
        f"  L3 exit: OOS_L3_EXIT_MODE={_oem}  (use atr_trailing for ATR ablation; OOS_ATR_TRAIL_MULT, OOS_ATR_TRAIL_MIN_HOLD)",
        flush=True,
    )
    print(f"  Regime block: OOS_BLOCK_ENTRY_L1A_REGIMES={_reg_b}  (comma ids, e.g. 1 to skip regime 1 entries)", flush=True)
    print(f"  OOS window: [{OOS_START}, {OOS_END})  (override via env OOS_START / OOS_END)", flush=True)
    if (os.environ.get("OOS_CHART_TITLE") or "").strip():
        print("  Chart title: OOS_CHART_TITLE is set (see _oos_figure_title format keys).", flush=True)
    print(f"  Results dir: {RESULTS_DIR.resolve()}  (override via OOS_RESULTS_DIR)", flush=True)
    _sb, _cb = _oos_cost_fracs()
    if _sb > 0.0 or _cb > 0.0:
        print(
            f"  Execution costs: slippage={float(os.environ.get('OOS_SLIPPAGE_BPS', '0') or '0'):g} bps  "
            f"commission_per_side={float(os.environ.get('OOS_COMMISSION_BPS_PER_SIDE', '0') or '0'):g} bps",
            flush=True,
        )
    print("=" * 70, flush=True)
    print("  Loading stack artifacts (L1a → L3)...", flush=True)
    _ensure_backtest_artifacts_exist()
    # Load L1a first, then lazy-import L2/L3 modules (prevents import-order segfaults on some macOS setups).
    l1a_model, l1a_meta = load_l1a_market_encoder()
    _lazy_import_stack_modules()
    l1b_models, l1b_meta = load_l1b_market_descriptor()
    l2_models, l2_meta = load_l2_trade_decision()
    _l2_da = l2_meta.get("direction_available", "direction" in l2_meta.get("model_files", {}))
    if not _l2_da and l2_meta.get("decision_mode") == "two_stage":
        print(
            "  [L2] NOTE: direction_available=False — no direction head; "
            "trade uses gate threshold only; long/short probs symmetric at d=0.5 (see l2_meta).",
            flush=True,
        )
    l3_models, l3_meta = load_l3_exit_manager()
    l3_enc, l3_tcfg = load_l3_trajectory_encoder_for_infer(l3_meta)
    if l3_enc is not None:
        l3_enc = l3_enc.to(TORCH_DEVICE)
    print("  Artifacts loaded. Starting per-symbol inference + backtest.", flush=True)
    all_trades = []
    plot_tasks: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    diag_by_symbol: dict[str, dict[str, Any]] = {}
    for sym in ["QQQ", "SPY"]:
        tr_df, price_df, sym_diag = run_single_symbol(
            sym,
            l1a_model,
            l1a_meta,
            l1b_models,
            l1b_meta,
            l2_models,
            l2_meta,
            l3_models,
            l3_meta,
            l3_traj_enc=l3_enc,
            l3_traj_cfg=l3_tcfg,
            torch_device=TORCH_DEVICE,
        )
        diag_by_symbol[sym] = dict(sym_diag or {})
        plot_tasks.append((sym, price_df, tr_df))
        if not tr_df.empty:
            all_trades.append(tr_df)
            tr_df.to_csv(RESULTS_DIR / f"trades_{sym}.csv", index=False)
            print(f"[{sym}] generated {len(tr_df)} trades.", flush=True)
        else:
            print(f"[{sym}] generated 0 trades.", flush=True)
    summary_blocks: list[dict[str, Any]] = []
    exit_diag_blocks: list[dict[str, Any]] = []
    regime_diag_blocks: list[dict[str, Any]] = []
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv(RESULTS_DIR / "trades_ALL.csv", index=False)
        print(f"\nTotal trades: {len(combined)}")
        print(f"Win Rate: {(combined['return'] > 0).mean():.2%}")
        print(f"Avg Return: {combined['return'].mean():.4%}")
        for sym in ["QQQ", "SPY"]:
            part = combined[combined["symbol"] == sym] if "symbol" in combined.columns else pd.DataFrame()
            if not part.empty:
                summary_blocks.append(summarize_trade_returns(part, label=sym))
                exit_diag_blocks.append(
                    summarize_exit_path_diagnostics(part, diag_by_symbol.get(sym, {}), label=sym)
                )
                regime_diag_blocks.append(summarize_regime_diagnostics(part, label=sym))
        summary_blocks.append(summarize_trade_returns(combined, label="COMBINED"))
        combined_diag = {
            "entries": int(sum(int((diag_by_symbol.get(sym) or {}).get("entries", 0)) for sym in ["QQQ", "SPY"])),
            "policy_exit_count": int(
                sum(int((diag_by_symbol.get(sym) or {}).get("policy_exit_count", 0)) for sym in ["QQQ", "SPY"])
            ),
            "flip_exit_count": int(
                sum(int((diag_by_symbol.get(sym) or {}).get("flip_exit_count", 0)) for sym in ["QQQ", "SPY"])
            ),
            "soft_enter_hits_early": int(
                sum(int((diag_by_symbol.get(sym) or {}).get("soft_enter_hits_early", 0)) for sym in ["QQQ", "SPY"])
            ),
            "soft_enter_hits_late": int(
                sum(int((diag_by_symbol.get(sym) or {}).get("soft_enter_hits_late", 0)) for sym in ["QQQ", "SPY"])
            ),
            "steps_early": int(sum(int((diag_by_symbol.get(sym) or {}).get("steps_early", 0)) for sym in ["QQQ", "SPY"])),
            "steps_late": int(sum(int((diag_by_symbol.get(sym) or {}).get("steps_late", 0)) for sym in ["QQQ", "SPY"])),
            "soft_enter_thr_early_sum": float(
                sum(float((diag_by_symbol.get(sym) or {}).get("soft_enter_thr_early_sum", 0.0)) for sym in ["QQQ", "SPY"])
            ),
            "soft_enter_thr_late_sum": float(
                sum(float((diag_by_symbol.get(sym) or {}).get("soft_enter_thr_late_sum", 0.0)) for sym in ["QQQ", "SPY"])
            ),
        }
        exit_diag_blocks.append(summarize_exit_path_diagnostics(combined, combined_diag, label="COMBINED"))
        regime_diag_blocks.append(summarize_regime_diagnostics(combined, label="COMBINED"))
        symbol_guard = evaluate_symbol_asymmetry_guardrail(summary_blocks)
        _print_oos_summary(summary_blocks)
        _print_exit_path_diagnostics(exit_diag_blocks)
        _print_regime_diagnostics(regime_diag_blocks)
        if bool(symbol_guard.get("flagged", False)):
            print(
                "\n  [L3][WARN][SYMBOL_GUARD] asymmetry guard flagged: "
                + ", ".join(symbol_guard.get("reasons", [])),
                flush=True,
            )
        eq_rows: list[dict[str, Any]] = []
        for label, df_sub in [("QQQ", combined[combined["symbol"] == "QQQ"]), ("SPY", combined[combined["symbol"] == "SPY"]), ("COMBINED", combined)]:
            if df_sub.empty:
                continue
            ts = df_sub.sort_values("exit_time").reset_index(drop=True)
            r = ts["return"].to_numpy(dtype=np.float64)
            equity = np.cumprod(1.0 + r)
            for i in range(len(ts)):
                eq_rows.append(
                    {
                        "curve": label,
                        "i": i + 1,
                        "exit_time": ts["exit_time"].iloc[i],
                        "return_frac": float(r[i]),
                        "equity": float(equity[i]),
                    }
                )
        if eq_rows:
            pd.DataFrame(eq_rows).to_csv(RESULTS_DIR / "oos_equity_curves.csv", index=False)
        m_atr, m_min = _oos_atr_trailing_params()
        try:
            oem = _oos_l3_exit_mode()
        except ValueError:
            oem = "l3"
        payload = {
            "oos_start": OOS_START,
            "oos_end": OOS_END,
            "backtest_config": {
                "OOS_L3_EXIT_MODE": oem,
                "OOS_ATR_TRAIL_MULT": m_atr,
                "OOS_ATR_TRAIL_MIN_HOLD": m_min,
                "OOS_BLOCK_ENTRY_L1A_REGIMES": sorted(_oos_block_entry_l1a_regime_ids()),
            },
            "metrics": summary_blocks,
            "exit_path_diagnostics": exit_diag_blocks,
            "regime_diagnostics": regime_diag_blocks,
            "symbol_asymmetry_guard": symbol_guard,
        }
        with open(RESULTS_DIR / "oos_summary.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
    else:
        m_atr, m_min = _oos_atr_trailing_params()
        try:
            oem = _oos_l3_exit_mode()
        except ValueError:
            oem = "l3"
        with open(RESULTS_DIR / "oos_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "oos_start": OOS_START,
                    "oos_end": OOS_END,
                    "backtest_config": {
                        "OOS_L3_EXIT_MODE": oem,
                        "OOS_ATR_TRAIL_MULT": m_atr,
                        "OOS_ATR_TRAIL_MIN_HOLD": m_min,
                        "OOS_BLOCK_ENTRY_L1A_REGIMES": sorted(_oos_block_entry_l1a_regime_ids()),
                    },
                    "metrics": [],
                    "exit_path_diagnostics": [],
                    "regime_diagnostics": [],
                    "symbol_asymmetry_guard": {"enabled": True, "flagged": False, "reasons": []},
                },
                f,
                indent=2,
            )

    metrics_by = {str(b["label"]): b for b in summary_blocks}
    for sym, price_df, tr_df in plot_tasks:
        m = metrics_by.get(sym) or summarize_trade_returns(tr_df, label=sym)
        plot_oos_price_and_equity(sym, price_df, tr_df, m, RESULTS_DIR / f"oos_chart_{sym}.png")


if __name__ == "__main__":
    main()
