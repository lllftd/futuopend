"""
Out-of-sample backtest for the dual-view stack: L1a -> L1b -> L2 -> L3.

**按 regime 选结构 + L3 平仓（无模板止盈/止损/定时强平）**  
- ``OOS_STRATEGY_ROUTER=vol_regime`` 或 ``l2_regime``、``OOS_L3_EXIT_MODE=l3``；需 L3 meta 开启 straddle 仿真，合成 MTM% 喂给 L3。  
- **l2_regime**：开仓由 **L2** 经 ``l3_entry_side_from_l2`` 与门槛决定；**品种**由 L1a regime 映射到与 vol_regime 相同模板表。  
- **vol_regime**：仅 L1a regime → 结构（无 L2 入场门，具体见主循环逻辑）。  
- 平仓均走 **L3**（+ hybrid 轨迹若启用）；**trade_max_hold** 仍为兜底。  
- **仅跨式 + L3**：``OOS_L2L3_STACK=1`` 或 ``OOS_STRATEGY_ROUTER=straddle`` + ``OOS_L3_EXIT_MODE=l3``。

**归因与 stack router:** 见 ``metrics_by_exit_driver``、``build_backtest_interpretation``。

**OOS 生产基线**（`OOS_BASELINE_*`）：跨式、``mtm_adaptive``、MTM state-hold ``frac=0.55``、不屏蔽 L1a regime、不限仓，均在本文件写死，**不**用环境变量配置。
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKTESTS_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BACKTESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKTESTS_DIR))

from bar_exit_diagnostics import BarExitDiagnostics, default_bar_exit_track_bars

from core.foundation.indicators import atr as compute_atr
from core.foundation.pa_rules import add_pa_features
from core.training.common.constants import (
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
from core.training.l1a import infer_l1a_market_encoder, load_l1a_market_encoder
from core.training.l1b import infer_l1b_market_descriptor, load_l1b_market_descriptor
from core.training.l1b.l1a_bridge import (
    attach_l1a_outputs_to_df,
    l1a_feature_cols_from_l1b_meta,
    meta_expects_l1a_features,
)


def _l3_oos_exit_hold_interactions() -> bool:
    return True


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
load_l3_exit_manager = None
load_l3_trajectory_encoder_for_infer = None
l3_single_trajectory_embedding = None
build_straddle_features = None
build_base_iv_series = None
StraddleSimulator = None
_live_trade_state_from_bar = None
_net_edge_atr_from_state = None
_apply_l3_exit_calibrator = None
_l3_straddle_sim_mode_enabled = None
TORCH_DEVICE = None


def _lazy_import_stack_modules() -> None:
    global infer_l2_trade_decision, load_l2_trade_decision
    global L3ExitInferenceState, L3TrajRollingState, l3_entry_policy_params, l3_entry_side_from_l2
    global l3_exit_decision_live, l3_exit_policy_params
    global load_l3_exit_manager, load_l3_trajectory_encoder_for_infer, l3_single_trajectory_embedding
    global build_straddle_features, build_base_iv_series, StraddleSimulator
    global _live_trade_state_from_bar, _net_edge_atr_from_state
    global _apply_l3_exit_calibrator, _l3_straddle_sim_mode_enabled
    global TORCH_DEVICE
    if infer_l2_trade_decision is not None:
        return
    from core.training.l2 import infer_l2_trade_decision as _infer_l2_trade_decision, load_l2_trade_decision as _load_l2_trade_decision
    from core.training.unified import (
        L3ExitInferenceState as _L3ExitInferenceState,
        L3TrajRollingState as _L3TrajRollingState,
        l3_entry_policy_params as _l3_entry_policy_params,
        l3_entry_side_from_l2 as _l3_entry_side_from_l2,
        l3_exit_decision_live as _l3_exit_decision_live,
        l3_exit_policy_params as _l3_exit_policy_params,
        load_l3_exit_manager as _load_l3_exit_manager,
        load_l3_trajectory_encoder_for_infer as _load_l3_trajectory_encoder_for_infer,
        l3_single_trajectory_embedding as _l3_single_trajectory_embedding,
    )
    from core.training.unified.features import build_straddle_features as _build_straddle_features
    from core.training.unified.simulation.iv_models import build_base_iv_series as _build_base_iv_series
    from core.training.unified.simulation.straddle_simulator import StraddleSimulator as _StraddleSimulator
    from core.training.common.lgbm_utils import _live_trade_state_from_bar as _live_state, _net_edge_atr_from_state as _net_edge
    from core.training.unified.policy_data import (
        _apply_l3_exit_calibrator as _apply_exit_cal,
        _l3_straddle_sim_mode_enabled as _straddle_mode_enabled,
    )
    from core.training.tcn.tcn_constants import DEVICE as _TORCH_DEVICE

    infer_l2_trade_decision = _infer_l2_trade_decision
    load_l2_trade_decision = _load_l2_trade_decision
    L3ExitInferenceState = _L3ExitInferenceState
    L3TrajRollingState = _L3TrajRollingState
    l3_entry_policy_params = _l3_entry_policy_params
    l3_entry_side_from_l2 = _l3_entry_side_from_l2
    l3_exit_decision_live = _l3_exit_decision_live
    l3_exit_policy_params = _l3_exit_policy_params
    load_l3_exit_manager = _load_l3_exit_manager
    load_l3_trajectory_encoder_for_infer = _load_l3_trajectory_encoder_for_infer
    l3_single_trajectory_embedding = _l3_single_trajectory_embedding
    build_straddle_features = _build_straddle_features
    build_base_iv_series = _build_base_iv_series
    StraddleSimulator = _StraddleSimulator
    _live_trade_state_from_bar = _live_state
    _net_edge_atr_from_state = _net_edge
    _apply_l3_exit_calibrator = _apply_exit_cal
    _l3_straddle_sim_mode_enabled = _straddle_mode_enabled
    TORCH_DEVICE = _TORCH_DEVICE

DATA_DIR = _REPO_ROOT / "data"
RESULTS_DIR = Path(os.environ.get("OOS_RESULTS_DIR", str(_REPO_ROOT / "results" / "modeloos")))

# OOS production baseline: edit here for experiments; these are not read from OOS_* environment variables.
OOS_BASELINE_MTM_STATE_HOLD_FRAC: float = 0.55  # ref ≈ int(trade_max_hold * frac), e.g. 16 @ max_hold=30
OOS_BASELINE_L3_EXIT_MODE: str = "mtm_adaptive"
OOS_BASELINE_STRATEGY_ROUTER: str = "straddle"
# Empty set: trade all L1a argmax regimes (no block).
OOS_BASELINE_BLOCK_ENTRY_L1A_REGIMES: frozenset[int] = frozenset()
# 0: no post-sim portfolio overlap cap.
OOS_BASELINE_PORTFOLIO_MAX_OPEN_POSITIONS: int = 0


def _oos_symbols_from_env() -> list[str]:
    """Symbols to backtest. Default QQQ, SPY. Override: OOS_SYMBOLS=QQQ,SPY (comma-separated)."""
    raw = (os.environ.get("OOS_SYMBOLS") or "").strip()
    if not raw:
        return ["QQQ", "SPY"]
    out = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not out:
        raise ValueError("OOS_SYMBOLS is set but resolves to an empty list")
    return out


# Default OOS = bars never used in any layer fit (same as data_prep "Holdout" / L3 holdout: time >= TEST_END).
# L3 OOT train/val lives in [CAL_END, TEST_END); L2 "holdout" slice is [CAL_END, TEST_END) for metrics only (no fit).
# OOS_END default is far in the future so CSVs through "today" stay included; narrow with env if needed.
OOS_START = os.environ.get("OOS_START", str(TEST_END))
OOS_END = os.environ.get("OOS_END", "2035-01-01")
os.makedirs(RESULTS_DIR, exist_ok=True)

_OOS_ENTRY_DIAG_PRINTED: bool = False
_OOS_L3_ENTRY_FUNC_SOURCE_PRINTED: bool = False
_OOS_L3_ENTRY_POLICY_SOURCE_PRINTED: bool = False


def _sigmoid_float(x: float) -> float:
    """``1/(1+exp(-x))`` without ``exp`` overflow when ``x`` is large negative (online log-odds can drift)."""
    if x >= 0.0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


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
    """L3 exit path: fixed to ``OOS_BASELINE_L3_EXIT_MODE`` (``mtm_adaptive`` = MTM vote; env OOS_L3_EXIT_MODE ignored)."""
    return str(OOS_BASELINE_L3_EXIT_MODE)


def _oos_l3_value_exit_unreal_frac() -> float:
    return float(np.clip(float((os.environ.get("OOS_L3_VALUE_EXIT_UNREAL_FRAC") or "0.8").strip() or "0.8"), 0.01, 0.999))


def _oos_mtm_state_hold_ref_for_adaptive(*, trade_max_hold: int) -> int:
    """Gating value passed into ``evaluate_adaptive_mtm_exit(..., state_hold_ref=)`` as eff_min with regime min.

    L3 policy ``l3_exit_policy_params`` often returns ``state_hold_ref`` ≈ full episode length (~max_hold), which
    would make eff_min = max(rmin, ref) ≈ 30 and block all MTM sub-signals until the last bar. For ``mtm_adaptive``
    we use ``max(6, int(trade_max * OOS_BASELINE_MTM_STATE_HOLD_FRAC))`` (env OOS_MTM_STATE_HOLD_* is not used).
    """
    tmh = int(max(1, int(trade_max_hold)))
    frac = float(np.clip(float(OOS_BASELINE_MTM_STATE_HOLD_FRAC), 0.05, 0.95))
    return int(max(6, int(tmh * frac)))


def _oos_cumulative_plot_mode() -> str:
    """Which trades feed the price chart & legend metrics (default: all).

    - all: every closed trade (headline)
    - l3_learned: ``primary_exit_driver == l3_learned`` only (L3 exit classifier)
    - synthetic_rules: vol-regime / IC rule exits (see trade records)
    """
    v = (os.environ.get("OOS_CUMULATIVE_PLOT_MODE") or "all").strip().lower()
    if v in {"", "all", "headline", "total"}:
        return "all"
    if v in {"l3", "l3_learned", "learned", "model", "l3_exit"}:
        return "l3_learned"
    if v in {"synthetic", "synthetic_rules", "rules", "regime", "vol_regime"}:
        return "synthetic_rules"
    raise ValueError(
        "OOS_CUMULATIVE_PLOT_MODE must be one of: all, l3_learned, synthetic_rules"
    )


def _oos_stack_primary_exit_driver(
    *,
    synthetic: bool,
    l3_exit_mode: str,
    policy_exit: bool,
    flip_against: bool,
    short_vol_explosion: bool,
    deadline_exit: bool,
) -> str:
    """Label who effectively closed the trade (for PnL attribution, not double-counting)."""
    if synthetic:
        return "synthetic_rules"
    if short_vol_explosion:
        return "short_vol_explosion"
    if flip_against:
        return "signal_flip"
    if deadline_exit:
        return "deadline"
    if policy_exit:
        if l3_exit_mode == "l3":
            return "l3_learned"
        if l3_exit_mode == "mtm_adaptive":
            return "l3_mtm_adaptive"
        if l3_exit_mode in ("value_lt_zero", "value_lt_unreal"):
            return "l3_value_rule"
        if l3_exit_mode == "atr_trailing":
            return "atr_trailing"
    return "unknown"


def _ensure_primary_exit_driver_series(t: pd.DataFrame) -> pd.Series:
    """Back-compat for CSVs run before per-trade driver labels were added."""
    if t is None or t.empty:
        return pd.Series(dtype=object)
    if "primary_exit_driver" in t.columns:
        return t["primary_exit_driver"].astype("object")
    s = t.get("exit_reason", pd.Series([""] * len(t)))
    st = t.get("strategy_type", pd.Series([""] * len(t)))
    out: list[str] = []
    for i in range(len(t)):
        _st = str(st.iloc[i]) if i < len(st) else ""
        if any(
            _st.startswith(p)
            for p in (
                "SHORT_STRADDLE_",
                "LONG_STRADDLE_",
                "GAMMA_SCALP_",
                "IRON_CONDOR_",
                "IRON_BUTTERFLY_",
                "IRON_CONDOR_SHORT",
            )
        ):
            out.append("synthetic_rules")
        else:
            out.append("legacy_unlabeled")
    return pd.Series(out, index=t.index, dtype=object)


def build_backtest_interpretation(
    *,
    strategy_router: str,
    oos_l3_exit_mode: str,
    combined: pd.DataFrame,
) -> dict[str, Any]:
    """Explain what aggregate OOS numbers measure (esp. when router != plain straddle)."""
    t = combined if combined is not None else pd.DataFrame()
    drivers = _ensure_primary_exit_driver_series(t)
    n = int(len(t))
    share: dict[str, float | None] = {}
    if n > 0 and len(drivers) == n:
        vc = drivers.value_counts(normalize=True)
        for k in (
            "synthetic_rules",
            "l3_learned",
            "l3_mtm_adaptive",
            "l3_value_rule",
            "atr_trailing",
            "deadline",
            "signal_flip",
            "short_vol_explosion",
            "unknown",
            "legacy_unlabeled",
        ):
            share[k] = float(vc.get(k, 0.0)) if k in vc.index else 0.0
        for k, v in vc.items():
            sk = str(k)
            if sk not in share:
                share[sk] = float(v)
    router_lines = {
        "straddle": "Stack trade path: L2/L3 gating and L3 exit mode apply to the configured straddle simulation.",
        "regime_ic": "Iron-condor path in selected L1a regimes: exits follow synthetic IC rules; L3 may not drive exits unless configured.",
        "vol_regime": (
            "L1a regime → synthetic option templates. **Exits** use the L3 head with straddle-sim MTM% "
            "features (no template profit/stop/timed rules); max-hold is a hard backstop. "
            "Requires L3 straddle sim in meta."
        ),
        "l2_regime": (
            "L2 entry gate (l3_entry_side_from_l2) + L1a regime → same structure templates as vol_regime. "
            "**Exits** same as vol_regime: L3 with MTM%; max-hold backstop. Requires L3 straddle sim in meta."
        ),
    }
    l3m = (oos_l3_exit_mode or "l3").strip().lower()
    return {
        "strategy_router": str(strategy_router),
        "oos_l3_exit_mode": str(l3m),
        "headline_aggregates": "Sum of per-trade return fractions in exit_time order; not reinvested (simple cumulative).",
        "router_summary": router_lines.get(
            str(strategy_router),
            "See OOS_STRATEGY_ROUTER and runtime diagnostics for stack vs synthetic mix.",
        ),
        "n_trades": n,
        "share_of_trades_by_exit_driver": share,
        "reading_guide": [
            "metrics: headline blocks = all (post cap) closed trades",
            "metrics_by_exit_driver: same stats split by primary_exit_driver",
            "OOS_CUMULATIVE_PLOT_MODE=l3_learned: chart l3_learned + l3_mtm_adaptive (may be few when many exits are max-hold)",
        ],
    }


def _oos_l3_value_exit_min_hold() -> int:
    return int(np.clip(int((os.environ.get("OOS_L3_VALUE_EXIT_MIN_HOLD") or "3").strip() or "3"), 0, 10000))


def _oos_l3_value_exit_confirm_bars() -> int:
    return int(np.clip(int((os.environ.get("OOS_L3_VALUE_EXIT_CONFIRM_BARS") or "1").strip() or "1"), 1, 10000))


def _oos_current_unreal_for_value_exit(static: np.ndarray, _sidx: dict[str, int], *, straddle_mode: bool) -> float:
    if straddle_mode:
        return float(static[_sidx["l3_straddle_pnl_pct"]])
    return float(static[_sidx["l3_unreal_pnl_atr"]])


def _oos_atr_trailing_params() -> tuple[float, int]:
    """(mult, min_hold_bars). Straddle: exit when drawdown_from_peak >= mult * (ATR/close). Directional: mult in ATR units of unreal."""
    mult = float(np.clip(float(os.environ.get("OOS_ATR_TRAIL_MULT", "1.2")), 0.1, 20.0))
    min_h = int(np.clip(int(os.environ.get("OOS_ATR_TRAIL_MIN_HOLD", "1")), 1, 10000))
    return mult, min_h


def _parse_l1a_regime_id_csv(raw: str) -> set[int]:
    out: set[int] = set()
    for part in (raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        v = int(p)
        if v < 0 or v > 4:
            raise ValueError(
                f"L1a regime id must be 0..4, got {v!r} in comma list (5 vol-lifecycle classes)."
            )
        out.add(v)
    return out


def _oos_block_entry_l1a_regime_ids() -> set[int]:
    """L1a argmax ids **blocked** at entry. Default: ``OOS_BASELINE_BLOCK_ENTRY_L1A_REGIMES`` in this file.

    **Env overrides** (allow-list takes precedence over block-list):

    - ``OOS_ALLOW_ENTRY_L1A_REGIMES`` — comma allowlist, e.g. ``1`` = R1 only; ``0,1,3`` = R0/R1/R3, block R2 and R4;
      all other ``0..4`` not listed are blocked.
    - ``OOS_BLOCK_ENTRY_L1A_REGIMES`` — explicit block list, e.g. ``0,2,3,4`` (same as allow ``1`` only). Unused if
      ``OOS_ALLOW_ENTRY_L1A_REGIMES`` is set.
    """
    allow = (os.environ.get("OOS_ALLOW_ENTRY_L1A_REGIMES") or "").strip()
    if allow:
        a = _parse_l1a_regime_id_csv(allow)
        if not a:
            raise ValueError("OOS_ALLOW_ENTRY_L1A_REGIMES is set but empty after parsing")
        return {i for i in range(5) if i not in a}
    block = (os.environ.get("OOS_BLOCK_ENTRY_L1A_REGIMES") or "").strip()
    if block:
        return _parse_l1a_regime_id_csv(block)
    return set(OOS_BASELINE_BLOCK_ENTRY_L1A_REGIMES)


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
        return 14
    vals = sorted(vals)
    return int(vals[len(vals) // 2])


def _oos_strategy_router() -> str:
    """OOS path router: fixed to ``OOS_BASELINE_STRATEGY_ROUTER`` (straddle-only live baseline)."""
    return str(OOS_BASELINE_STRATEGY_ROUTER)


def _oos_l2l3_stack_preset_active() -> bool:
    """If true, main() forces straddle path + L3 exit head (L2 entry / L3 exit stack)."""
    v = (os.environ.get("OOS_L2L3_STACK") or os.environ.get("OOS_PURE_L2L3") or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on", "l2l3", "stack", "pure"}


def _oos_parse_hhmm_minutes(raw: str, *, default: str) -> int:
    text = (raw or default).strip() or default
    hh, mm = text.split(":", 1)
    return int(np.clip(int(hh), 0, 23)) * 60 + int(np.clip(int(mm), 0, 59))


def _oos_time_minutes(ts: Any) -> int:
    t = pd.Timestamp(ts)
    return int(t.hour) * 60 + int(t.minute)


def _oos_ic_regime_ids() -> set[int]:
    raw = (os.environ.get("OOS_IC_REGIME_IDS") or "0,3").strip()
    out: set[int] = set()
    for part in raw.split(","):
        p = part.strip()
        if p:
            out.add(int(p))
    return out


def _oos_ic_params() -> dict[str, float | int]:
    start_m = _oos_parse_hhmm_minutes(os.environ.get("OOS_IC_ENTRY_START", "10:45"), default="10:45")
    cutoff_m = _oos_parse_hhmm_minutes(os.environ.get("OOS_IC_ENTRY_CUTOFF", "15:00"), default="15:00")
    force_m = _oos_parse_hhmm_minutes(os.environ.get("OOS_IC_FORCE_EXIT", "15:30"), default="15:30")
    expiry_m = _oos_parse_hhmm_minutes(os.environ.get("OOS_IC_SYNTH_EXPIRY", "16:00"), default="16:00")
    return {
        "entry_start_min": int(start_m),
        "entry_cutoff_min": int(cutoff_m),
        "force_exit_min": int(force_m),
        "expiry_min": int(expiry_m),
        "short_delta": float(np.clip(float(os.environ.get("OOS_IC_SHORT_DELTA", "0.16")), 0.03, 0.45)),
        "wing_width_pct": float(np.clip(float(os.environ.get("OOS_IC_WING_WIDTH_PCT", "0.01")), 0.001, 0.20)),
        "min_wing_width": float(np.clip(float(os.environ.get("OOS_IC_MIN_WING_WIDTH", "0.25")), 0.01, 100.0)),
        "profit_take_frac": float(np.clip(float(os.environ.get("OOS_IC_PROFIT_TAKE_FRAC", "0.60")), 0.05, 0.99)),
        "stop_credit_mult": float(np.clip(float(os.environ.get("OOS_IC_STOP_CREDIT_MULT", "1.50")), 0.10, 20.0)),
        "stop_max_loss_frac": float(np.clip(float(os.environ.get("OOS_IC_STOP_MAX_LOSS_FRAC", "0.50")), 0.01, 1.0)),
    }


def _oos_vol_regime_strategy(regime_id: int) -> str:
    rid = int(regime_id)
    r0_mode = (os.environ.get("OOS_R0_STRATEGY") or "short_straddle").strip().lower()
    if rid == 0 and r0_mode in {"ibf", "iron_butterfly", "butterfly"}:
        return "IRON_BUTTERFLY_R0"
    return {
        0: "SHORT_STRADDLE_R0",
        1: "LONG_STRADDLE_R1",
        2: "GAMMA_SCALP_R2",
        3: "SHORT_STRADDLE_R3",
        4: "IRON_CONDOR_R4",
    }.get(rid, "LONG_STRADDLE_R1")


def _oos_strategy_time_params(strategy_type: str) -> dict[str, int]:
    defaults = {
        "SHORT_STRADDLE_R0": ("10:45", "15:00", "15:30", "16:00"),
        "IRON_BUTTERFLY_R0": ("10:45", "15:00", "15:30", "16:00"),
        "LONG_STRADDLE_R1": ("09:35", "15:30", "15:55", "16:00"),
        "GAMMA_SCALP_R2": ("09:35", "15:30", "15:55", "16:00"),
        "SHORT_STRADDLE_R3": ("10:30", "11:00", "15:30", "16:00"),
        "IRON_CONDOR_R4": ("10:45", "15:00", "15:30", "16:00"),
    }
    start_d, cutoff_d, force_d, expiry_d = defaults.get(strategy_type, defaults["LONG_STRADDLE_R1"])
    prefix = {
        "SHORT_STRADDLE_R0": "OOS_R0",
        "IRON_BUTTERFLY_R0": "OOS_R0",
        "LONG_STRADDLE_R1": "OOS_R1",
        "GAMMA_SCALP_R2": "OOS_R2",
        "SHORT_STRADDLE_R3": "OOS_R3",
        "IRON_CONDOR_R4": "OOS_R4",
    }.get(strategy_type, "OOS_R1")
    return {
        "entry_start_min": _oos_parse_hhmm_minutes(os.environ.get(f"{prefix}_ENTRY_START", start_d), default=start_d),
        "entry_cutoff_min": _oos_parse_hhmm_minutes(os.environ.get(f"{prefix}_ENTRY_CUTOFF", cutoff_d), default=cutoff_d),
        "force_exit_min": _oos_parse_hhmm_minutes(os.environ.get(f"{prefix}_FORCE_EXIT", force_d), default=force_d),
        "expiry_min": _oos_parse_hhmm_minutes(os.environ.get(f"{prefix}_SYNTH_EXPIRY", expiry_d), default=expiry_d),
    }


def _oos_synthetic_strategy_params(strategy_type: str) -> dict[str, float | int]:
    t = _oos_strategy_time_params(strategy_type)
    out: dict[str, float | int] = dict(t)
    _d_pt, _d_sc = {
        "SHORT_STRADDLE_R0": ("0.50", "0.80"),
        "IRON_BUTTERFLY_R0": ("0.50", "0.80"),
        "IRON_CONDOR_R4": ("0.50", "1.50"),
        "SHORT_STRADDLE_R3": ("0.40", "0.60"),
    }.get(str(strategy_type), ("0.80", "1.0"))
    out["stop_credit_mult"] = float(np.clip(float(os.environ.get(f"OOS_{strategy_type}_STOP_CREDIT_MULT", _d_sc)), 0.05, 20.0))
    out["profit_take_frac"] = float(np.clip(float(os.environ.get(f"OOS_{strategy_type}_PROFIT_TAKE_FRAC", _d_pt)), 0.05, 0.99))
    out["no_move_timeout"] = int(np.clip(int(os.environ.get(f"OOS_{strategy_type}_NO_MOVE_TIMEOUT", "120")), 1, 390))
    out["no_move_frac"] = float(np.clip(float(os.environ.get(f"OOS_{strategy_type}_NO_MOVE_FRAC", "0.0015")), 0.0001, 0.05))
    out["hedge_move_frac"] = float(np.clip(float(os.environ.get(f"OOS_{strategy_type}_HEDGE_MOVE_FRAC", "0.0015")), 0.0001, 0.05))
    out["ic_short_delta"] = float(np.clip(float(os.environ.get(f"OOS_{strategy_type}_SHORT_DELTA", "0.16")), 0.03, 0.45))
    out["ic_wing_width_pct"] = float(np.clip(float(os.environ.get(f"OOS_{strategy_type}_WING_WIDTH_PCT", "0.01")), 0.001, 0.20))
    out["ic_min_wing_width"] = float(np.clip(float(os.environ.get(f"OOS_{strategy_type}_MIN_WING_WIDTH", "0.25")), 0.01, 100.0))
    out["ic_stop_max_loss_frac"] = float(np.clip(float(os.environ.get(f"OOS_{strategy_type}_STOP_MAX_LOSS_FRAC", "0.50")), 0.01, 1.0))
    return out


def _oos_env_yn(name: str, default: bool) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default


def _oos_is_synthetic_strategy(strategy_type: str) -> bool:
    return strategy_type in {
        "SHORT_STRADDLE_R0",
        "IRON_BUTTERFLY_R0",
        "LONG_STRADDLE_R1",
        "GAMMA_SCALP_R2",
        "SHORT_STRADDLE_R3",
        "IRON_CONDOR_R4",
        "IRON_CONDOR_SHORT",
    }


def _oos_regime_stack_router(strategy_router: str) -> bool:
    """vol_regime / l2_regime: 合成结构 + MTM 只喂 L3 平仓；无模板止盈/止损/到点强平。"""
    return str(strategy_router) in ("vol_regime", "l2_regime")


def _oos_trade_size_mult(entry_regime_id: int, l2_conf: float) -> float:
    """Scale linear return (plan layer 3). Env OOS_SIZE_MULT_R0..R4, OOS_SIZE_USE_SIGNAL."""
    r = int(np.clip(int(entry_regime_id), 0, 4))
    ddef = {0: 0.7, 1: 1.0, 2: 1.0, 3: 0.8, 4: 0.6}.get(r, 1.0)
    b = float(os.environ.get(f"OOS_SIZE_MULT_R{r}", str(ddef)))
    b = float(np.clip(b, 0.01, 10.0))
    if not _oos_env_yn("OOS_SIZE_USE_SIGNAL", True):
        return b
    c = float(l2_conf)
    if c > 0.6:
        g = 1.0
    elif c >= 0.5:
        g = 0.7
    else:
        g = 0.5
    return float(b * g)


def _prepare_symbol_df(symbol: str) -> pd.DataFrame:
    from core.training.prep.data_prep import ensure_breakout_features, ensure_structure_context_features

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


def _l3_frac_path_feats_from_hist(hist: list[float]) -> dict[str, float]:
    """Match L3 policy-dataset fractional path features (train.py _build_l3_policy_dataset)."""
    if not hist:
        z = 0.0
        return {
            "l3_unreal_pnl_frac": z,
            "l3_drawdown_from_peak_frac": z,
            "l3_ret_last_3_frac": z,
            "l3_ret_last_5_frac": z,
            "l3_volatility_in_trade_frac": z,
            "l3_trend_slope_frac": z,
        }
    cur_f = float(hist[-1])
    n = len(hist)
    peak_f = float(max(hist))
    dd_f = peak_f - cur_f
    r3 = cur_f - float(hist[max(0, n - 4)])
    r5 = cur_f - float(hist[max(0, n - 6)])
    arr = np.asarray(hist, dtype=np.float64)
    d = np.diff(arr, prepend=arr[0])
    vol_tf = float(np.std(d)) if d.size > 1 else 0.0
    t_idx = np.arange(n, dtype=np.float64)
    if n >= 2:
        xm = float(np.mean(t_idx))
        ym = float(np.mean(arr))
        num = float(np.sum((t_idx - xm) * (arr - ym)))
        den = float(np.sum((t_idx - xm) ** 2))
        slope = float(num / den) if den > 1e-18 else 0.0
    else:
        slope = 0.0
    return {
        "l3_unreal_pnl_frac": cur_f,
        "l3_drawdown_from_peak_frac": dd_f,
        "l3_ret_last_3_frac": r3,
        "l3_ret_last_5_frac": r5,
        "l3_volatility_in_trade_frac": vol_tf,
        "l3_trend_slope_frac": slope,
    }


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
) -> tuple[np.ndarray, float, float]:
    from core.training.unified.policy_data import (
        L3_MOMENTUM_LEADING_FEATURE_NAMES,
        _l3_episode_momentum_leading_block,
        l3_meta_max_hold_bars,
    )

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
        _ov = l3_aux.get("oos_synth_mtm_pnl_pct")
        if _ov is not None and np.isfinite(float(_ov)):
            pnl_pct = float(_ov)
        peak_in = float(peak_unreal)
        peak_unreal = max(peak_in, pnl_pct)
        drawdown_from_peak = float(peak_unreal - pnl_pct)
        running_mfe = float(max(float(l3_aux.get("running_mfe", 0.0)), max(pnl_pct, 0.0)))
        running_mae = float(max(float(l3_aux.get("running_mae", 0.0)), max(-pnl_pct, 0.0)))
        l3_aux["running_mfe"] = running_mfe
        l3_aux["running_mae"] = running_mae
        live_edge = float(running_mfe - running_mae)
        entry_regime = l2_out.loc[entry_idx, [f"l2_entry_regime_{i}" for i in range(len(L1A_REGIME_COLS))]].to_numpy(
            dtype=np.float32
        )
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
        sig_e = float(l2_out.loc[entry_idx].get("l2_vol_signal", 1.0 if float(l2_out.loc[entry_idx].get("l2_straddle_on", 0)) > 0.5 else 0.0))
        sig_c = float(l2_out.loc[idx].get("l2_vol_signal", 1.0 if float(l2_out.loc[idx].get("l2_straddle_on", 0)) > 0.5 else 0.0))
        signal_direction_agree = float((sig_e > 0 and sig_c > 0) or (sig_e < 0 and sig_c < 0))
        rid_e = int(np.argmax(entry_regime.astype(np.float64)))
        regime_changed = float(int(np.argmax(current_regime.astype(np.float64)) != rid_e))
        gate_e = float(np.clip(float(l2_out.loc[entry_idx, "l2_gate_prob"]), 0.0, 1.0)) if "l2_gate_prob" in l2_out.columns else 0.0
        gate_curr = float(np.clip(float(l2_out.loc[idx, "l2_gate_prob"]), 0.0, 1.0)) if "l2_gate_prob" in l2_out.columns else 0.0
        gate_decay = float(gate_curr - gate_e)
        regime_probs_i = l2_out.loc[idx, [f"l2_entry_regime_{k}" for k in range(len(L1A_REGIME_COLS))]].to_numpy(dtype=np.float32)
        entry_vol_i = float(l2_out.loc[idx, "l2_entry_vol"])
        pa_row = df.loc[idx, PA_STATE_FEATURES] if all(c in df.columns for c in PA_STATE_FEATURES) else None
        min_c, min_sz, _, _ = l3_entry_policy_params(regime_probs_i, entry_vol_i, l3_meta, pa_state=pa_row)
        if "l2_decision_class" in l2_out.columns:
            cls_i = int(np.clip(int(l2_out.loc[idx, "l2_decision_class"]), 0, 2))
        else:
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
        w_fav = 0.28
        w_adv = -0.35
        w_reg = -0.45
        w_gate = -0.18
        w_gthr = -0.12
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
        trade_quality_bayes = _sigmoid_float(lo)
        hist = l3_aux.get("pnl_hist")
        if not isinstance(hist, list):
            hist = []
        hist.append(float(pnl_pct))
        if len(hist) > 64:
            hist = hist[-64:]
        l3_aux["pnl_hist"] = hist
        br_step = float(df["close"].iloc[idx] / max(abs(float(entry_underlying)), 1e-9) - 1.0)
        uh = l3_aux.get("under_ret_hist")
        if not isinstance(uh, list):
            uh = []
        uh.append(br_step)
        if len(uh) > 64:
            uh = uh[-64:]
        l3_aux["under_ret_hist"] = uh
        frac_path_feats = _l3_frac_path_feats_from_hist(uh)
        for _k, _v in (
            ("trail_u", pnl_pct),
            ("trail_dd", drawdown_from_peak),
            ("trail_mfe", running_mfe),
            ("trail_mae", running_mae),
            ("trail_vs", vol_surprise),
        ):
            l3_aux.setdefault(_k, []).append(float(_v))
            if len(l3_aux[_k]) > 64:
                l3_aux[_k] = l3_aux[_k][-64:]
        _nt = len(l3_aux["trail_u"])
        _mb = _l3_episode_momentum_leading_block(
            _nt,
            np.asarray(l3_aux["trail_u"], dtype=np.float64),
            np.asarray(l3_aux["trail_dd"], dtype=np.float64),
            np.asarray(l3_aux["trail_mfe"], dtype=np.float64),
            np.asarray(l3_aux["trail_mae"], dtype=np.float64),
            np.asarray(l3_aux["trail_vs"], dtype=np.float64),
            np.asarray(l3_aux["under_ret_hist"], dtype=np.float64),
            np.arange(1, _nt + 1, dtype=np.float32),
            l3_meta_max_hold_bars(l3_meta),
            float(l2_out.loc[entry_idx, "l2_pred_mfe"]),
        )
        mom_feats = {L3_MOMENTUM_LEADING_FEATURE_NAMES[i]: float(_mb[-1, i]) for i in range(len(L3_MOMENTUM_LEADING_FEATURE_NAMES))}
        implied_entry_atr = float(l2_out.loc[entry_idx].get("l2_implied_proxy_range", l2_out.loc[entry_idx].get("l2_range_pred", 1.0)))
        implied_entry_frac = float(max(implied_entry_atr * max(entry_atr, 1e-6) / max(abs(entry_underlying), 1e-9), 1e-6))
        range_realization_ratio = float(abs(br_step) / implied_entry_frac)
        theta_burn_fraction = float(np.sqrt(max(float(hold), 0.0) / max(float(l3_meta_max_hold_bars(l3_meta)), 1.0)))
        if "vixy_zscore_390" in df.columns:
            vz_now = float(pd.to_numeric(df.loc[idx, "vixy_zscore_390"], errors="coerce") or 0.0)
            if "vixy_z_entry" not in l3_aux:
                l3_aux["vixy_z_entry"] = float(pd.to_numeric(df.loc[entry_idx, "vixy_zscore_390"], errors="coerce") or 0.0)
            vixy_change_since_entry = float(vz_now - float(l3_aux.get("vixy_z_entry", 0.0)))
        else:
            vixy_change_since_entry = 0.0
        range_expansion_speed = float(abs(br_step) / max(float(hold), 1.0))
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
            "l3_side": float(in_pos),
            "l3_log_hold_bars": float(log_h),
            "l3_hold_bars_sq": float(h_sq),
            "l3_hold_bucket": float(h_bkt),
            "l3_drawdown_from_peak_atr": float(drawdown_from_peak),
            **frac_path_feats,
            **mom_feats,
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
            "l3_range_realization_ratio": float(range_realization_ratio),
            "l3_theta_burn_fraction": float(theta_burn_fraction),
            "l3_vixy_change_since_entry": float(vixy_change_since_entry),
            "l3_range_expansion_speed": float(range_expansion_speed),
            **(
                {
                    "l3_hold_bars_x_unreal_pnl_atr": float(hold) * float(pnl_pct),
                    "l3_hold_bars_x_price_velocity_3bar_atr": float(hold) * float(vel3),
                }
                if _l3_oos_exit_hold_interactions()
                else {}
            ),
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
        feat = np.asarray(
            [0.0 if c in cox_names else float(vals.get(c, 0.0)) for c in static_names],
            dtype=np.float32,
        )
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
    entry_regime = l2_out.loc[entry_idx, [f"l2_entry_regime_{i}" for i in range(len(L1A_REGIME_COLS))]].to_numpy(
        dtype=np.float32
    )
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

    w_fav = 0.28
    w_adv = -0.35
    w_reg = -0.45
    w_gate = -0.18
    w_gthr = -0.12
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
    trade_quality_bayes = _sigmoid_float(lo)
    ep_ff = max(float(entry_price), 1e-9)
    cl_px = float(df["close"].iloc[idx])
    unreal_frac = float((cl_px - ep_ff) / ep_ff) if float(in_pos) > 0 else float((ep_ff - cl_px) / ep_ff)
    hist_ff = l3_aux.get("bar_ret_frac_hist")
    if not isinstance(hist_ff, list):
        hist_ff = []
    hist_ff.append(unreal_frac)
    if len(hist_ff) > 64:
        hist_ff = hist_ff[-64:]
    l3_aux["bar_ret_frac_hist"] = hist_ff
    frac_path_feats = _l3_frac_path_feats_from_hist(hist_ff)
    for _k, _v in (
        ("trail_u", u),
        ("trail_dd", drawdown_from_peak),
        ("trail_mfe", live_mfe),
        ("trail_mae", live_mae),
        ("trail_vs", vol_surprise),
    ):
        l3_aux.setdefault(_k, []).append(float(_v))
        if len(l3_aux[_k]) > 64:
            l3_aux[_k] = l3_aux[_k][-64:]
    _nt = len(l3_aux["trail_u"])
    _mb = _l3_episode_momentum_leading_block(
        _nt,
        np.asarray(l3_aux["trail_u"], dtype=np.float64),
        np.asarray(l3_aux["trail_dd"], dtype=np.float64),
        np.asarray(l3_aux["trail_mfe"], dtype=np.float64),
        np.asarray(l3_aux["trail_mae"], dtype=np.float64),
        np.asarray(l3_aux["trail_vs"], dtype=np.float64),
        np.asarray(hist_ff, dtype=np.float64),
        np.arange(1, _nt + 1, dtype=np.float32),
        l3_meta_max_hold_bars(l3_meta),
        float(l2_out.loc[entry_idx, "l2_pred_mfe"]),
    )
    mom_feats = {L3_MOMENTUM_LEADING_FEATURE_NAMES[i]: float(_mb[-1, i]) for i in range(len(L3_MOMENTUM_LEADING_FEATURE_NAMES))}

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
        **frac_path_feats,
        **mom_feats,
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
    if _l3_oos_exit_hold_interactions():
        vals["l3_hold_bars_x_unreal_pnl_atr"] = float(hold) * float(unreal)
        vals["l3_hold_bars_x_price_velocity_3bar_atr"] = float(hold) * float(vel3)
    feature_cols = list(l3_meta["feature_cols"])
    static_names = [c for c in feature_cols if not c.startswith("l3_traj_emb_")]
    cox_names = {"l3_cox_log_partial_hazard", "l3_cox_baseline_cumhaz_at_stop"}
    feat = np.asarray(
        [0.0 if c in cox_names else float(vals.get(c, 0.0)) for c in static_names],
        dtype=np.float32,
    )
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
            "Train the new stack first, e.g. `./scripts/training/run_train.sh layer1`."
        )


def _oos_deadline_pnl_atr(
    *,
    straddle_mode: bool,
    df: pd.DataFrame,
    l3_aux_state: dict[str, Any],
    l3_meta: dict[str, Any],
    entry_idx: int,
    entry_price: float,
    entry_atr: float,
    in_pos: int,
    trade_max_hold: int,
) -> float:
    """Counterfactual MTM at the policy horizon bar (entry + max_hold), same units as ``l3_unreal_pnl_atr`` in OOS.

    Directional: unrealized return vs entry in **entry ATR** units (``_live_trade_state_from_bar``).
    Straddle: **premium fraction** (model mark vs entry premium), matching straddle ``l3_unreal_pnl_atr`` in features.
    """
    n = int(len(df))
    dj = int(min(max(int(entry_idx) + int(trade_max_hold), int(entry_idx)), n - 1))
    if straddle_mode:
        if StraddleSimulator is None:
            raise RuntimeError("StraddleSimulator not loaded; call _lazy_import_stack_modules() before OOS.")
        dte_days = int(l3_aux_state.get("dte_days", _oos_default_straddle_dte_days(l3_meta)))
        strike = float(l3_aux_state.get("strike", float(df["close"].iloc[dj])))
        entry_iv = float(
            l3_aux_state.get(
                "entry_iv",
                float(df["l3_base_iv"].iloc[min(dj, n - 1)]) if "l3_base_iv" in df.columns else 0.25,
            )
        )
        hold_d = max(0, dj - int(entry_idx))
        current_iv_base = float(df.loc[dj, "l3_base_iv"]) if "l3_base_iv" in df.columns else entry_iv
        current_iv = float(np.clip(0.65 * entry_iv + 0.35 * current_iv_base, 0.05, 5.0))
        t_remaining = max(dte_days / 252.0 - hold_d / (390.0 * 252.0), 1e-9)
        sim = StraddleSimulator(risk_free_rate=float(l3_aux_state.get("risk_free_rate", 0.04)))
        greeks = sim.straddle_greeks(float(df["close"].iloc[dj]), strike, t_remaining, current_iv)
        return float((greeks.price - float(entry_price)) / max(float(entry_price), 1e-9))
    _, _, u = _live_trade_state_from_bar(
        side=float(in_pos),
        entry_price=float(entry_price),
        atr=float(entry_atr),
        high_price=float(df["high"].iloc[dj]),
        low_price=float(df["low"].iloc[dj]),
        close_price=float(df["close"].iloc[dj]),
    )
    return float(u)


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
    _vfc = list(l3_meta.get("l3_value_feature_columns") or [])
    if not _vfc:
        _vfc = list(feature_cols)  # legacy: value head used full LGBM column set
    try:
        _value_feat_ix = [feature_cols.index(c) for c in _vfc]
    except ValueError as ex:
        raise RuntimeError(
            f"L3 value feature / exit feature_cols mismatch: {ex!r}. Retrain L3 or align l3_value_feature_columns."
        ) from ex
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
    from core.training.unified.policy_data import l3_meta_max_hold_bars as _l3_meta_mh

    _sm = l3_meta.get("l3_straddle_max_hold_minutes")
    if _l3_straddle_sim_mode_enabled(l3_meta):
        max_hold = int(_sm) if _sm is not None else int(_l3_meta_mh(l3_meta))
    else:
        max_hold = int(_l3_meta_mh(l3_meta))
    l2_abstain_margin = float((l2_meta.get("two_stage_policy") or l2_meta).get("direction_abstain_margin", 0.0))
    slip_f, comm_f = _oos_cost_fracs()
    l3_exit_mode = _oos_l3_exit_mode()
    adaptive_mtm_cfg = None
    adapt_mtm_eval = None
    adapt_mtm_trim = None
    if l3_exit_mode == "mtm_adaptive":
        from core.training.unified.adaptive_mtm_exit import (
            evaluate_adaptive_mtm_exit as _adapt_mtm_eval,
            load_adaptive_mtm_config_from_env,
            trim_histories_in_place as _adapt_mtm_trim,
        )

        adaptive_mtm_cfg = load_adaptive_mtm_config_from_env()
        adapt_mtm_eval = _adapt_mtm_eval
        adapt_mtm_trim = _adapt_mtm_trim
    atr_trail_mult, atr_trail_min_hold = _oos_atr_trailing_params()
    value_exit_min_hold = _oos_l3_value_exit_min_hold()
    value_exit_confirm_bars = _oos_l3_value_exit_confirm_bars()
    value_exit_unreal_frac = _oos_l3_value_exit_unreal_frac()
    block_entry_l1a_regimes = _oos_block_entry_l1a_regime_ids()
    strategy_router = _oos_strategy_router()
    if _oos_regime_stack_router(str(strategy_router)) and not _l3_straddle_sim_mode_enabled(l3_meta):
        _sem = l3_meta.get("l3_trade_semantics", "<missing>")
        raise RuntimeError(
            "OOS vol_regime|l2_regime needs L3 straddle simulation: synthetic MTM feeds l3_straddle_* features. "
            f"Current l3_meta l3_trade_semantics={_sem!r}. Fix one of: "
            'set "l3_trade_semantics" to "straddle_bs_sim" in the L3 meta JSON; '
            "or if feature_cols already include l3_straddle_pnl_pct, run with OOS_L3_STRADDLE_SIM=1; "
            "or if l3_trade_semantics is absent from meta, set L3_STRADDLE_SIM_MODE=1."
        )
    ic_regime_ids = _oos_ic_regime_ids()
    ic_params = _oos_ic_params()
    straddle_oos = _l3_straddle_sim_mode_enabled(l3_meta)
    _l2_br = (
        "l2_decision_class"
        if "l2_decision_class" in l2_out.columns
        else ("l2_straddle_on" if "l2_straddle_on" in l2_out.columns else "none")
    )
    print(
        f"  [{symbol}] exit={l3_exit_mode}  ATR_trail_mult={atr_trail_mult}  ATR_min_hold={atr_trail_min_hold}  "
        f"value_min_hold={value_exit_min_hold}  value_confirm={value_exit_confirm_bars}  "
        f"value_unreal_frac={value_exit_unreal_frac:.3f}  "
        f"strategy_router={strategy_router}  ic_regimes={sorted(ic_regime_ids)}  "
        f"block_L1a_regimes={sorted(block_entry_l1a_regimes) if block_entry_l1a_regimes else '—'}",
        flush=True,
    )
    if l3_exit_mode == "mtm_adaptive" and adaptive_mtm_cfg is not None:
        c = adaptive_mtm_cfg
        _ref_prev = _oos_mtm_state_hold_ref_for_adaptive(trade_max_hold=int(max_hold))
        print(
            f"  [{symbol}] MTM adaptive exit: fast={c.mtm_fast_window}  slow={c.mtm_slow_window}  "
            f"min_hold={c.min_hold_bars}  vote_k={c.vote_k}  z={c.z_score_hi:.2f}  "
            f"mtm_state_hold_ref={_ref_prev} (eff_min uses max(regime_min, this); "
            f"OOS_BASELINE_MTM_STATE_HOLD_FRAC={OOS_BASELINE_MTM_STATE_HOLD_FRAC:g}; not L3 policy state_hold_ref)",
            flush=True,
        )
    print(
        f"  [{symbol}] straddle_oos={straddle_oos}  l2_row_signal={_l2_br}  — "
        f"if l2_row_signal=l2_decision_class and not straddle_oos, L2 can close via signal_flip before L3 prob crosses threshold",
        flush=True,
    )
    trades: list[dict[str, object]] = []
    exit_infer_state = L3ExitInferenceState()
    peak_unreal_atr = float("-inf")
    l3_aux_state: dict[str, float | int] = {}
    runtime_diag: dict[str, Any] = {
        "symbol": symbol,
        "l3_exit_mode": l3_exit_mode,
        "entry_blocked_l1a_regime": 0,
        "entries": 0,
        "policy_exit_count": 0,
        "flip_exit_count": 0,
        "atr_trail_exit_count": 0,
        "deadline_exit_count": 0,
        "value_exit_signal_count": 0,
        "value_exit_confirmed_count": 0,
        "ic_entries": 0,
        "ic_time_window_skips": 0,
        "ic_profit_take_exit_count": 0,
        "ic_stop_exit_count": 0,
        "ic_forced_exit_count": 0,
        "ibf_entries": 0,
        "synthetic_entries": 0,
        "synthetic_time_window_skips": 0,
        "synthetic_exit_counts": {},
        "synthetic_rule_exit_count": 0,
        "non_synthetic_exit_count": 0,
        "learned_l3_exit_count": 0,
        "adaptive_mtm_exit_count": 0,
        "soft_enter_hits_early": 0,
        "soft_enter_hits_late": 0,
        "steps_early": 0,
        "steps_late": 0,
        "soft_enter_thr_early_sum": 0.0,
        "soft_enter_thr_late_sum": 0.0,
        "bar_exit_records": [],
        "_exit_prob_hold_samples": [],
        "_signal_flip_at_exit_probs": [],
    }

    _bar_track = default_bar_exit_track_bars()
    bar_exit_diag: BarExitDiagnostics | None = None if straddle_oos else BarExitDiagnostics(_bar_track)
    bar_exit_records: list[dict[str, Any]] = []
    _all_closes = df["close"].to_numpy(dtype=np.float64, copy=False)

    in_pos = 0
    entry_idx = -1
    entry_price = 0.0
    entry_atr = 1e-3
    entry_time = None
    current_strategy_type = ""
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

    global _OOS_ENTRY_DIAG_PRINTED, _OOS_L3_ENTRY_FUNC_SOURCE_PRINTED, _OOS_L3_ENTRY_POLICY_SOURCE_PRINTED
    if not _OOS_ENTRY_DIAG_PRINTED and n_bars > 0:
        _OOS_ENTRY_DIAG_PRINTED = True
        _sl = l2_out.iloc[:n_bars]
        _diag_cls = _sl.get("l2_decision_class", pd.Series(dtype=float))
        _diag_strad = _sl.get("l2_straddle_on", pd.Series(dtype=float))
        _diag_conf = _sl.get("l2_decision_confidence", pd.Series(dtype=float))
        _diag_size = _sl.get("l2_size", pd.Series(dtype=float))
        entry_min_conf = float(l3_meta.get("l3_entry_min_confidence", 0.0) or 0.0)
        entry_min_size = float(l3_meta.get("l3_entry_min_size", 0.0) or 0.0)
        print(f"[ENTRY DIAG] symbol={symbol}  sim bars: {n_bars}", flush=True)
        print(f"[ENTRY DIAG] total l2_out rows: {len(l2_out)}", flush=True)
        print(f"[ENTRY DIAG] l2_decision_class distribution (first {n_bars} bars):", flush=True)
        if len(_diag_cls) and _diag_cls.notna().any():
            print(_diag_cls.value_counts().sort_index().to_string(), flush=True)
        else:
            print("  column missing or all NaN", flush=True)
        if len(_diag_strad) and _diag_strad.notna().any():
            print(f"[ENTRY DIAG] l2_straddle_on > 0.5: {int((_diag_strad > 0.5).sum())}", flush=True)
        else:
            print("[ENTRY DIAG] l2_straddle_on: (missing/empty)", flush=True)
        if len(_diag_conf) and _diag_conf.notna().any():
            print(
                f"[ENTRY DIAG] l2_decision_confidence: "
                f"min={float(_diag_conf.min()):.4f} median={float(_diag_conf.median()):.4f} "
                f"max={float(_diag_conf.max()):.4f}",
                flush=True,
            )
        else:
            print("[ENTRY DIAG] l2_decision_confidence: (missing/empty)", flush=True)
        if len(_diag_size) and _diag_size.notna().any():
            print(
                f"[ENTRY DIAG] l2_size: "
                f"min={float(_diag_size.min()):.4f} median={float(_diag_size.median()):.4f} "
                f"max={float(_diag_size.max()):.4f}",
                flush=True,
            )
        else:
            print("[ENTRY DIAG] l2_size: (missing/empty)", flush=True)
        print(
            f"[ENTRY DIAG] meta l3_entry_min_confidence={entry_min_conf:.4g}  "
            f"l3_entry_min_size={entry_min_size:.4g}  (per-bar policy may differ)",
            flush=True,
        )
        if len(_diag_cls) and _diag_cls.notna().any():
            _n_cls_pass = int((((_diag_cls == 0) | (_diag_cls == 2)) & _diag_cls.notna()).sum())
        else:
            _n_cls_pass = 0
        if len(_diag_strad) and _diag_strad.notna().any():
            _n_strad_pass = int((_diag_strad > 0.5).sum())
        else:
            _n_strad_pass = 0
        if len(_diag_conf) and _diag_conf.notna().any():
            _n_conf_pass = int((_diag_conf >= entry_min_conf).sum())
        else:
            _n_conf_pass = 0
        if len(_diag_size) and _diag_size.notna().any():
            _n_size_pass = int((_diag_size >= entry_min_size).sum())
        else:
            _n_size_pass = 0
        print(f"[ENTRY DIAG] cls in {{0,2}} (would enter on class): {_n_cls_pass}", flush=True)
        print(f"[ENTRY DIAG] straddle_on>0.5:                {_n_strad_pass}", flush=True)
        print(f"[ENTRY DIAG] conf >= meta min ({entry_min_conf:.4g}):     {_n_conf_pass}", flush=True)
        print(f"[ENTRY DIAG] size >= meta min ({entry_min_size:.4g}):     {_n_size_pass}", flush=True)
        if len(_diag_cls) and len(_diag_conf) and len(_diag_size):
            _m = _diag_cls.notna() & _diag_conf.notna() & _diag_size.notna()
            _joint_meta = _m & (
                (_diag_cls == 0) | (_diag_cls == 2)
            ) & (_diag_conf >= entry_min_conf) & (_diag_size >= entry_min_size)
            print(
                f"[ENTRY DIAG] joint (cls!=1 + conf + size) using meta min only: {int(_joint_meta.sum())}",
                flush=True,
            )
        _joint_side = 0
        for j in range(n_bars):
            if "l2_decision_class" in l2_out.columns:
                _cj = int(np.clip(int(l2_out.loc[j, "l2_decision_class"]), 0, 2))
            else:
                _cj = 0 if float(l2_out.loc[j].get("l2_straddle_on", 0)) > 0.5 else 1
            _reg_j = l2_out.loc[j, [f"l2_entry_regime_{ix}" for ix in range(len(L1A_REGIME_COLS))]].to_numpy(
                dtype=np.float32
            )
            _ev_j = float(l2_out.loc[j, "l2_entry_vol"])
            _pa_j = df.loc[j, PA_STATE_FEATURES] if all(c in df.columns for c in PA_STATE_FEATURES) else None
            _emc, _ems, _, _ = l3_entry_policy_params(_reg_j, _ev_j, l3_meta, pa_state=_pa_j)
            _side_j = l3_entry_side_from_l2(
                _cj,
                float(l2_out.loc[j, "l2_decision_confidence"]),
                float(l2_out.loc[j, "l2_size"]),
                min_confidence=_emc,
                min_size=_ems,
            )
            if _side_j != 0.0:
                _joint_side += 1
        print(
            f"[ENTRY DIAG] joint pass (per-bar l3 policy + l3_entry_side_from_l2, excl. L1a block): {_joint_side}",
            flush=True,
        )

    if not _OOS_L3_ENTRY_FUNC_SOURCE_PRINTED and l3_entry_side_from_l2 is not None:
        _OOS_L3_ENTRY_FUNC_SOURCE_PRINTED = True
        try:
            print("[FUNC DIAG] l3_entry_side_from_l2:\n" + inspect.getsource(l3_entry_side_from_l2), flush=True)
        except (OSError, TypeError) as ex:
            print(f"[FUNC DIAG] getsource failed: {ex!r}", flush=True)

    _loop_flat = 0
    _loop_cls_ok = 0
    _loop_side_nonzero = 0
    _loop_trade_opened = 0
    _loop_l1a_blocked = 0
    _loop_first_side_zero: list[dict[str, object]] = []
    _loop_exit_check = 0
    _loop_exit_probe = 0

    _oos_exit_dist_log = (os.environ.get("OOS_EXIT_DIST_LOG", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    _oos_route_ablation = (os.environ.get("OOS_EXIT_ROUTE_ABLATION", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    oos_exit_agg_date: object | None = None
    oos_exit_agg_probs: list[float] = []
    oos_exit_last_feat: np.ndarray | None = None

    def _oos_diag_to_file(msg: str) -> None:
        print(msg, flush=True)
        try:
            from core.training.logging.metrics_file_log import append_oos_unified_log

            append_oos_unified_log(msg)
        except Exception:  # noqa: BLE001
            pass

    for i in bar_iter:
        if in_pos == 0:
            if "l2_decision_class" in l2_out.columns:
                cls = int(np.clip(int(l2_out.loc[i, "l2_decision_class"]), 0, 2))
            else:
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
            _loop_flat += 1
            if cls != 1:
                _loop_cls_ok += 1
            _side_pre_l1a = float(side)
            if _side_pre_l1a != 0.0:
                _loop_side_nonzero += 1
            elif cls != 1 and len(_loop_first_side_zero) < 5:
                _loop_first_side_zero.append(
                    {
                        "bar": int(i),
                        "cls": int(cls),
                        "conf": round(float(conf), 4),
                        "size": round(float(size), 4),
                        "entry_min_conf": round(float(entry_min_conf), 4),
                        "entry_min_size": round(float(entry_min_size), 4),
                        "side_pre_l1a": float(_side_pre_l1a),
                        "conf_ok": bool(float(conf) >= float(entry_min_conf)),
                        "size_ok": bool(float(size) >= float(entry_min_size)),
                    }
                )
            l1a_entry_regime = int(
                np.argmax(l1a_out.loc[i, L1A_REGIME_COLS].to_numpy(dtype=np.float32))
            )
            if side != 0.0 and l1a_entry_regime in block_entry_l1a_regimes:
                _loop_l1a_blocked += 1
                runtime_diag["entry_blocked_l1a_regime"] = int(runtime_diag["entry_blocked_l1a_regime"]) + 1
                side = 0.0
            if side != 0.0:
                entry_fill_time = df["time_key"].iloc[i + 1]
                entry_fill_min = _oos_time_minutes(entry_fill_time)
                strategy_type = "STRADDLE"
                strategy_params: dict[str, float | int] = {}
                if strategy_router == "regime_ic" and int(l1a_entry_regime) in ic_regime_ids:
                    if (
                        entry_fill_min < int(ic_params["entry_start_min"])
                        or entry_fill_min >= int(ic_params["entry_cutoff_min"])
                    ):
                        runtime_diag["ic_time_window_skips"] = int(runtime_diag["ic_time_window_skips"]) + 1
                        side = 0.0
                    else:
                        strategy_type = "IRON_CONDOR_SHORT"
                        side = -1.0
                        strategy_params = dict(ic_params)
                elif strategy_router in ("vol_regime", "l2_regime"):
                    strategy_type = _oos_vol_regime_strategy(int(l1a_entry_regime))
                    strategy_params = _oos_synthetic_strategy_params(strategy_type)
                    if (
                        entry_fill_min < int(strategy_params["entry_start_min"])
                        or entry_fill_min >= int(strategy_params["entry_cutoff_min"])
                    ):
                        runtime_diag["synthetic_time_window_skips"] = int(runtime_diag["synthetic_time_window_skips"]) + 1
                        side = 0.0
                    elif strategy_type in {"SHORT_STRADDLE_R0", "SHORT_STRADDLE_R3", "IRON_CONDOR_R4", "IRON_BUTTERFLY_R0"}:
                        side = -1.0
                    else:
                        side = 1.0
                if side == 0.0:
                    continue
                _loop_trade_opened += 1
                runtime_diag["entries"] = int(runtime_diag["entries"]) + 1
                if _oos_is_synthetic_strategy(strategy_type):
                    runtime_diag["synthetic_entries"] = int(runtime_diag["synthetic_entries"]) + 1
                if strategy_type in {"IRON_CONDOR_SHORT", "IRON_CONDOR_R4"}:
                    runtime_diag["ic_entries"] = int(runtime_diag["ic_entries"]) + 1
                if strategy_type == "IRON_BUTTERFLY_R0":
                    runtime_diag["ibf_entries"] = int(runtime_diag.get("ibf_entries", 0)) + 1
                straddle_sim_mode = _l3_straddle_sim_mode_enabled(l3_meta)
                in_pos = int(side)
                entry_idx = i
                entry_open = float(df["open"].iloc[i + 1])
                exit_infer_state.reset()
                p0 = float(np.clip(conf, 0.05, 0.95))
                l3_aux_state.clear()
                l3_aux_state["trade_max_hold"] = int(max_hold)
                l3_aux_state["trade_exit_prob_max"] = None
                l3_aux_state["log_odds"] = float(np.log(p0 / (1.0 - p0)))
                l3_aux_state["bars_since_peak"] = 0
                l3_aux_state["prev_unreal"] = 0.0
                l3_aux_state["running_mfe"] = 0.0
                l3_aux_state["running_mae"] = 0.0
                l3_aux_state["value_exit_streak"] = 0
                l3_aux_state["strategy_type"] = strategy_type
                l3_aux_state["entry_l2_confidence"] = float(conf)
                if l3_exit_mode == "mtm_adaptive":
                    l3_aux_state["adaptive_mtm_hist"] = []
                    l3_aux_state["adaptive_exit_prob_hist"] = []
                    l3_aux_state["l1a_entry_regime_id"] = int(l1a_entry_regime)
                    l3_aux_state["adaptive_last_reason"] = ""
                    l3_aux_state["adaptive_last_diag"] = None
                current_strategy_type = strategy_type
                if strategy_type in {"IRON_CONDOR_SHORT", "IRON_CONDOR_R4"}:
                    sim = StraddleSimulator(risk_free_rate=float(l3_meta.get("l3_straddle_risk_free_rate", 0.04) or 0.04))
                    entry_iv = float(df["l3_base_iv"].iloc[i + 1]) if "l3_base_iv" in df.columns else max(entry_vol, 0.05)
                    _p = strategy_params or ic_params
                    minutes_to_expiry = max(int(_p["expiry_min"]) - entry_fill_min, 1)
                    t_entry = max(float(minutes_to_expiry) / (390.0 * 252.0), 1e-9)
                    quote = sim.iron_condor_quote(
                        entry_open,
                        t=t_entry,
                        sigma=entry_iv,
                        short_delta=float(_p.get("short_delta", _p.get("ic_short_delta", 0.16))),
                        wing_width_pct=float(_p.get("wing_width_pct", _p.get("ic_wing_width_pct", 0.01))),
                        min_wing_width=float(_p.get("min_wing_width", _p.get("ic_min_wing_width", 0.25))),
                    )
                    credit = float(max(quote.credit * (1.0 - slip_f - comm_f), 1e-9))
                    entry_price = credit
                    l3_aux_state["entry_iv"] = entry_iv
                    l3_aux_state["entry_underlying"] = float(entry_open)
                    l3_aux_state["risk_free_rate"] = float(l3_meta.get("l3_straddle_risk_free_rate", 0.04) or 0.04)
                    l3_aux_state["ic_short_put"] = float(quote.short_put)
                    l3_aux_state["ic_long_put"] = float(quote.long_put)
                    l3_aux_state["ic_short_call"] = float(quote.short_call)
                    l3_aux_state["ic_long_call"] = float(quote.long_call)
                    l3_aux_state["ic_entry_credit"] = credit
                    l3_aux_state["ic_raw_entry_credit"] = float(quote.credit)
                    l3_aux_state["ic_max_loss"] = float(max(quote.max_loss, 1e-9))
                    l3_aux_state["ic_expiry_minutes"] = int(minutes_to_expiry)
                    l3_aux_state["trade_max_hold"] = int(max(1, int(_p["force_exit_min"]) - entry_fill_min))
                    l3_aux_state["strategy_params"] = dict(_p)
                elif strategy_type == "IRON_BUTTERFLY_R0":
                    sim = StraddleSimulator(risk_free_rate=float(l3_meta.get("l3_straddle_risk_free_rate", 0.04) or 0.04))
                    entry_iv = float(df["l3_base_iv"].iloc[i + 1]) if "l3_base_iv" in df.columns else max(entry_vol, 0.05)
                    _p = strategy_params
                    minutes_to_expiry = max(int(_p["expiry_min"]) - entry_fill_min, 1)
                    t_entry = max(float(minutes_to_expiry) / (390.0 * 252.0), 1e-9)
                    bq = sim.iron_butterfly_quote(
                        entry_open,
                        t=t_entry,
                        sigma=entry_iv,
                        wing_width_pct=float(_p.get("ic_wing_width_pct", 0.02)),
                        min_wing_width=float(_p.get("ic_min_wing_width", 0.25)),
                    )
                    credit = float(max(bq.credit * (1.0 - slip_f - comm_f), 1e-9))
                    entry_price = credit
                    l3_aux_state["entry_iv"] = entry_iv
                    l3_aux_state["entry_underlying"] = float(entry_open)
                    l3_aux_state["risk_free_rate"] = float(l3_meta.get("l3_straddle_risk_free_rate", 0.04) or 0.04)
                    l3_aux_state["ibf_k_atm"] = float(bq.k_atm)
                    l3_aux_state["ibf_long_put"] = float(bq.long_put)
                    l3_aux_state["ibf_long_call"] = float(bq.long_call)
                    l3_aux_state["ibf_entry_credit"] = float(credit)
                    l3_aux_state["ibf_raw_credit"] = float(bq.credit)
                    l3_aux_state["ibf_max_loss"] = float(max(bq.max_loss, 1e-9))
                    l3_aux_state["ibf_expiry_minutes"] = int(minutes_to_expiry)
                    l3_aux_state["trade_max_hold"] = int(max(1, int(_p["force_exit_min"]) - entry_fill_min))
                    l3_aux_state["strategy_params"] = dict(_p)
                elif _oos_is_synthetic_strategy(strategy_type):
                    # Must not depend on straddle_sim_mode: when off, the old else-branch priced the *underlying*
                    # (≈$500) as entry_price; MTM/exit still used BS straddle (≈$5–$30) → pnl/「premium」 ≈ -1 every time.
                    sim = StraddleSimulator(risk_free_rate=float(l3_meta.get("l3_straddle_risk_free_rate", 0.04) or 0.04))
                    entry_iv = float(df["l3_base_iv"].iloc[i + 1]) if "l3_base_iv" in df.columns else max(entry_vol, 0.05)
                    minutes_to_expiry = max(int(strategy_params["expiry_min"]) - entry_fill_min, 1)
                    t_entry = max(float(minutes_to_expiry) / (390.0 * 252.0), 1e-9)
                    strike = float(entry_open)
                    raw_premium = sim.straddle_price(entry_open, strike, t_entry, entry_iv)
                    if strategy_type in {"SHORT_STRADDLE_R0", "SHORT_STRADDLE_R3"}:
                        entry_price = float(max(raw_premium * (1.0 - slip_f - comm_f), 1e-9))
                    else:
                        entry_price = float(max(raw_premium * (1.0 + slip_f + comm_f), 1e-9))
                    l3_aux_state["strike"] = strike
                    l3_aux_state["entry_iv"] = entry_iv
                    l3_aux_state["entry_underlying"] = float(entry_open)
                    l3_aux_state["risk_free_rate"] = float(l3_meta.get("l3_straddle_risk_free_rate", 0.04) or 0.04)
                    l3_aux_state["synth_entry_premium"] = float(entry_price)
                    l3_aux_state["synth_raw_entry_premium"] = float(raw_premium)
                    l3_aux_state["synth_expiry_minutes"] = int(minutes_to_expiry)
                    l3_aux_state["trade_max_hold"] = int(max(1, int(strategy_params["force_exit_min"]) - entry_fill_min))
                    l3_aux_state["strategy_params"] = dict(strategy_params)
                    l3_aux_state["synth_peak_mtm"] = float("-inf")
                    l3_aux_state["synth_trail_armed"] = 0.0
                    if str(strategy_type) == "SHORT_STRADDLE_R3":
                        r3_cap = int((os.environ.get("OOS_R3_MAX_HOLD_BARS") or "15").strip() or "15")
                        if r3_cap > 0:
                            l3_aux_state["trade_max_hold"] = min(
                                int(l3_aux_state["trade_max_hold"]),
                                r3_cap,
                            )
                    if strategy_type == "GAMMA_SCALP_R2":
                        l3_aux_state["gamma_roll_tps"] = 0
                        greeks = sim.straddle_greeks(entry_open, strike, t_entry, entry_iv)
                        hedge_pos = float(-greeks.delta)
                        l3_aux_state["hedge_pos"] = hedge_pos
                        l3_aux_state["hedge_cash"] = float(-hedge_pos * entry_open)
                        l3_aux_state["hedge_count"] = 1
                        l3_aux_state["last_hedge_underlying"] = float(entry_open)
                elif straddle_sim_mode:
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
                if _oos_regime_stack_router(strategy_router) and _oos_is_synthetic_strategy(str(strategy_type)):
                    l3_aux_state["oos_l2_regime_synth"] = 1.0
                entry_atr = max(float(df["lbl_atr"].iloc[i]), 1e-3)
                if l3_exit_mode == "mtm_adaptive":
                    l3_aux_state["adaptive_entry_atr"] = float(entry_atr)
                entry_time = entry_fill_time
                hold = 0
                peak_unreal_atr = float("-inf")
                _synth_l3 = _oos_regime_stack_router(strategy_router) and _oos_is_synthetic_strategy(str(strategy_type))
                if hybrid and l3_exit_mode in ("l3", "mtm_adaptive", "value_lt_zero", "value_lt_unreal") and (
                    not _oos_is_synthetic_strategy(str(strategy_type)) or _synth_l3
                ):
                    ref = max(int(l3_traj_cfg.max_seq_len), max_hold)
                    traj_buf = L3TrajRollingState(
                        max_seq_len=int(l3_traj_cfg.max_seq_len),
                        max_seq_ref=ref,
                        seq_feat_dim=int(l3_traj_cfg.seq_feat_dim),
                        mfe_norm_scale=float(getattr(l3_traj_cfg, "mfe_norm_scale", 5.0)),
                        mae_norm_scale=float(getattr(l3_traj_cfg, "mae_norm_scale", 5.0)),
                    )
        else:
            if _loop_exit_check < 5:
                if _loop_exit_check == 0:
                    print(f"[EXIT META] max_hold_cap = {_l3_meta_mh(l3_meta)}", flush=True)
                    print(f"[EXIT META] l3_target_horizon_bars = {l3_meta.get('l3_target_horizon_bars')!r}", flush=True)
                    print(f"[EXIT META] max_hold from first entry = {max_hold}", flush=True)
                print(
                    f"[EXIT DIAG] bar={i} in_pos={in_pos} bars_held={i - entry_idx} max_hold={max_hold}",
                    flush=True,
                )
                _loop_exit_check += 1
            hold += 1
            active_strategy = str(l3_aux_state.get("strategy_type", current_strategy_type))
            if _oos_is_synthetic_strategy(active_strategy):
                strat_params = dict(l3_aux_state.get("strategy_params", {}) or {})
                entry_iv = float(l3_aux_state.get("entry_iv", df["l3_base_iv"].iloc[max(i, 0)] if "l3_base_iv" in df.columns else 0.25))
                current_iv = float(0.65 * entry_iv + 0.35 * (float(df["l3_base_iv"].iloc[i]) if "l3_base_iv" in df.columns else entry_iv))
                current_iv = float(np.clip(current_iv, 0.05, 5.0))
                sim = StraddleSimulator(risk_free_rate=float(l3_aux_state.get("risk_free_rate", 0.04)))
                under_now = float(df["close"].iloc[i])
                is_ic = active_strategy in {"IRON_CONDOR_SHORT", "IRON_CONDOR_R4"}
                is_ibf = active_strategy == "IRON_BUTTERFLY_R0"
                is_short_straddle = active_strategy in {"SHORT_STRADDLE_R0", "SHORT_STRADDLE_R3"}
                is_long_straddle = active_strategy == "LONG_STRADDLE_R1"
                is_gamma = active_strategy == "GAMMA_SCALP_R2"
                if is_ic:
                    t_remaining = max(
                        (float(l3_aux_state.get("ic_expiry_minutes", 1)) - float(hold)) / (390.0 * 252.0),
                        1e-9,
                    )
                    current_option_value = sim.iron_condor_price(
                        under_now,
                        short_put=float(l3_aux_state["ic_short_put"]),
                        long_put=float(l3_aux_state["ic_long_put"]),
                        short_call=float(l3_aux_state["ic_short_call"]),
                        long_call=float(l3_aux_state["ic_long_call"]),
                        t=t_remaining,
                        sigma=current_iv,
                    )
                    pnl_basis = float(max(float(l3_aux_state.get("ic_max_loss", entry_price)), 1e-9))
                    mtm_pnl = float(float(l3_aux_state.get("ic_entry_credit", entry_price)) - current_option_value)
                elif is_ibf:
                    t_remaining = max(
                        (float(l3_aux_state.get("ibf_expiry_minutes", 1)) - float(hold)) / (390.0 * 252.0),
                        1e-9,
                    )
                    current_option_value = sim.iron_butterfly_price(
                        under_now,
                        k_atm=float(l3_aux_state["ibf_k_atm"]),
                        long_put=float(l3_aux_state["ibf_long_put"]),
                        long_call=float(l3_aux_state["ibf_long_call"]),
                        t=t_remaining,
                        sigma=current_iv,
                    )
                    pnl_basis = float(max(float(l3_aux_state.get("ibf_max_loss", entry_price)), 1e-9))
                    mtm_pnl = float(float(l3_aux_state.get("ibf_entry_credit", entry_price)) - current_option_value)
                else:
                    t_remaining = max(
                        (float(l3_aux_state.get("synth_expiry_minutes", 1)) - float(hold)) / (390.0 * 252.0),
                        1e-9,
                    )
                    strike = float(l3_aux_state.get("strike", under_now))
                    current_option_value = sim.straddle_price(under_now, strike, t_remaining, current_iv)
                    entry_premium = float(l3_aux_state.get("synth_entry_premium", entry_price))
                    pnl_basis = float(max(entry_premium, 1e-9))
                    if is_short_straddle:
                        mtm_pnl = float(entry_premium - current_option_value)
                    elif is_gamma:
                        greeks = sim.straddle_greeks(under_now, strike, t_remaining, current_iv)
                        last_hedge = float(l3_aux_state.get("last_hedge_underlying", under_now))
                        hedge_trigger = abs(under_now / max(last_hedge, 1e-9) - 1.0) >= float(strat_params.get("hedge_move_frac", 0.0015))
                        if hedge_trigger:
                            target_pos = float(-greeks.delta)
                            hedge_pos = float(l3_aux_state.get("hedge_pos", 0.0))
                            trade_qty = target_pos - hedge_pos
                            l3_aux_state["hedge_cash"] = float(l3_aux_state.get("hedge_cash", 0.0)) - trade_qty * under_now
                            l3_aux_state["hedge_pos"] = target_pos
                            l3_aux_state["hedge_count"] = int(l3_aux_state.get("hedge_count", 0)) + 1
                            l3_aux_state["last_hedge_underlying"] = under_now
                        hedge_pnl = float(l3_aux_state.get("hedge_cash", 0.0)) + float(l3_aux_state.get("hedge_pos", 0.0)) * under_now
                        mtm_pnl = float((current_option_value - entry_premium) + hedge_pnl)
                    else:
                        mtm_pnl = float(current_option_value - entry_premium)
                l3_aux_state["oos_synth_mtm_pnl_pct"] = float(mtm_pnl / max(pnl_basis, 1e-9))
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
            )
            static = static.ravel()
            if "l2_decision_class" in l2_out.columns:
                flip = int(l2_out.loc[i, "l2_decision_class"])
                if "l2_decision_long" in l2_out.columns and "l2_decision_short" in l2_out.columns:
                    p_long = float(l2_out.loc[i, "l2_decision_long"])
                    p_short = float(l2_out.loc[i, "l2_decision_short"])
                    directional_mass = max(p_long + p_short, 1e-6)
                    directional_conf = abs(p_long - p_short) / directional_mass
                    abstain_like = (flip == 1) or (directional_conf < l2_abstain_margin)
                else:
                    abstain_like = flip == 1
                flip_against = (not straddle_oos) and (not abstain_like) and ((in_pos == 1 and flip == 2) or (in_pos == -1 and flip == 0))
                if _oos_regime_stack_router(str(strategy_router)) and float(
                    l3_aux_state.get("oos_l2_regime_synth", 0.0) or 0.0
                ) > 0.5:
                    flip_against = False
            elif "l2_straddle_on" in l2_out.columns:
                flip = 0 if float(l2_out.loc[i, "l2_straddle_on"]) > 0.5 else 1
                abstain_like = flip == 1
                flip_against = False
            else:
                flip, abstain_like, flip_against = 1, True, False
            deadline_exit = False
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
                if hybrid and traj_buf is not None and l3_exit_mode in (
                    "l3",
                    "mtm_adaptive",
                    "value_lt_zero",
                    "value_lt_unreal",
                ):
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
                _oos_value_exit = l3_exit_mode in ("value_lt_zero", "value_lt_unreal")
                if _oos_value_exit and value_model is None:
                    raise RuntimeError(
                        f"OOS_L3_EXIT_MODE={l3_exit_mode} requires a trained L3 value head "
                        "(meta l3_value_disabled or missing value model)."
                    )
                if not _oos_value_exit:
                    # feat_vec rows follow l3_meta feature_cols (policy matrix). Unified L2+L3 exit
                    # (has_exit_value_heads) uses the same width; market/regime/position are split inside
                    # UnifiedExitWrapper.
                    exit_raw = float(np.ravel(exit_model.predict(feat_vec))[0])
                    raw_arr = np.asarray([exit_raw], dtype=np.float64)
                    if exit_calibrator is not None:
                        exit_prob = float(np.clip(_apply_l3_exit_calibrator(raw_arr, exit_calibrator)[0], 0.0, 1.0))
                    else:
                        exit_prob = float(np.clip(exit_raw, 0.0, 1.0))
                else:
                    exit_prob = 0.0
                if in_pos != 0:
                    runtime_diag["_exit_prob_hold_samples"].append(float(exit_prob))
                    _tpx = l3_aux_state.get("trade_exit_prob_max")
                    l3_aux_state["trade_exit_prob_max"] = float(exit_prob) if _tpx is None else max(float(_tpx), float(exit_prob))
                if (not _oos_value_exit) and (_oos_exit_dist_log or _oos_route_ablation):
                    from core.training.unified.exit_wrapper import UnifiedExitWrapper, log_exit_distribution

                    d_i = pd.Timestamp(df["time_key"].iloc[i]).date()
                    if oos_exit_agg_date is not None and d_i != oos_exit_agg_date:
                        old_d = oos_exit_agg_date
                        if _oos_exit_dist_log and oos_exit_agg_probs:
                            log_exit_distribution(
                                np.asarray(oos_exit_agg_probs, dtype=np.float64),
                                prefix=f"[OOS exit] {symbol} {old_d}",
                            )
                        if _oos_route_ablation and oos_exit_last_feat is not None and isinstance(
                            exit_model, UnifiedExitWrapper
                        ):
                            _cr = exit_model.route_contribution(oos_exit_last_feat)
                            _oos_diag_to_file(f"  [OOS route_contrib] {symbol} {old_d} {_cr}")
                        oos_exit_agg_probs.clear()
                    oos_exit_agg_date = d_i
                    oos_exit_agg_probs.append(exit_prob)
                    oos_exit_last_feat = np.array(feat_vec, dtype=np.float32, copy=True)
                if value_model is None:
                    value_left = 0.0
                else:
                    from core.training.unified.policy_data import _l3_inverse_value_target_transform

                    vprep = dict(l3_meta.get("l3_value_training") or {})
                    vfeat = feat_vec[:, _value_feat_ix]
                    # Binary value heads only; regression modes (incl. remaining_value_atr) use inverse transform / hurdle below.
                    if str(vprep.get("value_target_mode")) in ("peak_cls", "trade_outcome", "remaining_value"):
                        value_left = float(np.clip(np.ravel(value_model.predict(vfeat))[0], 0.0, 1.0))
                    elif value_nonzero_model is None:
                        mu_m = float(np.ravel(value_model.predict(vfeat))[0])
                        value_left = float(_l3_inverse_value_target_transform(np.asarray([mu_m], dtype=np.float64), vprep)[0])
                    else:
                        mu = float(np.ravel(value_model.predict(vfeat))[0])
                        p_nz = float(np.clip(np.ravel(value_nonzero_model.predict(vfeat))[0], 0.0, 1.0))
                        mu_r = float(_l3_inverse_value_target_transform(np.asarray([mu], dtype=np.float64), vprep)[0])
                        value_left = float(mu_r * (p_nz ** float(np.clip(value_hurdle_power, 0.5, 2.0))))
                exit_state_probs = l1a_out.loc[i, L1A_REGIME_COLS].to_numpy(dtype=np.float32)
                exit_state_vol = float(l1a_out.loc[i, "l1a_vol_forecast"])
                pa_state = df.loc[i, PA_STATE_FEATURES] if all(col in df.columns for col in PA_STATE_FEATURES) else None
                u_atr = float(static[_sidx["l3_unreal_pnl_atr"]])
                rdiv = float(static[_sidx["l3_regime_divergence"]])
                exit_prob_threshold, state_hold_ref, _exit_state_key = l3_exit_policy_params(
                    exit_state_probs,
                    exit_state_vol,
                    hold,
                    l3_meta,
                    pa_state=pa_state,
                    unreal_pnl_atr=u_atr,
                    regime_divergence=rdiv,
                )
                _ = _exit_state_key
                ent_soft = float(exit_prob_threshold)
                lev_soft = float(exit_prob_threshold)
                if _oos_value_exit:
                    u_cur = _oos_current_unreal_for_value_exit(
                        static, _sidx, straddle_mode=bool(_l3_straddle_sim_mode_enabled(l3_meta))
                    )
                    if l3_exit_mode == "value_lt_zero":
                        value_exit_candidate = bool(value_left < 0.0)
                    else:
                        value_exit_candidate = bool(value_left < u_cur * value_exit_unreal_frac)
                    value_exit_candidate = bool(value_exit_candidate and hold >= value_exit_min_hold)
                    if value_exit_candidate:
                        l3_aux_state["value_exit_streak"] = int(l3_aux_state.get("value_exit_streak", 0)) + 1
                        runtime_diag["value_exit_signal_count"] = int(runtime_diag["value_exit_signal_count"]) + 1
                    else:
                        l3_aux_state["value_exit_streak"] = 0
                    policy_exit = bool(int(l3_aux_state.get("value_exit_streak", 0)) >= value_exit_confirm_bars)
                    if policy_exit:
                        runtime_diag["value_exit_confirmed_count"] = int(runtime_diag["value_exit_confirmed_count"]) + 1
                else:
                    if (
                        l3_exit_mode == "mtm_adaptive"
                        and adapt_mtm_eval is not None
                        and adaptive_mtm_cfg is not None
                    ):
                        _mh = l3_aux_state.get("adaptive_mtm_hist")
                        _ph = l3_aux_state.get("adaptive_exit_prob_hist")
                        _e_atr = float(l3_aux_state.get("adaptive_entry_atr", 0.0) or 0.0)
                        _ok = (
                            isinstance(_mh, list)
                            and isinstance(_ph, list)
                            and _e_atr > 0.0
                            and np.isfinite(u_atr)
                        )
                        _tmh_ux = int(l3_aux_state.get("trade_max_hold", max_hold))
                        _mtm_shr = _oos_mtm_state_hold_ref_for_adaptive(trade_max_hold=_tmh_ux)
                        if _ok:
                            _mh.append(float(u_atr))
                            _ph.append(float(exit_prob))
                            _cap = min(
                                int(adaptive_mtm_cfg.history_max_bars),
                                int(l3_aux_state.get("trade_max_hold", max_hold)) + 5,
                            )
                            adapt_mtm_trim(_mh, _ph, _cap)
                            policy_exit, _mtm_r, _mtm_d = adapt_mtm_eval(
                                config=adaptive_mtm_cfg,
                                bars_held=hold,
                                mtm_history=_mh,
                                exit_prob_history=_ph,
                                current_exit_prob=float(exit_prob),
                                entry_atr=_e_atr,
                                entry_regime_id=int(l3_aux_state.get("l1a_entry_regime_id", 0) or 0),
                                state_hold_ref=int(_mtm_shr),
                                trade_max_hold=_tmh_ux,
                            )
                            l3_aux_state["adaptive_last_reason"] = str(_mtm_r)
                            l3_aux_state["adaptive_last_diag"] = _mtm_d
                        else:
                            _wn = int(runtime_diag.get("_mtm_adaptive_fallback_warn", 0))
                            if _wn < 3:
                                print(
                                    f"[OOS] WARN: mtm_adaptive missing MTM/ATR state; using exit-prob threshold fallback "
                                    f"(symbol={symbol} bar={i})",
                                    flush=True,
                                )
                                runtime_diag["_mtm_adaptive_fallback_warn"] = _wn + 1
                            if hold < int(_mtm_shr):
                                runtime_diag["steps_early"] = int(runtime_diag["steps_early"]) + 1
                                runtime_diag["soft_enter_thr_early_sum"] = float(
                                    runtime_diag["soft_enter_thr_early_sum"]
                                ) + float(ent_soft)
                                if exit_prob >= ent_soft:
                                    runtime_diag["soft_enter_hits_early"] = int(
                                        runtime_diag["soft_enter_hits_early"]
                                    ) + 1
                            else:
                                runtime_diag["steps_late"] = int(runtime_diag["steps_late"]) + 1
                                runtime_diag["soft_enter_thr_late_sum"] = float(
                                    runtime_diag["soft_enter_thr_late_sum"]
                                ) + float(lev_soft)
                                if exit_prob >= ent_soft:
                                    runtime_diag["soft_enter_hits_late"] = int(
                                        runtime_diag["soft_enter_hits_late"]
                                    ) + 1
                            policy_exit, exit_infer_state = l3_exit_decision_live(
                                exit_prob,
                                exit_infer_state,
                                exit_prob_threshold=exit_prob_threshold,
                            )
                    else:
                        if hold < int(state_hold_ref):
                            runtime_diag["steps_early"] = int(runtime_diag["steps_early"]) + 1
                            runtime_diag["soft_enter_thr_early_sum"] = float(
                                runtime_diag["soft_enter_thr_early_sum"]
                            ) + float(ent_soft)
                            if exit_prob >= ent_soft:
                                runtime_diag["soft_enter_hits_early"] = int(runtime_diag["soft_enter_hits_early"]) + 1
                        else:
                            runtime_diag["steps_late"] = int(runtime_diag["steps_late"]) + 1
                            runtime_diag["soft_enter_thr_late_sum"] = float(
                                runtime_diag["soft_enter_thr_late_sum"]
                            ) + float(lev_soft)
                            if exit_prob >= ent_soft:
                                runtime_diag["soft_enter_hits_late"] = int(runtime_diag["soft_enter_hits_late"]) + 1
                        policy_exit, exit_infer_state = l3_exit_decision_live(
                            exit_prob,
                            exit_infer_state,
                            exit_prob_threshold=exit_prob_threshold,
                        )
                if _oos_value_exit and not policy_exit and hold >= int(l3_aux_state.get("trade_max_hold", max_hold)):
                    policy_exit = True
                    deadline_exit = True
                    runtime_diag["deadline_exit_count"] = int(runtime_diag["deadline_exit_count"]) + 1
                if (
                    (not _oos_value_exit)
                    and _oos_regime_stack_router(str(strategy_router))
                    and float(l3_aux_state.get("oos_l2_regime_synth", 0.0) or 0.0) > 0.5
                    and l3_exit_mode in ("l3", "mtm_adaptive")
                    and (not bool(policy_exit))
                    and hold >= int(l3_aux_state.get("trade_max_hold", max_hold))
                ):
                    policy_exit = True
                    deadline_exit = True
                    runtime_diag["deadline_exit_count"] = int(runtime_diag["deadline_exit_count"]) + 1
                if _loop_exit_probe < 20:
                    _ep_extra = (
                        f" mtm_reason={l3_aux_state.get('adaptive_last_reason', '')!r}"
                        if l3_exit_mode == "mtm_adaptive"
                        else ""
                    )
                    print(
                        f"[EXIT PROBE] bar={i} hold={hold} "
                        f"exit_prob={exit_prob:.4f} value_left={value_left:.4f} "
                        f"exit_thresh={exit_prob_threshold:.4f} "
                        f"policy_exit={policy_exit} flip_against={flip_against} "
                        f"state_hold_ref={state_hold_ref}{_ep_extra}",
                        flush=True,
                    )
                    _loop_exit_probe += 1
            short_vol_explosion = False
            if straddle_oos and in_pos < 0 and entry_idx >= 0:
                implied_atr = float(l2_out.loc[entry_idx].get("l2_implied_proxy_range", l2_out.loc[entry_idx].get("l2_range_pred", 1.0)))
                implied_frac = implied_atr * max(float(entry_atr), 1e-6) / max(abs(float(l3_aux_state.get("entry_underlying", entry_price))), 1e-9)
                move_frac = abs(float(df["close"].iloc[i]) / max(abs(float(l3_aux_state.get("entry_underlying", entry_price))), 1e-9) - 1.0)
                vol_exit_cfg = l3_meta.get("l3_vol_exit_config") if isinstance(l3_meta.get("l3_vol_exit_config"), dict) else {}
                short_mult = float(
                    np.clip(
                        float(l3_meta.get("l3_short_vol_range_explosion_mult", vol_exit_cfg.get("short_vol_range_explosion_mult", 1.50))),
                        1.0,
                        4.0,
                    )
                )
                short_vol_explosion = bool(move_frac >= implied_frac * short_mult)
            if policy_exit or flip_against or short_vol_explosion:
                if flip_against and (not bool(policy_exit)) and (not short_vol_explosion):
                    runtime_diag["_signal_flip_at_exit_probs"].append(float(exit_prob))
                if policy_exit:
                    runtime_diag["policy_exit_count"] = int(runtime_diag["policy_exit_count"]) + 1
                elif flip_against:
                    runtime_diag["flip_exit_count"] = int(runtime_diag["flip_exit_count"]) + 1
                elif short_vol_explosion:
                    runtime_diag["policy_exit_count"] = int(runtime_diag["policy_exit_count"]) + 1
                runtime_diag["non_synthetic_exit_count"] = int(runtime_diag["non_synthetic_exit_count"]) + 1
                # L3 exit head actually fired (not L2 flip / vol stop). Do not conflate with all OOS_L3_EXIT_MODE=l3 closes.
                if l3_exit_mode == "l3" and bool(policy_exit):
                    runtime_diag["learned_l3_exit_count"] = int(runtime_diag["learned_l3_exit_count"]) + 1
                elif l3_exit_mode == "mtm_adaptive" and bool(policy_exit):
                    runtime_diag["adaptive_mtm_exit_count"] = int(runtime_diag["adaptive_mtm_exit_count"]) + 1
                exit_open = float(df["open"].iloc[i + 1])
                is_l2s_exit = float(l3_aux_state.get("oos_l2_regime_synth", 0.0) or 0.0) > 0.5
                if is_l2s_exit:
                    stx = str(l3_aux_state.get("strategy_type", "LONG_STRADDLE_R1"))
                    sim2 = StraddleSimulator(risk_free_rate=float(l3_aux_state.get("risk_free_rate", 0.04)))
                    eiv = float(l3_aux_state.get("entry_iv", 0.25))
                    exiv = float(0.65 * eiv + 0.35 * (float(df["l3_base_iv"].iloc[i + 1]) if "l3_base_iv" in df.columns else eiv))
                    exiv = float(np.clip(exiv, 0.05, 5.0))
                    icx = stx in {"IRON_CONDOR_SHORT", "IRON_CONDOR_R4"}
                    ibfx = stx == "IRON_BUTTERFLY_R0"
                    ssd = stx in {"SHORT_STRADDLE_R0", "SHORT_STRADDLE_R3"}
                    gam = stx == "GAMMA_SCALP_R2"
                    pnl_basis_x = max(float(l3_aux_state.get("synth_entry_premium", entry_price)), 1e-9)
                    if icx:
                        t_ex = max(
                            (float(l3_aux_state.get("ic_expiry_minutes", 1)) - float(hold + 1)) / (390.0 * 252.0),
                            1e-9,
                        )
                        ex_raw = sim2.iron_condor_price(
                            exit_open,
                            short_put=float(l3_aux_state["ic_short_put"]),
                            long_put=float(l3_aux_state["ic_long_put"]),
                            short_call=float(l3_aux_state["ic_short_call"]),
                            long_call=float(l3_aux_state["ic_long_call"]),
                            t=t_ex,
                            sigma=exiv,
                        )
                        exit_price = float(ex_raw * (1.0 + slip_f + comm_f))
                        pnl_x = float(float(l3_aux_state.get("ic_entry_credit", entry_price)) - exit_price)
                        pnl_basis_x = float(max(float(l3_aux_state.get("ic_max_loss", 1.0)), 1e-9))
                    elif ibfx:
                        t_ex = max(
                            (float(l3_aux_state.get("ibf_expiry_minutes", 1)) - float(hold + 1)) / (390.0 * 252.0),
                            1e-9,
                        )
                        ex_raw = sim2.iron_butterfly_price(
                            exit_open,
                            k_atm=float(l3_aux_state["ibf_k_atm"]),
                            long_put=float(l3_aux_state["ibf_long_put"]),
                            long_call=float(l3_aux_state["ibf_long_call"]),
                            t=t_ex,
                            sigma=exiv,
                        )
                        exit_price = float(ex_raw * (1.0 + slip_f + comm_f))
                        pnl_x = float(float(l3_aux_state.get("ibf_entry_credit", entry_price)) - exit_price)
                        pnl_basis_x = float(max(float(l3_aux_state.get("ibf_max_loss", 1.0)), 1e-9))
                    else:
                        t_ex = max(
                            (float(l3_aux_state.get("synth_expiry_minutes", 1)) - float(hold + 1)) / (390.0 * 252.0),
                            1e-9,
                        )
                        stkx = float(l3_aux_state.get("strike", exit_open))
                        ex_raw = sim2.straddle_price(exit_open, stkx, t_ex, exiv)
                        if ssd:
                            exit_price = float(ex_raw * (1.0 + slip_f + comm_f))
                            pnl_x = float(float(l3_aux_state.get("synth_entry_premium", entry_price)) - exit_price)
                        elif gam:
                            exit_price = float(ex_raw * (1.0 - slip_f - comm_f))
                            hpx = float(l3_aux_state.get("hedge_cash", 0.0)) + float(
                                l3_aux_state.get("hedge_pos", 0.0)
                            ) * float(exit_open)
                            pnl_x = float(
                                (exit_price - float(l3_aux_state.get("synth_entry_premium", entry_price))) + hpx
                            )
                        else:
                            exit_price = float(ex_raw * (1.0 - slip_f - comm_f))
                            pnl_x = float(exit_price - float(l3_aux_state.get("synth_entry_premium", entry_price)))
                    _er_x = (
                        int(np.argmax(l1a_out.loc[entry_idx, L1A_REGIME_COLS].to_numpy(dtype=np.float32)))
                        if entry_idx >= 0
                        else 0
                    )
                    _nsx = _oos_trade_size_mult(_er_x, float(l3_aux_state.get("entry_l2_confidence", 0.5)))
                    ret = float(pnl_x / max(pnl_basis_x, 1e-9)) * _nsx
                elif _l3_straddle_sim_mode_enabled(l3_meta):
                    dte_days = int(l3_aux_state.get("dte_days", _oos_default_straddle_dte_days(l3_meta)))
                    strike = float(l3_aux_state.get("strike", exit_open))
                    entry_iv = float(l3_aux_state.get("entry_iv", df["l3_base_iv"].iloc[max(i, 0)] if "l3_base_iv" in df.columns else 0.25))
                    current_iv = float(0.65 * entry_iv + 0.35 * (float(df["l3_base_iv"].iloc[i + 1]) if "l3_base_iv" in df.columns else entry_iv))
                    t_remaining = max(dte_days / 252.0 - (hold + 1) / (390.0 * 252.0), 1e-9)
                    sim = StraddleSimulator(risk_free_rate=float(l3_aux_state.get("risk_free_rate", 0.04)))
                    raw_exit_value = sim.straddle_price(exit_open, strike, t_remaining, current_iv)
                    exit_price = float(raw_exit_value * (1.0 - slip_f - comm_f))
                    ret = ((exit_price / entry_price) - 1.0) * in_pos
                else:
                    exit_price = _oos_adjust_fill(exit_open, is_buy=(in_pos == -1), slip_f=slip_f, comm_f=comm_f)
                    ret = (exit_price / entry_price - 1.0) * in_pos
                _er_ns = (
                    int(np.argmax(l1a_out.loc[entry_idx, L1A_REGIME_COLS].to_numpy(dtype=np.float32)))
                    if entry_idx >= 0
                    else 0
                )
                _ns = _oos_trade_size_mult(_er_ns, float(l3_aux_state.get("entry_l2_confidence", 0.5)))
                if not is_l2s_exit:
                    ret = float(ret) * _ns
                if l3_exit_mode == "atr_trailing" and policy_exit:
                    exit_reason = "ATR_Trail_Exit"
                elif deadline_exit:
                    exit_reason = "Deadline_Exit"
                elif policy_exit:
                    if l3_exit_mode == "value_lt_zero":
                        exit_reason = "ValueLt0_Exit"
                    elif l3_exit_mode == "value_lt_unreal":
                        exit_reason = "ValueLtUnrealFrac_Exit"
                    elif l3_exit_mode == "mtm_adaptive":
                        exit_reason = f"MtmAdaptive_{str(l3_aux_state.get('adaptive_last_reason', 'unknown'))}"
                    else:
                        exit_reason = "Policy_Exit"
                elif short_vol_explosion:
                    exit_reason = "ShortVol_Explosion_Stop"
                else:
                    exit_reason = "Signal_Flip"
                if bar_exit_diag is not None and entry_idx >= 0:
                    _ep = float(entry_price)
                    if _ep > 0.0 and np.isfinite(_ep):
                        _er_id = int(
                            np.argmax(l1a_out.loc[entry_idx, L1A_REGIME_COLS].to_numpy(dtype=np.float32))
                        )
                        bar_exit_diag.record_trade(
                            len(trades),
                            _ep,
                            side=int(in_pos),
                            entry_regime=_er_id,
                            actual_exit_bar=int(hold),
                            entry_bar_idx=int(entry_idx),
                            all_close_prices=_all_closes,
                        )
                        _s = int(entry_idx) + 1
                        _e = min(_s + int(bar_exit_diag.max_track), int(_all_closes.shape[0]))
                        bar_exit_records.append(
                            {
                                "trade_id": len(trades),
                                "entry_price": _ep,
                                "bar_prices": _all_closes[_s:_e].tolist(),
                                "side": int(in_pos),
                                "entry_regime": _er_id,
                                "actual_exit_bar": int(hold),
                            }
                        )
                _tmh = int(l3_aux_state.get("trade_max_hold", max_hold))
                _deadline_pnl = _oos_deadline_pnl_atr(
                    straddle_mode=bool(_l3_straddle_sim_mode_enabled(l3_meta)),
                    df=df,
                    l3_aux_state=l3_aux_state,
                    l3_meta=l3_meta,
                    entry_idx=int(entry_idx),
                    entry_price=float(entry_price),
                    entry_atr=float(entry_atr),
                    in_pos=int(in_pos),
                    trade_max_hold=_tmh,
                )
                _primary_exit_driver = _oos_stack_primary_exit_driver(
                    synthetic=False,
                    l3_exit_mode=l3_exit_mode,
                    policy_exit=bool(policy_exit),
                    flip_against=bool(flip_against),
                    short_vol_explosion=bool(short_vol_explosion),
                    deadline_exit=bool(deadline_exit),
                )
                _st_out = str(l3_aux_state.get("strategy_type", "")) if is_l2s_exit else ""
                trades.append(
                    {
                        "symbol": symbol,
                        "primary_exit_driver": _primary_exit_driver,
                        "entry_time": entry_time,
                        "exit_time": df["time_key"].iloc[i + 1],
                        "direction": (
                            _st_out
                            if is_l2s_exit
                            else (
                                "STRADDLE"
                                if _l3_straddle_sim_mode_enabled(l3_meta)
                                else ("LONG" if in_pos == 1 else "SHORT")
                            )
                        ),
                        "strategy_type": (
                            _st_out
                            if is_l2s_exit
                            else ("STRADDLE_LONG" if in_pos > 0 else "STRADDLE_SHORT")
                        ),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return": ret,
                        "holding_bars": hold,
                        "deadline_pnl_atr": float(_deadline_pnl),
                        "trade_max_hold": int(_tmh),
                        "entry_regime_id": int(
                            np.argmax(l1a_out.loc[entry_idx, L1A_REGIME_COLS].to_numpy(dtype=np.float32))
                        ) if entry_idx >= 0 else -1,
                        "exit_soft_enter_thr": float(ent_soft),
                        "exit_soft_leave_thr": float(lev_soft),
                        "exit_hold_ref_bars": int(state_hold_ref),
                        "exit_prob": float(exit_prob),
                        "exit_prob_max_during_hold": float(l3_aux_state.get("trade_exit_prob_max", np.nan))
                        if l3_aux_state.get("trade_exit_prob_max") is not None
                        else float("nan"),
                        "exit_value_left": float(value_left),
                        "value_exit_min_hold": int(value_exit_min_hold),
                        "value_exit_confirm_bars": int(value_exit_confirm_bars),
                        "value_exit_unreal_frac": float(value_exit_unreal_frac),
                        "exit_reason": exit_reason,
                        "adaptive_exit_detail": str(l3_aux_state.get("adaptive_last_reason", ""))
                        if l3_exit_mode == "mtm_adaptive"
                        else "",
                        **(
                            {
                                "adaptive_underwater_active": l3_aux_state.get("adaptive_last_diag", {}).get(
                                    "underwater", {}
                                ).get("active", "")
                                if isinstance(l3_aux_state.get("adaptive_last_diag"), dict)
                                else "",
                                "adaptive_underwater_streak": l3_aux_state.get("adaptive_last_diag", {}).get(
                                    "underwater", {}
                                ).get("streak", "")
                                if isinstance(l3_aux_state.get("adaptive_last_diag"), dict)
                                else "",
                                "adaptive_vote_k_used": l3_aux_state.get("adaptive_last_diag", {}).get(
                                    "vote_k_used", ""
                                )
                                if isinstance(l3_aux_state.get("adaptive_last_diag"), dict)
                                else "",
                                "adaptive_is_profitable": l3_aux_state.get("adaptive_last_diag", {}).get(
                                    "is_profitable", ""
                                )
                                if isinstance(l3_aux_state.get("adaptive_last_diag"), dict)
                                else "",
                            }
                            if l3_exit_mode == "mtm_adaptive"
                            else {}
                        ),
                        "notional_scale": float(_ns),
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
                current_strategy_type = ""
                traj_buf = None
                peak_unreal_atr = float("-inf")
                l3_aux_state.clear()
                exit_infer_state.reset()
    if oos_exit_agg_date is not None and oos_exit_agg_probs and _oos_exit_dist_log:
        from core.training.unified.exit_wrapper import log_exit_distribution

        log_exit_distribution(
            np.asarray(oos_exit_agg_probs, dtype=np.float64),
            prefix=f"[OOS exit] {symbol} {oos_exit_agg_date} (eod)",
        )
    if _oos_route_ablation and oos_exit_last_feat is not None:
        from core.training.unified.exit_wrapper import UnifiedExitWrapper

        if isinstance(exit_model, UnifiedExitWrapper):
            _cr2 = exit_model.route_contribution(oos_exit_last_feat)
            _oos_diag_to_file(f"  [OOS route_contrib] {symbol} {oos_exit_agg_date} (eod) {_cr2}")
    print(f"[LOOP DIAG] symbol={symbol}", flush=True)
    print(f"[LOOP DIAG] flat bars:        {_loop_flat}", flush=True)
    print(f"[LOOP DIAG] cls != 1:         {_loop_cls_ok}", flush=True)
    print(f"[LOOP DIAG] side != 0:        {_loop_side_nonzero}", flush=True)
    print(f"[LOOP DIAG] l1a blocked:      {_loop_l1a_blocked}", flush=True)
    print(f"[LOOP DIAG] trades opened:    {_loop_trade_opened}", flush=True)
    print(f"[LOOP DIAG] first side==0 examples:", flush=True)
    for ex in _loop_first_side_zero:
        print(f"  {ex}", flush=True)
    print(f"[META DIAG] block_entry_l1a_regimes = {block_entry_l1a_regimes!r}", flush=True)
    if not _OOS_L3_ENTRY_POLICY_SOURCE_PRINTED and l3_entry_policy_params is not None:
        _OOS_L3_ENTRY_POLICY_SOURCE_PRINTED = True
        try:
            print(
                "[FUNC DIAG] l3_entry_policy_params:\n" + inspect.getsource(l3_entry_policy_params),
                flush=True,
            )
        except (OSError, TypeError) as ex:
            print(f"[FUNC DIAG] l3_entry_policy_params getsource failed: {ex!r}", flush=True)
    runtime_diag["loop_entry_diag"] = {
        "flat_bars": int(_loop_flat),
        "cls_ne1": int(_loop_cls_ok),
        "side_nonzero": int(_loop_side_nonzero),
        "l1a_blocked": int(_loop_l1a_blocked),
        "trades_opened": int(_loop_trade_opened),
        "first_side_zero": _loop_first_side_zero,
    }
    if bar_exit_diag is not None and bar_exit_diag.trades:
        runtime_diag["bar_exit_diagnostics"] = bar_exit_diag.report(label=symbol)
        runtime_diag["bar_exit_records"] = bar_exit_records
    else:
        runtime_diag["bar_exit_diagnostics"] = {}
        runtime_diag["bar_exit_records"] = bar_exit_records
    return pd.DataFrame(trades), df, runtime_diag


def _cumulative_return_on_timeline(time_key: pd.Series, trades: pd.DataFrame) -> np.ndarray:
    """Piecewise-constant cumulative return on the bar grid from closed trades."""
    n = len(time_key)
    if trades is None or trades.empty or n == 0:
        return np.zeros(n, dtype=np.float64)
    t = trades.sort_values("exit_time").reset_index(drop=True)
    r = t["return"].to_numpy(dtype=np.float64)
    cum_at_exit = np.cumsum(r)
    ts_bar = pd.to_datetime(time_key).to_numpy()
    ts_exit = pd.to_datetime(t["exit_time"]).to_numpy()
    out = np.zeros(n, dtype=np.float64)
    j = 0
    for i in range(n):
        while j < len(ts_exit) and ts_exit[j] <= ts_bar[i]:
            j += 1
        if j > 0:
            out[i] = cum_at_exit[j - 1]
    return out


def _format_metric_legend(m: dict[str, Any]) -> str:
    if int(m.get("n_trades", 0)) == 0:
        return "无成交"
    pf = m.get("profit_factor", float("nan"))
    pf_s = f"{pf:.2f}" if np.isfinite(pf) else "—"
    sh = float(m.get("sharpe_trade_annualized", float("nan")))
    sh_s = f"{sh:.3f}" if np.isfinite(sh) else "—"
    return (
        f"最大回撤 Max DD: {m['max_drawdown_pct']:.2f}%\n"
        f"胜率 Win rate: {m['win_rate']:.2%}\n"
        f"累计收益 Total: {m['total_return_pct']:.2f}%\n"
        f"夏普 Sharpe: {sh_s}  (per-trade ann.)\n"
        f"平均/笔 Mean: {100.0 * m['mean_return_per_trade_frac']:.2f}%\n"
        f"盈亏比 Profit factor: {pf_s}\n"
        f"笔数 Trades: {m['n_trades']}"
    )


def plot_oos_price_and_cumulative_return(
    symbol: str,
    df_price: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: dict[str, Any],
    out_path: Path,
    *,
    title_subtitle: str | None = None,
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

    from backtests.oos_regime_figure import plot_regime_candles_for_oos

    stats_txt = _format_metric_legend(metrics)
    _full_title = _oos_figure_title(symbol)
    if title_subtitle:
        _full_title = f"{_full_title}\n{title_subtitle}"
    if plot_regime_candles_for_oos(
        symbol,
        df_price,
        trades,
        out_path,
        full_title=_full_title,
        stats_txt=stats_txt,
    ):
        return

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti TC", "Noto Sans CJK SC", "DejaVu Sans"]

    tk = pd.to_datetime(df_price["time_key"])
    px = pd.to_numeric(df_price[price_col], errors="coerce").to_numpy(dtype=np.float64)
    cum_ret = _cumulative_return_on_timeline(tk, trades)
    cum_ret_pct = cum_ret * 100.0

    fig, ax1 = plt.subplots(figsize=(14, 6), layout="constrained")
    c_price = "#1f77b4"
    c_eq = "#ff7f0e"
    (ln_price,) = ax1.plot(tk, px, color=c_price, linewidth=1.0, label="价格 Close")
    ax1.set_xlabel("时间")
    ax1.set_ylabel("价格", color=c_price)
    ax1.tick_params(axis="y", labelcolor=c_price)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    # Do not use fig.autofmt_xdate() with layout="constrained": it calls subplots_adjust and conflicts
    # with the layout engine (UserWarning + broken spacing). Rotate x labels explicitly instead.
    ax1.tick_params(axis="x", rotation=22, labelsize=8)
    plt.setp(ax1.get_xticklabels(), ha="right")

    ax2 = ax1.twinx()
    (ln_eq,) = ax2.plot(tk, cum_ret_pct, color=c_eq, linewidth=1.2, linestyle="-", label="累计收益 %")
    ax2.set_ylabel("累计收益 (%)", color=c_eq)
    ax2.tick_params(axis="y", labelcolor=c_eq)
    ax2.axhline(0.0, color=c_eq, linewidth=0.6, alpha=0.35, zorder=0)

    leg_lines = [ln_price, ln_eq]
    leg_labels = [ln.get_label() for ln in leg_lines]
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

    ax1.set_title(_full_title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OOS plot] saved -> {out_path}", flush=True)


def summarize_trade_returns(trades: pd.DataFrame, *, label: str) -> dict[str, Any]:
    """Simple cumulative return on closed trades sorted by exit_time."""
    if trades is None or trades.empty:
        return {"label": label, "n_trades": 0}
    t = trades.sort_values("exit_time").reset_index(drop=True)
    r = t["return"].to_numpy(dtype=np.float64)
    n = int(len(r))
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    dd = cum - peak
    max_dd_frac = float(np.min(dd))
    total_return_frac = float(cum[-1])
    t0 = pd.Timestamp(t["exit_time"].iloc[0])
    t1 = pd.Timestamp(t["exit_time"].iloc[-1])
    span_days = max((t1 - t0).total_seconds() / 86400.0, 1e-6)
    years = float(span_days / 365.25)
    cagr_frac = float("nan")
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
    calmar = float("nan")
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


def _oos_portfolio_max_open_positions() -> int:
    return int(np.clip(int(OOS_BASELINE_PORTFOLIO_MAX_OPEN_POSITIONS), 0, 1000))


def apply_portfolio_open_position_cap(
    trades: pd.DataFrame,
    *,
    max_open_positions: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Post-sim portfolio guard: keep at most N overlapping symbol positions.

    Per-symbol simulation is independent, so this filter applies a portfolio-level
    capital constraint before combined metrics are computed.
    """
    if trades is None or trades.empty or int(max_open_positions) <= 0:
        return trades.copy() if trades is not None else pd.DataFrame(), {
            "enabled": False,
            "max_open_positions": int(max_open_positions),
            "skipped_trades": 0,
        }
    need = {"entry_time", "exit_time"}
    if not need.issubset(trades.columns):
        return trades.copy(), {
            "enabled": False,
            "max_open_positions": int(max_open_positions),
            "skipped_trades": 0,
            "reason": "missing_entry_or_exit_time",
        }
    t = trades.copy()
    t["_entry_ts"] = pd.to_datetime(t["entry_time"], errors="coerce")
    t["_exit_ts"] = pd.to_datetime(t["exit_time"], errors="coerce")
    t = t.sort_values(["_entry_ts", "symbol"], na_position="last").reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    active: list[pd.Timestamp] = []
    keep: list[bool] = []
    skip_records: list[dict[str, Any]] = []
    for _, row in t.iterrows():
        ent = row["_entry_ts"]
        ext = row["_exit_ts"]
        if pd.isna(ent) or pd.isna(ext):
            keep.append(True)
            continue
        active = [x for x in active if x > ent]
        if len(active) >= int(max_open_positions):
            keep.append(False)
            if len(skip_records) < 20:
                skip_records.append(
                    {
                        "symbol": str(row.get("symbol", "")),
                        "entry_time": str(row.get("entry_time", "")),
                        "exit_time": str(row.get("exit_time", "")),
                        "active_positions": int(len(active)),
                    }
                )
            continue
        keep.append(True)
        active.append(ext)
    t["_portfolio_cap_keep"] = np.asarray(keep, dtype=bool)
    skipped = t.loc[~t["_portfolio_cap_keep"]].copy()
    kept = t.loc[t["_portfolio_cap_keep"]].copy()
    drop_cols = ["_entry_ts", "_exit_ts", "_orig_idx"]
    kept = kept.drop(columns=[c for c in drop_cols if c in kept.columns]).reset_index(drop=True)
    diag = {
        "enabled": True,
        "max_open_positions": int(max_open_positions),
        "input_trades": int(len(t)),
        "kept_trades": int(len(kept)),
        "skipped_trades": int(len(skipped)),
        "skipped_by_symbol": (
            {str(k): int(v) for k, v in skipped["symbol"].value_counts().sort_index().items()}
            if "symbol" in skipped.columns and not skipped.empty
            else {}
        ),
        "sample_skips": skip_records,
    }
    return kept, diag


def _exit_prob_dist_summary(arr: np.ndarray) -> dict[str, Any]:
    """Mean / percentiles / max of exit_prob; empty input → all None."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "n": 0,
            "mean": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "p75": float(np.percentile(a, 75.0)),
        "p90": float(np.percentile(a, 90.0)),
        "p95": float(np.percentile(a, 95.0)),
        "max": float(np.max(a)),
    }


def _finalize_per_symbol_exit_prob_lists(diag_by_symbol: dict[str, dict[str, Any]], symbols: list[str]) -> dict[str, Any]:
    """Turn raw exit_prob sample lists into per-symbol summaries; return pooled (all symbols) for COMBINED merge."""
    h_cat: list[float] = []
    f_cat: list[float] = []
    for sym in symbols:
        d = diag_by_symbol.get(sym) or {}
        h = d.pop("_exit_prob_hold_samples", None) or []
        f = d.pop("_signal_flip_at_exit_probs", None) or []
        if isinstance(h, list):
            h_cat.extend(h)
        if isinstance(f, list):
            f_cat.extend(f)
        d["exit_prob_during_hold"] = _exit_prob_dist_summary(np.asarray(h, dtype=np.float64))
        d["signal_flip_exit_prob_at_exit"] = _exit_prob_dist_summary(np.asarray(f, dtype=np.float64))
    return {
        "exit_prob_during_hold": _exit_prob_dist_summary(np.asarray(h_cat, dtype=np.float64)) if h_cat else _exit_prob_dist_summary(np.array([])),
        "signal_flip_exit_prob_at_exit": _exit_prob_dist_summary(np.asarray(f_cat, dtype=np.float64))
        if f_cat
        else _exit_prob_dist_summary(np.array([])),
    }


def summarize_exit_path_diagnostics(
    trades: pd.DataFrame,
    runtime_diag: dict[str, Any] | None,
    *,
    label: str,
    runtime_prefix: str | None = "runtime",
) -> dict[str, Any]:
    t = trades if trades is not None else pd.DataFrame()
    total = int(len(t))
    _policy_like = {"Policy_Exit", "ValueLt0_Exit", "ValueLtUnrealFrac_Exit", "Deadline_Exit"}
    policy_n = int(t["exit_reason"].isin(_policy_like).sum()) if total > 0 and "exit_reason" in t.columns else 0
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
    out = {
        "label": label,
        "n_trades": total,
        "policy_exit_share": float(policy_n / max(total, 1)),
        "flip_exit_share": float(flip_n / max(total, 1)),
        "holding_bars_histogram": hold_hist,
        "hold_by_exit_reason": by_reason_hold,
    }
    if runtime_diag is not None and runtime_prefix:
        early_steps = int(runtime_diag.get("steps_early", 0))
        late_steps = int(runtime_diag.get("steps_late", 0))
        thr_early_sum = float(runtime_diag.get("soft_enter_thr_early_sum", 0.0))
        thr_late_sum = float(runtime_diag.get("soft_enter_thr_late_sum", 0.0))
        out.update(
            {
                f"{runtime_prefix}_policy_exit_count": int(runtime_diag.get("policy_exit_count", 0)),
                f"{runtime_prefix}_flip_exit_count": int(runtime_diag.get("flip_exit_count", 0)),
                f"{runtime_prefix}_deadline_exit_count": int(runtime_diag.get("deadline_exit_count", 0)),
                f"{runtime_prefix}_value_exit_signal_count": int(runtime_diag.get("value_exit_signal_count", 0)),
                f"{runtime_prefix}_value_exit_confirmed_count": int(runtime_diag.get("value_exit_confirmed_count", 0)),
                f"{runtime_prefix}_synthetic_entries": int(runtime_diag.get("synthetic_entries", 0)),
                f"{runtime_prefix}_synthetic_time_window_skips": int(runtime_diag.get("synthetic_time_window_skips", 0)),
                f"{runtime_prefix}_synthetic_exit_counts": dict(runtime_diag.get("synthetic_exit_counts", {}) or {}),
                f"{runtime_prefix}_synthetic_rule_exit_count": int(runtime_diag.get("synthetic_rule_exit_count", 0)),
                f"{runtime_prefix}_non_synthetic_exit_count": int(runtime_diag.get("non_synthetic_exit_count", 0)),
                f"{runtime_prefix}_learned_l3_exit_count": int(runtime_diag.get("learned_l3_exit_count", 0)),
                f"{runtime_prefix}_entry_count": int(runtime_diag.get("entries", 0)),
                f"{runtime_prefix}_soft_enter_hit_rate_early": float(
                    runtime_diag.get("soft_enter_hits_early", 0) / max(early_steps, 1)
                ),
                f"{runtime_prefix}_soft_enter_hit_rate_late": float(
                    runtime_diag.get("soft_enter_hits_late", 0) / max(late_steps, 1)
                ),
                f"{runtime_prefix}_soft_enter_thr_early_mean": float(thr_early_sum / max(early_steps, 1)),
                f"{runtime_prefix}_soft_enter_thr_late_mean": float(thr_late_sum / max(late_steps, 1)),
            }
        )
        for _ek, _pkey in (
            ("exit_prob_during_hold", "exit_prob_during_hold"),
            ("signal_flip_exit_prob_at_exit", "signal_flip_exit_prob_at_exit"),
        ):
            if isinstance((runtime_diag or {}).get(_ek), dict):
                out[f"{runtime_prefix}_{_pkey}"] = dict(runtime_diag[_ek])
    return out


def _combine_runtime_diagnostics(diag_by_symbol: dict[str, dict[str, Any]], symbols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "entries": int(sum(int((diag_by_symbol.get(sym) or {}).get("entries", 0)) for sym in symbols)),
        "policy_exit_count": int(sum(int((diag_by_symbol.get(sym) or {}).get("policy_exit_count", 0)) for sym in symbols)),
        "flip_exit_count": int(sum(int((diag_by_symbol.get(sym) or {}).get("flip_exit_count", 0)) for sym in symbols)),
        "deadline_exit_count": int(sum(int((diag_by_symbol.get(sym) or {}).get("deadline_exit_count", 0)) for sym in symbols)),
        "value_exit_signal_count": int(
            sum(int((diag_by_symbol.get(sym) or {}).get("value_exit_signal_count", 0)) for sym in symbols)
        ),
        "value_exit_confirmed_count": int(
            sum(int((diag_by_symbol.get(sym) or {}).get("value_exit_confirmed_count", 0)) for sym in symbols)
        ),
        "synthetic_entries": int(sum(int((diag_by_symbol.get(sym) or {}).get("synthetic_entries", 0)) for sym in symbols)),
        "synthetic_time_window_skips": int(
            sum(int((diag_by_symbol.get(sym) or {}).get("synthetic_time_window_skips", 0)) for sym in symbols)
        ),
        "synthetic_rule_exit_count": int(
            sum(int((diag_by_symbol.get(sym) or {}).get("synthetic_rule_exit_count", 0)) for sym in symbols)
        ),
        "non_synthetic_exit_count": int(
            sum(int((diag_by_symbol.get(sym) or {}).get("non_synthetic_exit_count", 0)) for sym in symbols)
        ),
        "learned_l3_exit_count": int(
            sum(int((diag_by_symbol.get(sym) or {}).get("learned_l3_exit_count", 0)) for sym in symbols)
        ),
        "soft_enter_hits_early": int(
            sum(int((diag_by_symbol.get(sym) or {}).get("soft_enter_hits_early", 0)) for sym in symbols)
        ),
        "soft_enter_hits_late": int(
            sum(int((diag_by_symbol.get(sym) or {}).get("soft_enter_hits_late", 0)) for sym in symbols)
        ),
        "steps_early": int(sum(int((diag_by_symbol.get(sym) or {}).get("steps_early", 0)) for sym in symbols)),
        "steps_late": int(sum(int((diag_by_symbol.get(sym) or {}).get("steps_late", 0)) for sym in symbols)),
        "soft_enter_thr_early_sum": float(
            sum(float((diag_by_symbol.get(sym) or {}).get("soft_enter_thr_early_sum", 0.0)) for sym in symbols)
        ),
        "soft_enter_thr_late_sum": float(
            sum(float((diag_by_symbol.get(sym) or {}).get("soft_enter_thr_late_sum", 0.0)) for sym in symbols)
        ),
        "synthetic_exit_counts": {},
    }
    sec: dict[str, int] = {}
    for sym in symbols:
        for k, v in dict((diag_by_symbol.get(sym) or {}).get("synthetic_exit_counts", {}) or {}).items():
            sec[str(k)] = int(sec.get(str(k), 0)) + int(v)
    out["synthetic_exit_counts"] = sec
    return out


def _trade_counts_by_symbol(trades: pd.DataFrame, symbols: list[str]) -> dict[str, int]:
    if trades is None or trades.empty or "symbol" not in trades.columns:
        return {sym: 0 for sym in symbols}
    vc = trades["symbol"].astype(str).value_counts()
    return {sym: int(vc.get(sym, 0)) for sym in symbols}


def build_portfolio_cap_reconciliation(
    combined_raw: pd.DataFrame,
    combined: pd.DataFrame,
    portfolio_cap_diag: dict[str, Any],
    *,
    symbols: list[str],
) -> dict[str, Any]:
    raw_counts = _trade_counts_by_symbol(combined_raw, symbols)
    kept_counts = _trade_counts_by_symbol(combined, symbols)
    skipped_counts = {
        sym: int(raw_counts.get(sym, 0)) - int(kept_counts.get(sym, 0))
        for sym in symbols
    }
    return {
        "cap_enabled": bool(portfolio_cap_diag.get("enabled", False)),
        "max_open_positions": int(portfolio_cap_diag.get("max_open_positions", 0)),
        "raw_total_trades": int(len(combined_raw)) if combined_raw is not None else 0,
        "post_cap_total_trades": int(len(combined)) if combined is not None else 0,
        "raw_counts_by_symbol": raw_counts,
        "post_cap_counts_by_symbol": kept_counts,
        "skipped_counts_by_symbol": skipped_counts,
        "runtime_counts_are_pre_cap": True,
        "metrics_and_curves_are_post_cap": bool(portfolio_cap_diag.get("enabled", False)),
        "artifacts": {
            "pre_cap_combined": "trades_ALL_pre_portfolio_cap.csv" if bool(portfolio_cap_diag.get("enabled", False)) else None,
            "post_cap_combined": "trades_ALL.csv",
            "pre_cap_by_symbol": {sym: f"trades_{sym}_pre_portfolio_cap.csv" for sym in symbols},
            "post_cap_by_symbol": {sym: f"trades_{sym}_post_portfolio_cap.csv" for sym in symbols},
            "legacy_by_symbol": {sym: f"trades_{sym}.csv" for sym in symbols},
            "legacy_by_symbol_semantics": "pre_portfolio_cap",
        },
    }


def summarize_exit_engine_diagnostics(diag_by_symbol: dict[str, dict[str, Any]], symbols: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, diag in [(sym, diag_by_symbol.get(sym, {}) or {}) for sym in symbols] + [
        ("COMBINED", _combine_runtime_diagnostics(diag_by_symbol, symbols))
    ]:
        synthetic_rule_exits = int(diag.get("synthetic_rule_exit_count", 0))
        learned_l3_exits = int(diag.get("learned_l3_exit_count", 0))
        non_synthetic_exits = int(diag.get("non_synthetic_exit_count", 0))
        if synthetic_rule_exits > 0 and learned_l3_exits == 0:
            engine = "synthetic_rules"
        elif synthetic_rule_exits > 0 and learned_l3_exits > 0:
            engine = "mixed_synthetic_rules_and_l3"
        elif learned_l3_exits > 0:
            engine = "learned_l3"
        elif non_synthetic_exits > 0:
            engine = "non_synthetic_non_l3"
        else:
            engine = "none"
        out.append(
            {
                "label": label,
                "strategy_router": _oos_strategy_router(),
                "effective_exit_engine": engine,
                "synthetic_rule_exits": synthetic_rule_exits,
                "learned_l3_exits": learned_l3_exits,
                "non_synthetic_exits": non_synthetic_exits,
                "synthetic_entries": int(diag.get("synthetic_entries", 0)),
                "total_entries": int(diag.get("entries", 0)),
                "note": (
                    "Synthetic vol-regime strategies use rule exits and bypass the learned L3 exit branch."
                    if synthetic_rule_exits > 0
                    else ""
                ),
            }
        )
    return out


def _oos_json_float(x: float) -> float | None:
    """Finite floats for JSON; None if missing or non-finite (avoids NaN in summaries)."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if np.isfinite(xf) else None


def summarize_exit_driver_table(trades: pd.DataFrame, *, label: str) -> dict[str, Any]:
    t = trades if trades is not None else pd.DataFrame()
    d = _ensure_primary_exit_driver_series(t)
    out: dict[str, Any] = {"label": label, "by_driver": []}
    if t.empty or len(d) != len(t):
        return out
    t2 = t.copy()
    t2["_driver"] = d
    rets = pd.to_numeric(t2["return"], errors="coerce")
    t2["_r"] = rets
    tot_abs = float(np.nansum(np.abs(t2["_r"].to_numpy(dtype=np.float64))))
    for drv, g in t2.groupby("_driver", sort=True):
        rr = g["_r"].to_numpy(dtype=np.float64)
        rr = rr[np.isfinite(rr)]
        if not rr.size:
            continue
        s = float(np.sum(rr))
        out["by_driver"].append(
            {
                "primary_exit_driver": str(drv),
                "n": int(len(g)),
                "share_n": float(len(g) / max(len(t2), 1)),
                "sum_return_frac": _oos_json_float(s),
                "mean_return_frac": _oos_json_float(float(np.mean(rr))),
                "share_of_abs_pnl": _oos_json_float((float(np.nansum(np.abs(rr))) / tot_abs) if tot_abs > 0 else None),
            }
        )
    return out


def filter_trades_for_cumulative_plot_mode(t: pd.DataFrame, mode: str) -> pd.DataFrame:
    if t is None or t.empty or mode in {"", "all"}:
        return t.copy() if t is not None and not t.empty else pd.DataFrame()
    d = _ensure_primary_exit_driver_series(t)
    t2 = t.copy()
    t2["_driver"] = d
    if mode == "l3_learned":
        return t2[t2["_driver"].isin(["l3_learned", "l3_mtm_adaptive"])].drop(columns=["_driver"], errors="ignore").reset_index(
            drop=True
        )
    if mode == "synthetic_rules":
        return t2[t2["_driver"] == "synthetic_rules"].drop(columns=["_driver"], errors="ignore").reset_index(drop=True)
    return t.copy()


def build_exit_driver_metric_slices(
    combined: pd.DataFrame,
    symbols: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Per-symbol + COMBINED trade-return summaries restricted to a single primary_exit_driver."""
    out: dict[str, list[dict[str, Any]]] = {
        "l3_learned_only": [],
        "synthetic_rules_only": [],
    }
    for sym in symbols:
        part = (
            combined[combined["symbol"] == sym]
            if (combined is not None and not combined.empty and "symbol" in combined.columns)
            else pd.DataFrame()
        )
        for mode, key in (("l3_learned", "l3_learned_only"), ("synthetic_rules", "synthetic_rules_only")):
            flt = filter_trades_for_cumulative_plot_mode(part, mode)
            out[key].append(summarize_trade_returns(flt, label=f"{sym}__{mode}"))
    for mode, key in (("l3_learned", "l3_learned_only"), ("synthetic_rules", "synthetic_rules_only")):
        flt = filter_trades_for_cumulative_plot_mode(
            combined if combined is not None else pd.DataFrame(), mode
        )
        out[key].append(summarize_trade_returns(flt, label=f"COMBINED__{mode}"))
    return out


def _print_backtest_interpretation(block: dict[str, Any]) -> None:
    if not block:
        return
    print("\n" + "=" * 70)
    print("  OOS 解读 (what headline metrics measure)")
    print("=" * 70)
    print(f"  router={block.get('strategy_router')}  L3 exit mode={block.get('oos_l3_exit_mode')}", flush=True)
    print(f"  {block.get('router_summary', '')}", flush=True)
    sh = block.get("share_of_trades_by_exit_driver") or {}
    if sh:
        parts = [f"{k}={100.0 * float(v):.1f}%" for k, v in sorted(sh.items()) if v is not None and float(v) > 0]
        if parts:
            print("  share of trades by primary_exit_driver: " + "  ".join(parts), flush=True)
    for ln in block.get("reading_guide") or []:
        print(f"  - {ln}", flush=True)


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


def summarize_strategy_diagnostics(trades: pd.DataFrame, *, label: str) -> dict[str, Any]:
    t = trades if trades is not None else pd.DataFrame()
    out = {"label": label, "strategy_stats": []}
    if t.empty or "strategy_type" not in t.columns:
        return out
    stats = []
    for st, grp in t.groupby("strategy_type"):
        ret = pd.to_numeric(grp["return"], errors="coerce").to_numpy(dtype=np.float64)
        ret = ret[np.isfinite(ret)]
        if ret.size == 0:
            continue
        wins = ret[ret > 0.0]
        losses = ret[ret < 0.0]
        gross_loss = float(np.sum(losses))
        pf = float(np.sum(wins) / (-gross_loss)) if gross_loss < 0.0 else float("nan")
        hedge_pnl = pd.to_numeric(grp.get("hedge_pnl"), errors="coerce") if "hedge_pnl" in grp.columns else pd.Series(dtype=float)
        hedge_count = pd.to_numeric(grp.get("hedge_count"), errors="coerce") if "hedge_count" in grp.columns else pd.Series(dtype=float)
        stats.append(
            {
                "strategy_type": str(st),
                "n": int(len(grp)),
                "share": float(len(grp) / max(len(t), 1)),
                "mean_return_frac": _oos_json_float(float(np.mean(ret))),
                "win_rate": float(np.mean(ret > 0.0)),
                "profit_factor": _oos_json_float(pf),
                "p01_return_frac": _oos_json_float(float(np.nanquantile(ret, 0.01))),
                "p05_return_frac": _oos_json_float(float(np.nanquantile(ret, 0.05))),
                "min_return_frac": _oos_json_float(float(np.nanmin(ret))),
                "avg_holding_bars": _oos_json_float(float(pd.to_numeric(grp.get("holding_bars"), errors="coerce").mean())),
                "avg_hedge_pnl": _oos_json_float(float(hedge_pnl.mean())) if not hedge_pnl.empty else None,
                "avg_hedge_count": _oos_json_float(float(hedge_count.mean())) if not hedge_count.empty else None,
            }
        )
    out["strategy_stats"] = sorted(stats, key=lambda x: x["strategy_type"])
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
    ret_gap_thr = 0.00012
    sharpe_gap_thr = 0.80
    win_gap_thr = 0.08
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


def _write_oos_summary_file(RESULTS_DIR: Path, payload: dict[str, Any]) -> None:
    path = RESULTS_DIR / "oos_summary.txt"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  [OOS] saved summary -> {path.name}", flush=True)


def _print_oos_summary(blocks: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 70)
    print("  OOS performance (simple cumulative return on closed trades by exit_time)")
    print("=" * 70)
    for row in blocks:
        if int(row.get("n_trades", 0)) == 0:
            print(f"\n  [{row.get('label')}] no trades")
            continue
        print(f"\n  [{row.get('label')}]  trades={row['n_trades']}  {row.get('first_exit', '')} .. {row.get('last_exit', '')}")
        print(
            f"    cumulative_return={row['total_return_pct']:.2f}%  max_dd={row['max_drawdown_pct']:.2f}%",
            flush=True,
        )
        print(
            f"    win_rate={row['win_rate']:.2%}  Sharpe~{row['sharpe_trade_annualized']:.3f}  "
            f"Sortino~{row['sortino_trade_annualized']:.3f}  profit_factor={row['profit_factor']:.3f}",
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
        runtime_prefix = "runtime_pre_cap" if "runtime_pre_cap_entry_count" in row else "runtime"
        if f"{runtime_prefix}_entry_count" in row or f"{runtime_prefix}_synthetic_entries" in row:
            print(
                f"    {runtime_prefix}: entries={int(row.get(f'{runtime_prefix}_entry_count', 0))}  "
                f"synthetic_entries={int(row.get(f'{runtime_prefix}_synthetic_entries', 0))}  "
                f"synthetic_rule_exits={int(row.get(f'{runtime_prefix}_synthetic_rule_exit_count', 0))}  "
                f"learned_l3_exits={int(row.get(f'{runtime_prefix}_learned_l3_exit_count', 0))}",
                flush=True,
            )
            print(
                f"    soft_enter_hit_rate early={100.0 * float(row.get(f'{runtime_prefix}_soft_enter_hit_rate_early', 0.0)):.1f}%  "
                f"late={100.0 * float(row.get(f'{runtime_prefix}_soft_enter_hit_rate_late', 0.0)):.1f}%  "
                f"thr_mean early={float(row.get(f'{runtime_prefix}_soft_enter_thr_early_mean', float('nan'))):.3f}  "
                f"late={float(row.get(f'{runtime_prefix}_soft_enter_thr_late_mean', float('nan'))):.3f}",
                flush=True,
            )
        for _desc, _sk in (
            ("exit_prob all hold bars (calibrated, L3 path)", f"{runtime_prefix}_exit_prob_during_hold"),
            ("exit_prob at Signal_Flip exit bar (exit row value)", f"{runtime_prefix}_signal_flip_exit_prob_at_exit"),
        ):
            _ed = row.get(_sk)
            if not isinstance(_ed, dict) or not int(_ed.get("n", 0) or 0):
                continue
            print(
                f"    {_desc}:  n={_ed['n']}  mean={_ed.get('mean')}  p75={_ed.get('p75')}  "
                f"p90={_ed.get('p90')}  p95={_ed.get('p95')}  max={_ed.get('max')}",
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


def _print_strategy_diagnostics(blocks: list[dict[str, Any]]) -> None:
    if not blocks:
        return
    print("\n" + "=" * 70)
    print("  Strategy diagnostics")
    print("=" * 70)
    for row in blocks:
        stats = row.get("strategy_stats") or []
        if not stats:
            continue
        print(f"\n  [{row.get('label')}] strategy decomposition:", flush=True)
        for it in stats:
            mr = it.get("mean_return_frac")
            p05 = it.get("p05_return_frac")
            p01 = it.get("p01_return_frac")
            mn = it.get("min_return_frac")
            print(
                f"    {it['strategy_type']}  n={int(it['n'])}  share={100.0*float(it['share']):.1f}%  "
                f"mean={100.0*float(mr):.4f}%  win={100.0*float(it['win_rate']):.1f}%  "
                f"pf={float(it['profit_factor']) if it.get('profit_factor') is not None else float('nan'):.3f}  "
                f"p05={100.0*float(p05):.3f}%  p01={100.0*float(p01):.3f}%  "
                f"min={100.0*float(mn):.3f}%  avg_hold={float(it['avg_holding_bars']):.1f}",
                flush=True,
            )


def main():
    print("=" * 70, flush=True)
    print("  Running OOS Dual-View Pipeline (L1a -> L1b -> L2 -> L3)", flush=True)
    if _oos_l2l3_stack_preset_active():
        print(
            "  [preset] OOS_L2L3_STACK: L2 入场 + 本脚本内固定基线 (straddle + mtm_adaptive, 见 OOS_BASELINE_*)，"
            "不再通过环境变量改 router/exit。",
            flush=True,
        )
    _oem = _oos_l3_exit_mode()
    _blocked = sorted(_oos_block_entry_l1a_regime_ids())
    _reg_s = "[]" if not _blocked else str(_blocked)
    print(
        f"  L3 exit (code baseline): {_oem}  (mtm_adaptive: MTM / vote; "
        f"value / ATR 等其它开关仍见各自 OOS_* env 与本文件首段说明)",
        flush=True,
    )
    _allow_e = (os.environ.get("OOS_ALLOW_ENTRY_L1A_REGIMES") or "").strip()
    _block_e = (os.environ.get("OOS_BLOCK_ENTRY_L1A_REGIMES") or "").strip()
    if _allow_e or _block_e:
        _rb_src = (
            f"OOS_ALLOW_ENTRY_L1A_REGIMES={_allow_e!r} "
            if _allow_e
            else f"OOS_BLOCK_ENTRY_L1A_REGIMES={_block_e!r} "
        )
    else:
        _rb_src = "OOS_BASELINE_BLOCK_ENTRY_L1A_REGIMES (code) "
    print(
        f"  Regime block at entry: {_rb_src}→ blocked L1a ids = {_reg_s}"
        if _blocked
        else f"  Regime block at entry: {_rb_src}→ (none blocked)",
        flush=True,
    )
    print(
        f"  Strategy router (code): OOS_BASELINE_STRATEGY_ROUTER={_oos_strategy_router()}",
        flush=True,
    )
    _p_cap = _oos_portfolio_max_open_positions()
    print(
        f"  Portfolio cap (code): OOS_BASELINE_PORTFOLIO_MAX_OPEN_POSITIONS={_p_cap}  (0 = no post-sim overlap cap)",
        flush=True,
    )
    _cpm_env = (os.environ.get("OOS_CUMULATIVE_PLOT_MODE") or "all").strip() or "all"
    try:
        _cpm_d = _oos_cumulative_plot_mode()
    except ValueError as _e_cpm:
        _cpm_d = "all"
        print(f"  [OOS][WARN] OOS_CUMULATIVE_PLOT_MODE: {_e_cpm}  (using all)", flush=True)
    print(
        f"  Cumulative plot: OOS_CUMULATIVE_PLOT_MODE={_cpm_d}  (env raw={_cpm_env!r}; all | l3_learned | synthetic_rules)",
        flush=True,
    )
    print(f"  OOS window: [{OOS_START}, {OOS_END})  (override via env OOS_START / OOS_END)", flush=True)
    if (os.environ.get("OOS_CHART_TITLE") or "").strip():
        print("  Chart title: OOS_CHART_TITLE is set (see _oos_figure_title format keys).", flush=True)
    print(f"  Results dir: {RESULTS_DIR.resolve()}  (override via OOS_RESULTS_DIR)", flush=True)
    try:
        _syms_preview = _oos_symbols_from_env()
        print(
            f"  Symbols: OOS_SYMBOLS={','.join(_syms_preview)}  (default QQQ,SPY if unset)",
            flush=True,
        )
    except ValueError as _e_sym:
        print(f"  [OOS][ERROR] {_e_sym}", flush=True)
        raise
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
    _thr_env = (os.environ.get("OOS_L3_EXIT_PROB_THRESHOLD") or "").strip()
    if _thr_env:
        l3_meta = {**dict(l3_meta), "l3_exit_prob_threshold": float(_thr_env)}
        print(
            f"  [OOS] OOS_L3_EXIT_PROB_THRESHOLD={float(_thr_env):.4f}  "
            f"(overrides l3 meta default; per-state l3_exit_policy_by_state[...].exit_prob_threshold still wins if set)",
            flush=True,
        )
    if _oem.lower() in ("mtm_adaptive", "mtm_momentum", "adaptive_mtm", "adaptive"):
        print(
            "  [OOS] mtm_adaptive: exit uses OOS_MTM_* (OOS_MTM_FAST_WINDOW, OOS_MTM_MIN_HOLD, OOS_MTM_VOTE_K, ...); "
            "fixed OOS_L3_EXIT_PROB_THRESHOLD does not apply to the exit decision (logging / fallback only).",
            flush=True,
        )
    l3_enc, l3_tcfg = load_l3_trajectory_encoder_for_infer(l3_meta)
    if l3_enc is not None:
        l3_enc = l3_enc.to(TORCH_DEVICE)
    print("  Artifacts loaded. Starting per-symbol inference + backtest.", flush=True)
    symbols = _oos_symbols_from_env()
    all_trades = []
    plot_tasks: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    diag_by_symbol: dict[str, dict[str, Any]] = {}
    for sym in symbols:
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
            tr_df.to_csv(RESULTS_DIR / f"trades_{sym}_pre_portfolio_cap.csv", index=False)
            tr_df.to_csv(RESULTS_DIR / f"trades_{sym}.csv", index=False)
            print(f"[{sym}] generated {len(tr_df)} trades.", flush=True)
        else:
            print(f"[{sym}] generated 0 trades.", flush=True)
    _pooled_exit_prob = _finalize_per_symbol_exit_prob_lists(diag_by_symbol, symbols)
    summary_blocks: list[dict[str, Any]] = []
    exit_diag_blocks_pre_cap: list[dict[str, Any]] = []
    exit_diag_blocks_post_cap: list[dict[str, Any]] = []
    regime_diag_blocks: list[dict[str, Any]] = []
    strategy_diag_blocks: list[dict[str, Any]] = []
    exit_engine_blocks: list[dict[str, Any]] = []
    portfolio_cap_reconciliation: dict[str, Any] = {}
    portfolio_cap_diag: dict[str, Any] = {
        "enabled": False,
        "max_open_positions": int(_oos_portfolio_max_open_positions()),
        "skipped_trades": 0,
    }
    trades_for_plot_by_symbol: dict[str, pd.DataFrame] = {}
    if all_trades:
        combined_raw = pd.concat(all_trades, ignore_index=True)
        for sym in symbols:
            raw_part = combined_raw[combined_raw["symbol"] == sym] if "symbol" in combined_raw.columns else pd.DataFrame()
            if not raw_part.empty:
                exit_diag_blocks_pre_cap.append(
                    summarize_exit_path_diagnostics(
                        raw_part,
                        diag_by_symbol.get(sym, {}),
                        label=sym,
                        runtime_prefix="runtime_pre_cap",
                    )
                )
        combined_raw_diag = _combine_runtime_diagnostics(diag_by_symbol, symbols)
        combined_raw_diag["exit_prob_during_hold"] = _pooled_exit_prob.get("exit_prob_during_hold")
        combined_raw_diag["signal_flip_exit_prob_at_exit"] = _pooled_exit_prob.get("signal_flip_exit_prob_at_exit")
        exit_diag_blocks_pre_cap.append(
            summarize_exit_path_diagnostics(
                combined_raw,
                combined_raw_diag,
                label="COMBINED",
                runtime_prefix="runtime_pre_cap",
            )
        )
        portfolio_cap_max_open = _oos_portfolio_max_open_positions()
        combined, portfolio_cap_diag = apply_portfolio_open_position_cap(
            combined_raw,
            max_open_positions=portfolio_cap_max_open,
        )
        if bool(portfolio_cap_diag.get("enabled", False)):
            combined_raw.to_csv(RESULTS_DIR / "trades_ALL_pre_portfolio_cap.csv", index=False)
            skipped = int(portfolio_cap_diag.get("skipped_trades", 0))
            kept = int(portfolio_cap_diag.get("kept_trades", len(combined)))
            print(
                f"[PORTFOLIO CAP] max_open={portfolio_cap_max_open} kept={kept} skipped={skipped}",
                flush=True,
            )
        combined.to_csv(RESULTS_DIR / "trades_ALL.csv", index=False)
        portfolio_cap_reconciliation = build_portfolio_cap_reconciliation(
            combined_raw,
            combined,
            portfolio_cap_diag,
            symbols=symbols,
        )
        print(f"\nTotal trades: {len(combined)}")
        print(f"Win Rate: {(combined['return'] > 0).mean():.2%}")
        print(f"Avg Return: {combined['return'].mean():.4%}")
        for sym in symbols:
            part = combined[combined["symbol"] == sym] if "symbol" in combined.columns else pd.DataFrame()
            trades_for_plot_by_symbol[sym] = part.copy()
            part.to_csv(RESULTS_DIR / f"trades_{sym}_post_portfolio_cap.csv", index=False)
            if not part.empty:
                summary_blocks.append(summarize_trade_returns(part, label=sym))
                exit_diag_blocks_post_cap.append(
                    summarize_exit_path_diagnostics(part, None, label=sym, runtime_prefix=None)
                )
                regime_diag_blocks.append(summarize_regime_diagnostics(part, label=sym))
                strategy_diag_blocks.append(summarize_strategy_diagnostics(part, label=sym))
        summary_blocks.append(summarize_trade_returns(combined, label="COMBINED"))
        exit_diag_blocks_post_cap.append(summarize_exit_path_diagnostics(combined, None, label="COMBINED", runtime_prefix=None))
        regime_diag_blocks.append(summarize_regime_diagnostics(combined, label="COMBINED"))
        strategy_diag_blocks.append(summarize_strategy_diagnostics(combined, label="COMBINED"))
        exit_engine_blocks = summarize_exit_engine_diagnostics(diag_by_symbol, symbols)
        try:
            oem = _oos_l3_exit_mode()
        except ValueError:
            oem = "l3"
        m_atr, m_min = _oos_atr_trailing_params()
        exit_driver_table_blocks: list[dict[str, Any]] = []
        for sym_ in symbols:
            _part_ = combined[combined["symbol"] == sym_] if "symbol" in combined.columns else pd.DataFrame()
            if not _part_.empty:
                exit_driver_table_blocks.append(summarize_exit_driver_table(_part_, label=sym_))
        exit_driver_table_blocks.append(summarize_exit_driver_table(combined, label="COMBINED"))
        backtest_interpretation = build_backtest_interpretation(
            strategy_router=_oos_strategy_router(),
            oos_l3_exit_mode=str(oem),
            combined=combined,
        )
        driver_metric_slices = build_exit_driver_metric_slices(combined, symbols)
        symbol_guard = evaluate_symbol_asymmetry_guardrail(summary_blocks)
        _print_oos_summary(summary_blocks)
        _print_backtest_interpretation(backtest_interpretation)
        _print_exit_path_diagnostics(exit_diag_blocks_post_cap)
        _print_exit_path_diagnostics(exit_diag_blocks_pre_cap)
        _print_regime_diagnostics(regime_diag_blocks)
        _print_strategy_diagnostics(strategy_diag_blocks)
        if bool(symbol_guard.get("flagged", False)):
            print(
                "\n  [L3][WARN][SYMBOL_GUARD] asymmetry guard flagged: "
                + ", ".join(symbol_guard.get("reasons", [])),
                flush=True,
            )
        eq_rows: list[dict[str, Any]] = []
        if "symbol" in combined.columns:
            _eq_pairs: list[tuple[str, pd.DataFrame]] = [
                (s, combined[combined["symbol"] == s]) for s in symbols
            ] + [("COMBINED", combined)]
        else:
            _eq_pairs = [("COMBINED", combined)]
        for label, df_sub in _eq_pairs:
            if df_sub.empty:
                continue
            ts = df_sub.sort_values("exit_time").reset_index(drop=True)
            r = ts["return"].to_numpy(dtype=np.float64)
            cum_ret = np.cumsum(r)
            for i in range(len(ts)):
                eq_rows.append(
                    {
                        "curve": label,
                        "i": i + 1,
                        "exit_time": ts["exit_time"].iloc[i],
                        "return_frac": float(r[i]),
                        "cumulative_return_frac": float(cum_ret[i]),
                        "cumulative_return_pct": float(100.0 * cum_ret[i]),
                    }
                )
        if eq_rows:
            pd.DataFrame(eq_rows).to_csv(RESULTS_DIR / "oos_cumulative_returns.csv", index=False)
        try:
            _cpm = _oos_cumulative_plot_mode()
        except ValueError:
            _cpm = "all"
        payload = {
            "oos_start": OOS_START,
            "oos_end": OOS_END,
            "backtest_config": {
                "OOS_MTM_STATE_HOLD_FRAC": float(OOS_BASELINE_MTM_STATE_HOLD_FRAC),
                "OOS_L3_EXIT_MODE": oem,
                "OOS_L3_VALUE_EXIT_UNREAL_FRAC": _oos_l3_value_exit_unreal_frac(),
                "OOS_L3_VALUE_EXIT_MIN_HOLD": _oos_l3_value_exit_min_hold(),
                "OOS_L3_VALUE_EXIT_CONFIRM_BARS": _oos_l3_value_exit_confirm_bars(),
                "OOS_ATR_TRAIL_MULT": m_atr,
                "OOS_ATR_TRAIL_MIN_HOLD": m_min,
                "OOS_BLOCK_ENTRY_L1A_REGIMES": sorted(_oos_block_entry_l1a_regime_ids()),
                "OOS_PORTFOLIO_MAX_OPEN_POSITIONS": _oos_portfolio_max_open_positions(),
                "OOS_STRATEGY_ROUTER": _oos_strategy_router(),
                "OOS_L2L3_STACK": bool(_oos_l2l3_stack_preset_active()),
                "OOS_CUMULATIVE_PLOT_MODE": _cpm,
                "OOS_VOL_REGIME_ROUTER_MAP": {
                    "0": "SHORT_STRADDLE_R0",
                    "1": "LONG_STRADDLE_R1",
                    "2": "GAMMA_SCALP_R2",
                    "3": "SHORT_STRADDLE_R3",
                    "4": "IRON_CONDOR_R4",
                },
            },
            "backtest_interpretation": backtest_interpretation,
            "exit_driver_attribution": exit_driver_table_blocks,
            "metrics_by_exit_driver_slice": driver_metric_slices,
            "metrics": summary_blocks,
            "exit_path_diagnostics": exit_diag_blocks_post_cap,
            "exit_path_diagnostics_pre_cap": exit_diag_blocks_pre_cap,
            "exit_path_diagnostics_post_cap": exit_diag_blocks_post_cap,
            "regime_diagnostics": regime_diag_blocks,
            "strategy_diagnostics": strategy_diag_blocks,
            "exit_engine_diagnostics": exit_engine_blocks,
            "symbol_asymmetry_guard": symbol_guard,
            "portfolio_cap": portfolio_cap_diag,
            "portfolio_cap_reconciliation": portfolio_cap_reconciliation,
        }
        bar_exit_payload: list[dict[str, Any]] = []
        for sym in symbols:
            _bd = (diag_by_symbol.get(sym) or {}).get("bar_exit_diagnostics")
            if isinstance(_bd, dict) and int(_bd.get("n_trades", 0) or 0) > 0:
                bar_exit_payload.append(_bd)
        comb_be = BarExitDiagnostics(default_bar_exit_track_bars())
        for sym in symbols:
            for _rec in (diag_by_symbol.get(sym) or {}).get("bar_exit_records") or []:
                if isinstance(_rec, dict):
                    comb_be.record_trade(
                        int(_rec["trade_id"]),
                        float(_rec["entry_price"]),
                        side=int(_rec.get("side", 1)),
                        entry_regime=int(_rec.get("entry_regime", -1)),
                        actual_exit_bar=int(_rec.get("actual_exit_bar", -1)),
                        bar_prices=_rec.get("bar_prices"),
                    )
        if comb_be.trades:
            bar_exit_payload.append(comb_be.report("COMBINED"))
        payload["bar_exit_diagnostics"] = bar_exit_payload
        _write_oos_summary_file(RESULTS_DIR, payload)
    else:
        m_atr, m_min = _oos_atr_trailing_params()
        try:
            oem = _oos_l3_exit_mode()
        except ValueError:
            oem = "l3"
        try:
            _ce = _oos_cumulative_plot_mode()
        except ValueError:
            _ce = "all"
        _write_oos_summary_file(
            RESULTS_DIR,
            {
                "oos_start": OOS_START,
                "oos_end": OOS_END,
                "backtest_config": {
                    "OOS_MTM_STATE_HOLD_FRAC": float(OOS_BASELINE_MTM_STATE_HOLD_FRAC),
                    "OOS_L3_EXIT_MODE": oem,
                    "OOS_L3_VALUE_EXIT_UNREAL_FRAC": _oos_l3_value_exit_unreal_frac(),
                    "OOS_L3_VALUE_EXIT_MIN_HOLD": _oos_l3_value_exit_min_hold(),
                    "OOS_L3_VALUE_EXIT_CONFIRM_BARS": _oos_l3_value_exit_confirm_bars(),
                    "OOS_ATR_TRAIL_MULT": m_atr,
                    "OOS_ATR_TRAIL_MIN_HOLD": m_min,
                    "OOS_BLOCK_ENTRY_L1A_REGIMES": sorted(_oos_block_entry_l1a_regime_ids()),
                    "OOS_PORTFOLIO_MAX_OPEN_POSITIONS": _oos_portfolio_max_open_positions(),
                    "OOS_STRATEGY_ROUTER": _oos_strategy_router(),
                    "OOS_L2L3_STACK": bool(_oos_l2l3_stack_preset_active()),
                    "OOS_CUMULATIVE_PLOT_MODE": _ce,
                },
                "backtest_interpretation": build_backtest_interpretation(
                    strategy_router=_oos_strategy_router(),
                    oos_l3_exit_mode=str(oem),
                    combined=pd.DataFrame(),
                ),
                "exit_driver_attribution": [],
                "metrics_by_exit_driver_slice": {
                    "l3_learned_only": [],
                    "synthetic_rules_only": [],
                },
                "metrics": [],
                "exit_path_diagnostics": [],
                "exit_path_diagnostics_pre_cap": [],
                "exit_path_diagnostics_post_cap": [],
                "regime_diagnostics": [],
                "strategy_diagnostics": [],
                "exit_engine_diagnostics": [],
                "bar_exit_diagnostics": [],
                "symbol_asymmetry_guard": {"enabled": True, "flagged": False, "reasons": []},
                "portfolio_cap": portfolio_cap_diag,
                "portfolio_cap_reconciliation": {},
            },
        )

    metrics_by = {str(b["label"]): b for b in summary_blocks}
    try:
        _cpm_plot = _oos_cumulative_plot_mode()
    except ValueError:
        _cpm_plot = "all"
    for sym, price_df, tr_df in plot_tasks:
        tr_base = trades_for_plot_by_symbol.get(sym, tr_df)
        tr_plot = filter_trades_for_cumulative_plot_mode(tr_base, _cpm_plot)
        m_plot = (
            summarize_trade_returns(tr_plot, label=sym)
            if tr_plot is not None and not tr_plot.empty
            else {"label": sym, "n_trades": 0}
        )
        _tsub: str | None = None
        if _cpm_plot == "l3_learned":
            _tsub = f"累计曲线: 仅 L3 分类器平仓 (OOS_CUMULATIVE_PLOT_MODE={_cpm_plot}, n={int(m_plot.get('n_trades', 0) or 0)})"
        elif _cpm_plot == "synthetic_rules":
            _tsub = f"累计曲线: 仅合成规则平仓 (OOS_CUMULATIVE_PLOT_MODE={_cpm_plot}, n={int(m_plot.get('n_trades', 0) or 0)})"
        _op = (
            RESULTS_DIR / f"oos_chart_{sym}.png"
            if _cpm_plot == "all"
            else RESULTS_DIR / f"oos_chart_{sym}_{_cpm_plot}.png"
        )
        plot_oos_price_and_cumulative_return(
            sym, price_df, tr_plot, m_plot, _op, title_subtitle=_tsub
        )


if __name__ == "__main__":
    main()
