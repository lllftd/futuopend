"""MTM momentum / multi-signal adaptive exit (OOS), no fixed exit-prob threshold.

See OOS_L3_EXIT_MODE=mtm_adaptive and env OOS_MTM_* in backtests/oos_backtest.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_REGIME_PROFILES: dict[int, dict[str, float]] = {
    0: {"median_best": 18.0, "p75_best": 24.0, "decay_rate": 0.04},
    1: {"median_best": 19.0, "p75_best": 25.0, "decay_rate": 0.03},
    2: {"median_best": 17.0, "p75_best": 23.0, "decay_rate": 0.03},
    3: {"median_best": 15.0, "p75_best": 21.0, "decay_rate": 0.05},
    4: {"median_best": 15.0, "p75_best": 20.0, "decay_rate": 0.05},
}

DEFAULT_P90_HOLD: dict[int, int] = {
    0: 35,
    1: 40,
    2: 45,
    3: 30,
    4: 30,
}


@dataclass
class AdaptiveMtmConfig:
    mtm_fast_window: int = 2
    mtm_slow_window: int = 5
    min_hold_bars: int = 6
    vote_k: int = 2
    vote_k_profit: int = 2
    vote_k_loss: int = 1
    z_score_hi: float = 1.0
    z_score_trend_hi: float = 0.7
    zscore_min_history: int = 5
    zscore_level_enabled: bool = True
    zscore_level_threshold: float = 0.38
    enable_momentum: bool = True
    enable_drawdown: bool = True
    enable_time: bool = True
    enable_model: bool = True
    enable_underwater: bool = True
    history_max_bars: int = 2000
    regime_profiles: dict[int, dict[str, float]] = field(default_factory=lambda: dict(DEFAULT_REGIME_PROFILES))
    p90_hold_by_regime: dict[int, int] = field(default_factory=lambda: dict(DEFAULT_P90_HOLD))
    time_urgency_active_thr: float = 0.6
    time_early_frac: float = 0.50
    safety_net_use_regime_p90: bool = True
    dd_use_fractional: bool = True
    dd_frac_tolerance_profit: float = 0.35
    dd_frac_tolerance_loss: float = 0.50
    dd_abs_tolerance_loss: float = 0.15
    dd_base_tolerance_profit: float = 0.20
    dd_profit_scale: float = 0.5
    dd_max_tolerance: float = 0.40
    dd_tolerance_loss: float = 0.15
    underwater_enabled: bool = True
    underwater_threshold: float = -0.02
    underwater_streak_bars: int = 8
    underwater_deep_threshold: float = -0.10
    underwater_deep_streak_bars: int = 5
    regime_vote_k_loss_override: dict[int, int] = field(default_factory=lambda: {2: 1, 3: 1})
    regime_min_hold_override: dict[int, int] = field(default_factory=lambda: {0: 8, 4: 8})
    regime_underwater_streak_override: dict[int, int] = field(default_factory=lambda: {2: 6, 3: 5})


def _merge_regime_json(
    path: str, base_profiles: dict[int, dict[str, float]], base_p90: dict[int, int]
) -> tuple[dict[int, dict[str, float]], dict[int, int]]:
    p = (path or "").strip()
    if not p or not os.path.isfile(p):
        return base_profiles, base_p90
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    prof = {
        int(k): {kk: float(vv) for kk, vv in (v or {}).items()}
        for k, v in (raw.get("regime_profiles") or raw.get("profiles") or {}).items()
    }
    p90 = {int(k): int(v) for k, v in (raw.get("p90_hold") or raw.get("p90_hold_by_regime") or {}).items()}
    return {**base_profiles, **prof}, {**base_p90, **p90}


def load_adaptive_mtm_config_from_env() -> AdaptiveMtmConfig:
    c = AdaptiveMtmConfig()
    fast = int(np.clip(int((os.environ.get("OOS_MTM_FAST_WINDOW") or str(c.mtm_fast_window)).strip() or "2"), 1, 500))
    slow = int(np.clip(int((os.environ.get("OOS_MTM_SLOW_WINDOW") or str(c.mtm_slow_window)).strip() or "5"), 2, 500))
    if slow <= fast:
        slow = fast + 1
    min_h = int(np.clip(int((os.environ.get("OOS_MTM_MIN_HOLD") or str(c.min_hold_bars)).strip() or "6"), 0, 10_000))
    v_def = c.vote_k
    v_profit = int(
        np.clip(
            int((os.environ.get("OOS_MTM_VOTE_K_PROFIT") or os.environ.get("OOS_MTM_VOTE_K") or str(c.vote_k_profit)).strip() or "2"),
            1,
            6,
        )
    )
    v_loss = int(np.clip(int((os.environ.get("OOS_MTM_VOTE_K_LOSS") or str(c.vote_k_loss)).strip() or "1"), 1, 6))
    z_hi = float(np.clip(float((os.environ.get("OOS_MTM_ZSCORE") or str(c.z_score_hi)).strip() or "1.0"), 0.1, 10.0))
    z_tr = float(np.clip(float((os.environ.get("OOS_MTM_ZSCORE_TREND") or str(c.z_score_trend_hi)).strip() or "0.7"), 0.1, 10.0))
    zmin = int(np.clip(int((os.environ.get("OOS_MTM_ZSCORE_MIN_HIST") or str(c.zscore_min_history)).strip() or "5"), 2, 200))
    z_lvl = float(np.clip(float((os.environ.get("OOS_MTM_PROB_LEVEL") or str(c.zscore_level_threshold)).strip() or "0.38"), 0.0, 1.0))
    hmax = int(np.clip(int((os.environ.get("OOS_MTM_HISTORY_MAX_BARS") or "2000").strip() or "2000"), 32, 50_000))
    u_thr = float(np.clip(float((os.environ.get("OOS_MTM_TIME_URGENCY_THR") or str(c.time_urgency_active_thr)).strip() or "0.6"), 0.0, 1.0))
    t_early = float(np.clip(float((os.environ.get("OOS_MTM_TIME_EARLY_FRAC") or str(c.time_early_frac)).strip() or "0.5"), 0.1, 0.95))
    jpath = (os.environ.get("OOS_MTM_REGIME_PROFILES_JSON") or "").strip()
    prof, p90 = _merge_regime_json(jpath, dict(DEFAULT_REGIME_PROFILES), dict(DEFAULT_P90_HOLD))

    def _yn(key: str, default: bool) -> bool:
        raw = (os.environ.get(key) or "").strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}

    uw_bars = int(np.clip(int((os.environ.get("OOS_MTM_UNDERWATER_BARS") or str(c.underwater_streak_bars)).strip() or "8"), 1, 500))
    dd_frac_on = (os.environ.get("OOS_MTM_DD_FRACTIONAL", "1") or "1").strip() in {"1", "true", "yes", "on"}

    return AdaptiveMtmConfig(
        mtm_fast_window=fast,
        mtm_slow_window=slow,
        min_hold_bars=min_h,
        vote_k=v_def,
        vote_k_profit=v_profit,
        vote_k_loss=v_loss,
        z_score_hi=z_hi,
        z_score_trend_hi=z_tr,
        zscore_min_history=zmin,
        zscore_level_enabled=_yn("OOS_MTM_PROB_LEVEL_ENABLED", True),
        zscore_level_threshold=z_lvl,
        enable_momentum=_yn("OOS_MTM_ENABLE_MOMENTUM", True),
        enable_drawdown=_yn("OOS_MTM_ENABLE_DRAWDOWN", True),
        enable_time=_yn("OOS_MTM_ENABLE_TIME", True),
        enable_model=_yn("OOS_MTM_ENABLE_MODEL", True),
        enable_underwater=_yn("OOS_MTM_ENABLE_UNDERWATER", True),
        history_max_bars=hmax,
        regime_profiles=prof,
        p90_hold_by_regime=p90,
        time_urgency_active_thr=u_thr,
        time_early_frac=t_early,
        safety_net_use_regime_p90=_yn("OOS_MTM_SAFETY_REGIME_P90", True),
        dd_use_fractional=dd_frac_on,
        underwater_streak_bars=uw_bars,
    )


def _momentum_cross(
    mtm: np.ndarray,
    *,
    fast: int,
    slow: int,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"active": False, "reason": "disabled"}
    x = np.asarray(mtm, dtype=np.float64).ravel()
    if x.size < slow + 1:
        return {"active": False, "reason": "insufficient_data"}
    d = np.diff(x)
    if d.size < slow:
        return {"active": False, "reason": "insufficient_data"}
    fast_ma = float(np.mean(d[-fast:]))
    slow_ma = float(np.mean(d[-slow:]))
    if d.size > fast:
        prev_fast = float(np.mean(d[-fast - 1 : -1]))
    else:
        prev_fast = fast_ma
    if d.size > slow:
        prev_slow = float(np.mean(d[-slow - 1 : -1]))
    else:
        prev_slow = slow_ma
    cross_down = (prev_fast >= prev_slow) and (fast_ma < slow_ma)
    both_negative = (fast_ma < 0.0) and (slow_ma < 0.0)
    active = bool(cross_down or both_negative)
    return {
        "active": active,
        "fast_ma": fast_ma,
        "slow_ma": slow_ma,
        "cross_down": cross_down,
        "both_negative": both_negative,
    }


def _peak_drawdown(
    mtm: np.ndarray,
    cfg: AdaptiveMtmConfig,
    *,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"active": False, "reason": "disabled", "drawdown_abs": 0.0, "drawdown_frac": 0.0, "peak_mtm": 0.0, "current_mtm": 0.0}
    x = np.asarray(mtm, dtype=np.float64).ravel()
    if x.size < 3:
        return {"active": False, "reason": "insufficient_data", "drawdown_abs": 0.0, "drawdown_frac": 0.0, "peak_mtm": 0.0, "current_mtm": 0.0}
    peak_mtm = float(np.max(x))
    current_mtm = float(x[-1])
    drawdown_abs = max(0.0, peak_mtm - current_mtm)
    drawdown_frac = 0.0
    active_frac = False
    if cfg.dd_use_fractional and peak_mtm > 0.01:
        drawdown_frac = float(drawdown_abs / max(peak_mtm, 1e-9))
        active_frac = bool(drawdown_frac > float(cfg.dd_frac_tolerance_profit))
    if peak_mtm > 0:
        tolerance_abs = min(
            float(cfg.dd_base_tolerance_profit) + float(peak_mtm) * float(cfg.dd_profit_scale),
            float(cfg.dd_max_tolerance),
        )
    else:
        tolerance_abs = float(cfg.dd_tolerance_loss)
    active_abs = bool(drawdown_abs > tolerance_abs)
    active_loss = bool(current_mtm < -float(cfg.dd_abs_tolerance_loss) and peak_mtm <= 0.02)
    active = bool(active_frac or active_abs or active_loss)
    return {
        "active": active,
        "drawdown_abs": float(drawdown_abs),
        "drawdown_frac": float(drawdown_frac),
        "tolerance_abs": float(tolerance_abs),
        "active_frac": bool(active_frac),
        "active_abs": bool(active_abs),
        "active_loss": bool(active_loss),
        "peak_mtm": float(peak_mtm),
        "current_mtm": float(current_mtm),
    }


def _underwater_streak(
    mtm: list[float] | np.ndarray,
    regime: int,
    cfg: AdaptiveMtmConfig,
) -> dict[str, Any]:
    mtm = np.asarray(mtm, dtype=np.float64).ravel()
    if not cfg.underwater_enabled or mtm.size < 3:
        return {
            "active": False,
            "reason": "disabled_or_short",
            "streak": 0,
            "deep_streak": 0,
            "current_mtm": float(mtm[-1]) if mtm.size else 0.0,
        }
    rid = int(np.clip(int(regime), 0, 4))
    normal_streak = int(cfg.regime_underwater_streak_override.get(rid, cfg.underwater_streak_bars))
    deep_req = int(cfg.underwater_deep_streak_bars)
    th = float(cfg.underwater_threshold)
    dth = float(cfg.underwater_deep_threshold)
    streak = 0
    for v in reversed(mtm.tolist()):
        if float(v) < th:
            streak += 1
        else:
            break
    deep_streak = 0
    for v in reversed(mtm.tolist()):
        if float(v) < dth:
            deep_streak += 1
        else:
            break
    current_mtm = float(mtm[-1])
    active = bool((streak >= normal_streak) or (deep_streak >= deep_req))
    return {
        "active": bool(active),
        "streak": int(streak),
        "deep_streak": int(deep_streak),
        "normal_streak_req": int(normal_streak),
        "deep_streak_req": int(deep_req),
        "current_mtm": current_mtm,
    }


def _time_decay(
    bars_held: int,
    regime_id: int,
    cfg: AdaptiveMtmConfig,
    enabled: bool,
    *,
    trade_max_hold: int | None = None,
) -> dict[str, Any]:
    if not enabled:
        return {
            "active": False,
            "reason": "disabled",
            "urgency": 0.0,
            "regime_safety": False,
            "safety_cap_bars": 0,
        }
    rid = int(np.clip(int(regime_id), 0, 4))
    prof = cfg.regime_profiles.get(rid, cfg.regime_profiles.get(2, DEFAULT_REGIME_PROFILES[2]))
    median_best = float(prof.get("median_best", 17.0))
    decay_rate = float(prof.get("decay_rate", 0.04))
    bars = int(bars_held)
    early = float(np.clip(float(cfg.time_early_frac), 0.1, 0.9))
    urgency = 0.0
    active = False
    p90 = int(cfg.p90_hold_by_regime.get(rid, 40))
    if trade_max_hold is not None and int(trade_max_hold) > 0:
        cap = int(min(p90, int(trade_max_hold)))
    else:
        cap = int(p90)
    regime_safety = bool(
        int(cfg.safety_net_use_regime_p90) and int(bars) > int(cap) and int(cap) > 0
    )
    if bars <= median_best * early:
        urgency = 0.0
        active = False
    elif bars <= median_best:
        denom = max(median_best * (1.0 - early), 1e-6)
        urgency = float((bars - median_best * early) / denom) * 0.5
        active = False
    else:
        overshoot = float(bars - median_best)
        urgency = float(min(1.0, overshoot * decay_rate + 0.5))
        active = bool(urgency > float(cfg.time_urgency_active_thr))
    return {
        "active": bool(active),
        "bars_held": bars,
        "median_best": median_best,
        "urgency": float(urgency),
        "regime_safety": bool(regime_safety),
        "safety_cap_bars": int(cap),
    }


def _model_relative(
    full_exit_prob_in_trade: np.ndarray,
    cfg: AdaptiveMtmConfig,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"active": False, "reason": "disabled", "z_score": 0.0, "level_trigger": False, "trending_up": False}
    h = np.asarray(full_exit_prob_in_trade, dtype=np.float64).ravel()
    n = int(h.size)
    if n < 1:
        return {
            "active": False,
            "reason": "insufficient_history",
            "z_score": float("nan"),
            "level_trigger": False,
            "trending_up": False,
        }
    p = float(np.clip(h[-1], 0.0, 1.0))
    level_trigger = bool(cfg.zscore_level_enabled and p > float(cfg.zscore_level_threshold))
    if n < 2:
        return {
            "active": bool(level_trigger),
            "current_prob": p,
            "z_score": 0.0,
            "hist_mean": float(p),
            "hist_std": 0.0,
            "trending_up": False,
            "level_trigger": level_trigger,
        }
    k = int(max(1, int(cfg.zscore_min_history)))
    prior = h[:-1]
    z_score = 0.0
    z_active = False
    active_trend = False
    trending_up = False
    hist_mean = float(np.mean(prior))
    hist_std = float(np.std(prior, ddof=1)) if int(prior.size) > 1 else 0.0
    if int(prior.size) >= k and hist_std >= 1e-8:
        z_score = float((p - hist_mean) / max(hist_std, 1e-8))
        z_active = z_score > float(cfg.z_score_hi)
    if h.size >= 3:
        r3 = h[-3:]
        trending_up = bool((r3[1] > r3[0]) and (r3[2] > r3[1]))
    if trending_up and hist_std >= 1e-8:
        active_trend = z_score > float(cfg.z_score_trend_hi)
    mom_z = z_active or active_trend
    active = bool(mom_z or level_trigger)
    return {
        "active": bool(active),
        "current_prob": p,
        "z_score": float(z_score),
        "hist_mean": float(hist_mean),
        "hist_std": float(hist_std),
        "trending_up": trending_up,
        "level_trigger": level_trigger,
    }


def _p90_safety_exceeded(
    bars_held: int,
    regime_id: int,
    cfg: AdaptiveMtmConfig,
    trade_max_hold: int | None,
) -> tuple[bool, int]:
    """Hard stop when hold exceeds min(p90, trade_max) if configured."""
    if not bool(cfg.safety_net_use_regime_p90):
        return False, 0
    rid = int(np.clip(int(regime_id), 0, 4))
    p90 = int(cfg.p90_hold_by_regime.get(rid, 40))
    if trade_max_hold is not None and int(trade_max_hold) > 0:
        cap = int(min(p90, int(trade_max_hold)))
    else:
        cap = p90
    return bool(int(bars_held) > int(cap) and int(cap) > 0), int(cap)


def evaluate_adaptive_mtm_exit(
    *,
    config: AdaptiveMtmConfig,
    bars_held: int,
    mtm_history: list[float] | np.ndarray,
    exit_prob_history: list[float] | np.ndarray,
    current_exit_prob: float,
    entry_atr: float,
    entry_regime_id: int,
    state_hold_ref: int,
    trade_max_hold: int | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    _ = float(entry_atr)  # kept for call-site / API stability
    mtm = np.asarray(mtm_history, dtype=np.float64).ravel()
    ph = np.asarray(exit_prob_history, dtype=np.float64).ravel()
    rid0 = int(np.clip(int(entry_regime_id), 0, 4))
    rmin = int(config.regime_min_hold_override.get(rid0, int(config.min_hold_bars)))
    eff_min = int(max(int(rmin), int(state_hold_ref), 0))
    if int(bars_held) < eff_min:
        return (
            False,
            "hold_min",
            {
                "bars_held": int(bars_held),
                "eff_min_hold": int(eff_min),
                "momentum": {"active": False},
                "drawdown": {"active": False},
                "time": {"active": False},
                "model": {"active": False},
                "underwater": {"active": False},
            },
        )

    mom = _momentum_cross(
        mtm,
        fast=int(config.mtm_fast_window),
        slow=int(config.mtm_slow_window),
        enabled=config.enable_momentum,
    )
    dd = _peak_drawdown(mtm, config, enabled=config.enable_drawdown)
    tm = _time_decay(
        int(bars_held), int(rid0), config, config.enable_time, trade_max_hold=trade_max_hold
    )
    if ph.size == 0 and np.isfinite(float(current_exit_prob)):
        ph = np.asarray([float(current_exit_prob)], dtype=np.float64)
    md = _model_relative(ph, config, enabled=config.enable_model)
    uw = _underwater_streak(mtm, rid0, config) if config.enable_underwater else {"active": False}

    mom_a = bool(mom.get("active", False))
    dd_a = bool(dd.get("active", False))
    time_a = bool(tm.get("active", False))
    mod_a = bool(md.get("active", False))
    wat = bool(uw.get("active", False))
    reg_safe = bool(tm.get("regime_safety", False))
    is_profitable = bool(mtm.size > 0 and float(mtm[-1]) > 0.01)

    detail_base: dict[str, Any] = {
        "momentum": mom,
        "drawdown": dd,
        "time": tm,
        "model": md,
        "underwater": uw,
    }

    if mom_a and dd_a and is_profitable:
        return (
            True,
            "profit_decay",
            _pack_out("profit_decay", detail_base, n_act=2, is_profitable=is_profitable, vote_k=0, labels=[]),
        )

    if wat:
        return (
            True,
            "underwater_bleed",
            _pack_out(
                "underwater_bleed",
                detail_base,
                n_act=1,
                is_profitable=is_profitable,
                vote_k=0,
                labels=["water"],
            ),
        )

    if reg_safe:
        return (
            True,
            "regime_safety",
            _pack_out("regime_safety", detail_base, n_act=0, is_profitable=is_profitable, vote_k=0, cap=tm.get("safety_cap_bars", 0)),
        )

    if time_a and mod_a and (not mom_a) and (not dd_a) and mtm.size > 0 and float(mtm[-1]) <= 0.0:
        return (
            False,
            "time_model_underwater",
            _pack_out("time_model_underwater", detail_base, n_act=0, is_profitable=is_profitable, vote_k=0, blocked=True),
        )

    safe_hit, _cap = _p90_safety_exceeded(int(bars_held), int(rid0), config, trade_max_hold)
    if safe_hit and not reg_safe:
        return (
            True,
            "safety_net_p90",
            _pack_out("safety_net_p90", detail_base, n_act=0, is_profitable=is_profitable, cap=_cap, vote_k=0),
        )

    vote_profit = int(config.vote_k_profit)
    vote_loss_base = int(config.vote_k_loss)
    r_loss = int(config.regime_vote_k_loss_override.get(int(rid0), vote_loss_base))
    k_need = int(vote_profit if is_profitable else r_loss)
    names = [("mom", mom_a), ("dd", dd_a), ("time", time_a), ("model", mod_a), ("water", wat)]
    act_labels = [s for s, v in names if v]
    n_act = int(len(act_labels))
    d_out = _pack_out(
        f"vote_{n_act}",
        detail_base,
        n_act=n_act,
        is_profitable=is_profitable,
        vote_k=int(k_need),
        labels=act_labels,
    )
    if n_act >= k_need and k_need > 0:
        return True, f"vote_{n_act}", d_out
    d_out["code"] = "hold"
    return False, "hold", d_out


def _pack_out(
    code: str,
    detail: dict[str, Any],
    *,
    n_act: int,
    is_profitable: bool,
    vote_k: int,
    labels: list[str] | None = None,
    blocked: bool = False,
    cap: int = 0,
) -> dict[str, Any]:
    out: dict[str, Any] = {**detail, "code": code, "active_count": int(n_act), "is_profitable": is_profitable, "vote_k_used": int(vote_k)}
    if labels is not None:
        out["active_signals"] = list(labels)
    if blocked:
        out["blocked"] = True
    if cap is not None and int(cap) > 0 and "safety_cap_bars" not in out.get("time", {}):
        out["safety_cap_bars"] = int(cap)
    return out


def trim_histories_in_place(
    mtm: list[float],
    probs: list[float],
    max_len: int,
) -> None:
    if max_len <= 0:
        return
    while len(mtm) > max_len:
        mtm.pop(0)
    while len(probs) > max_len:
        probs.pop(0)
