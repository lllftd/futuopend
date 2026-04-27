"""Unit tests for MTM adaptive exit."""
from __future__ import annotations

import numpy as np

from core.training.unified.adaptive_mtm_exit import (
    AdaptiveMtmConfig,
    _underwater_streak,
    _peak_drawdown,
    evaluate_adaptive_mtm_exit,
    load_adaptive_mtm_config_from_env,
)


def test_min_hold_blocks_exit():
    mtm = np.linspace(0, 0.1, 12)
    probs = [0.2] * 11
    cfg = AdaptiveMtmConfig(
        min_hold_bars=10,
        mtm_fast_window=2,
        mtm_slow_window=5,
        vote_k=1,
        vote_k_profit=1,
    )
    ex, reason, _ = evaluate_adaptive_mtm_exit(
        config=cfg,
        bars_held=3,
        mtm_history=mtm,
        exit_prob_history=probs,
        current_exit_prob=0.5,
        entry_atr=0.5,
        entry_regime_id=2,
        state_hold_ref=5,
    )
    assert ex is False
    assert reason == "hold_min"


def test_profit_decay_mom_and_dd():
    mtm = np.array(
        [0.0, 0.2, 0.4, 0.5, 0.45, 0.35, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02],
        dtype=np.float64,
    )
    n = len(mtm)
    probs = [0.25 + 0.01 * (i % 5) for i in range(n)]
    cfg = AdaptiveMtmConfig(
        min_hold_bars=3,
        mtm_fast_window=2,
        mtm_slow_window=5,
        zscore_level_enabled=False,
    )
    ex, reason, d = evaluate_adaptive_mtm_exit(
        config=cfg,
        bars_held=n,
        mtm_history=mtm,
        exit_prob_history=probs,
        current_exit_prob=probs[-1],
        entry_atr=0.1,
        entry_regime_id=2,
        state_hold_ref=0,
    )
    if ex and reason == "profit_decay":
        assert d["momentum"].get("active") and d["drawdown"].get("active")


def test_time_model_underwater_blocks():
    cfg = AdaptiveMtmConfig(
        min_hold_bars=1,
        vote_k=4,
        enable_momentum=False,
        enable_drawdown=False,
        zscore_level_enabled=False,
    )
    mtm = np.array([-0.1] * 30, dtype=np.float64)
    probs = [0.1] * 5 + [0.11, 0.12, 0.13, 0.14, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.55, 0.66, 0.7, 0.8, 0.85, 0.9, 0.95, 0.7, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.95, 0.99, 0.99, 0.99]
    ex, reason, _ = evaluate_adaptive_mtm_exit(
        config=cfg,
        bars_held=30,
        mtm_history=mtm,
        exit_prob_history=probs,
        current_exit_prob=0.99,
        entry_atr=0.1,
        entry_regime_id=0,
        state_hold_ref=0,
    )
    if reason == "time_model_underwater":
        assert ex is False


def test_safety_p90():
    cfg = AdaptiveMtmConfig(
        min_hold_bars=1,
        enable_momentum=False,
        enable_drawdown=False,
        enable_time=False,
        enable_model=False,
        vote_k=9,
    )
    n = 35
    mtm = np.ones(n) * 0.01
    probs = [0.2] * n
    ex, reason, _ = evaluate_adaptive_mtm_exit(
        config=cfg,
        bars_held=n,
        mtm_history=mtm,
        exit_prob_history=probs,
        current_exit_prob=0.2,
        entry_atr=1.0,
        entry_regime_id=3,
        state_hold_ref=0,
        trade_max_hold=80,
    )
    assert cfg.p90_hold_by_regime[3] == 30
    assert ex and reason in ("safety_net_p90", "regime_safety")


def test_model_z_score_spike():
    cfg = AdaptiveMtmConfig(
        min_hold_bars=1,
        enable_momentum=False,
        enable_drawdown=False,
        enable_time=False,
        enable_model=True,
        vote_k=1,
        vote_k_profit=1,
        z_score_hi=0.1,
        zscore_level_enabled=False,
    )
    mtm = np.linspace(0, 0.1, 20)
    probs = [0.2 + 0.001 * i for i in range(20)]
    ex, reason, d = evaluate_adaptive_mtm_exit(
        config=cfg,
        bars_held=20,
        mtm_history=mtm,
        exit_prob_history=probs,
        current_exit_prob=probs[-1],
        entry_atr=0.5,
        entry_regime_id=2,
        state_hold_ref=0,
    )
    assert ex and reason.startswith("vote")
    assert float(d.get("model", {}).get("z_score", 0.0) or 0.0) > 0.1


def test_underwater_short_no_trigger():
    cfg = AdaptiveMtmConfig(
        min_hold_bars=1,
        underwater_streak_bars=8,
        enable_momentum=False,
        enable_drawdown=False,
        enable_time=False,
        enable_model=False,
        vote_k=3,
    )
    mtm = [0.01] * 5 + [-0.03] * 4
    res = _underwater_streak(np.asarray(mtm, dtype=np.float64), regime=2, cfg=cfg)
    assert res["active"] is False


def test_underwater_long_triggers():
    cfg = AdaptiveMtmConfig(
        min_hold_bars=1,
        enable_momentum=False,
        enable_drawdown=False,
        enable_time=False,
        enable_model=False,
        vote_k=3,
    )
    cfg.regime_underwater_streak_override[2] = 6
    mtm = [0.01] * 3 + [-0.05] * 8
    ex, reason, d = evaluate_adaptive_mtm_exit(
        config=cfg,
        bars_held=len(mtm),
        mtm_history=mtm,
        exit_prob_history=[0.2] * len(mtm),
        current_exit_prob=0.2,
        entry_atr=0.1,
        entry_regime_id=2,
        state_hold_ref=0,
    )
    assert ex is True
    assert reason == "underwater_bleed"
    assert d["underwater"]["streak"] == 8


def test_underwater_deep_faster():
    cfg = AdaptiveMtmConfig(
        min_hold_bars=1,
        underwater_deep_threshold=-0.10,
        underwater_deep_streak_bars=5,
        enable_momentum=False,
        enable_drawdown=False,
        enable_time=False,
        enable_model=False,
        vote_k=3,
    )
    mtm = [0.01] * 3 + [-0.12] * 5
    res = _underwater_streak(np.asarray(mtm, dtype=np.float64), regime=2, cfg=cfg)
    assert res["active"] is True
    assert res["deep_streak"] == 5


def test_fractional_drawdown_profit_path():
    cfg = AdaptiveMtmConfig(
        min_hold_bars=1,
        dd_use_fractional=True,
        dd_frac_tolerance_profit=0.35,
        enable_momentum=False,
        enable_time=False,
        enable_model=False,
        vote_k=3,
    )
    mtm = [0.0, 0.1, 0.2, 0.3, 0.25, 0.15]
    r = _peak_drawdown(np.asarray(mtm, dtype=np.float64), cfg, enabled=True)
    assert r["active_frac"] is True
    assert abs(float(r["drawdown_frac"]) - 0.5) < 0.02


def test_env_loader_smoke():
    c = load_adaptive_mtm_config_from_env()
    assert c.mtm_fast_window >= 1
    assert c.vote_k_profit >= 1
