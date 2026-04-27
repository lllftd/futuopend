from __future__ import annotations

from typing import NamedTuple

import numpy as np


class L3PolicyParams(NamedTuple):
    exit_rate_penalty: float
    hold_recall_w: float
    hold_recall_floor: float
    target_exit_rate: float
    diag_hold_rate: float
    diag_opp_cost: float
    diag_save_benefit: float
    diag_value_head_degen: bool


def derive_policy_params(
    labels: np.ndarray,
    values: np.ndarray,
    *,
    value_pred_std: float,
    value_target_std: float,
) -> L3PolicyParams:
    y = np.asarray(labels, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    valid = np.isfinite(y) & np.isfinite(v)
    if not valid.any():
        return L3PolicyParams(
            exit_rate_penalty=1.0,
            hold_recall_w=0.30,
            hold_recall_floor=0.20,
            target_exit_rate=0.50,
            diag_hold_rate=0.50,
            diag_opp_cost=0.0,
            diag_save_benefit=0.0,
            diag_value_head_degen=True,
        )
    y = y[valid]
    v = v[valid]
    exit_base = float(np.clip(np.mean(y), 0.01, 0.99))
    hold_base = float(1.0 - exit_base)
    hold_mask = y <= 0.5
    exit_mask = y > 0.5

    # Imbalance-aware penalty on over-exiting.
    exit_rate_penalty = float(np.log(max(exit_base, 1e-9) / max(hold_base, 1e-9)) + 1.0)

    pred_std = float(np.clip(value_pred_std, 0.0, np.inf))
    tgt_std = float(max(value_target_std, 1e-9))
    value_head_degen = (pred_std / tgt_std) < 0.10

    pos_hold = v[hold_mask & (v > 0.0)]
    opp_cost = float(np.mean(pos_hold)) if pos_hold.size >= 10 else 0.0
    neg_exit = v[exit_mask & (v < 0.0)]
    save_benefit = float(-np.mean(neg_exit)) if neg_exit.size >= 10 else 0.0

    denom = opp_cost + save_benefit
    if denom > 1e-9 and not value_head_degen:
        raw_w = float(opp_cost / denom)
    else:
        raw_w = hold_base
    if value_head_degen:
        degen_boost = hold_base * (1.0 - (pred_std / tgt_std)) * 0.5
        raw_w += float(degen_boost)
    hold_recall_w = float(np.clip(raw_w, 0.10, 0.60))
    hold_recall_floor = float(np.clip(hold_base * (2.0 / 3.0), 0.15, 0.50))

    return L3PolicyParams(
        exit_rate_penalty=round(exit_rate_penalty, 4),
        hold_recall_w=round(hold_recall_w, 4),
        hold_recall_floor=round(hold_recall_floor, 4),
        target_exit_rate=round(exit_base, 4),
        diag_hold_rate=round(hold_base, 4),
        diag_opp_cost=round(opp_cost, 4),
        diag_save_benefit=round(save_benefit, 4),
        diag_value_head_degen=bool(value_head_degen),
    )
