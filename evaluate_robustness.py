from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import V3_BEST_PARAMS
from optimize_ce_zlsma_kama_rule import (
    RuleParams,
    apply_ce_features,
    build_base_features,
    run_intraday_rule,
)
from utils import load_price_data


OUT_DIR = Path("results") / "robustness"
TRAIN_DAYS = 252
TEST_DAYS = 63
STEP_DAYS = 21
MIN_TEST_TRADES = 20


@dataclass
class FoldResult:
    symbol: str
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    train_win_rate: float
    test_win_rate: float
    train_dd: float
    test_dd: float
    train_trade_count: int
    test_trade_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "fold_id": self.fold_id,
            "train_start": self.train_start.date().isoformat(),
            "train_end": self.train_end.date().isoformat(),
            "test_start": self.test_start.date().isoformat(),
            "test_end": self.test_end.date().isoformat(),
            "train_sharpe": self.train_sharpe,
            "test_sharpe": self.test_sharpe,
            "train_total_return": self.train_return,
            "test_total_return": self.test_return,
            "train_win_rate": self.train_win_rate,
            "test_win_rate": self.test_win_rate,
            "train_max_drawdown": self.train_dd,
            "test_max_drawdown": self.test_dd,
            "train_trade_count": self.train_trade_count,
            "test_trade_count": self.test_trade_count,
        }


def _prepare_featured(symbol: str, params: RuleParams) -> pd.DataFrame:
    raw = load_price_data(symbol)
    base = build_base_features(
        raw,
        zlsma_length=params.zlsma_length,
        zlsma_offset=params.zlsma_offset,
        kama_er_length=params.kama_er_length,
        kama_fast_length=params.kama_fast_length,
        kama_slow_length=params.kama_slow_length,
        atr_percentile_lookback=params.atr_percentile_lookback,
        pseudo_cvd_method=params.pseudo_cvd_method,
        cvd_lookback=params.cvd_lookback,
        cvd_slope_lookback=params.cvd_slope_lookback,
    )
    return apply_ce_features(base, params.ce_length, params.ce_multiplier)


def _run_slice(df: pd.DataFrame, symbol: str, params: RuleParams) -> dict[str, object]:
    res = run_intraday_rule(df.reset_index(drop=True), symbol, params)
    return res.summary


def _walkforward(symbol: str, featured: pd.DataFrame, params: RuleParams) -> list[FoldResult]:
    days = pd.Series(featured["time_key"].dt.normalize().unique()).sort_values().reset_index(drop=True)
    folds: list[FoldResult] = []
    fold_id = 1
    start = 0
    total_days = len(days)
    while True:
        train_end_idx = start + TRAIN_DAYS
        test_end_idx = train_end_idx + TEST_DAYS
        if test_end_idx > total_days:
            break

        train_days = set(days.iloc[start:train_end_idx])
        test_days = set(days.iloc[train_end_idx:test_end_idx])
        train_df = featured[featured["time_key"].dt.normalize().isin(train_days)]
        test_df = featured[featured["time_key"].dt.normalize().isin(test_days)]

        train_summary = _run_slice(train_df, symbol, params)
        test_summary = _run_slice(test_df, symbol, params)

        folds.append(
            FoldResult(
                symbol=symbol,
                fold_id=fold_id,
                train_start=pd.Timestamp(days.iloc[start]),
                train_end=pd.Timestamp(days.iloc[train_end_idx - 1]),
                test_start=pd.Timestamp(days.iloc[train_end_idx]),
                test_end=pd.Timestamp(days.iloc[test_end_idx - 1]),
                train_sharpe=float(train_summary.get("sharpe", np.nan)),
                test_sharpe=float(test_summary.get("sharpe", np.nan)),
                train_return=float(train_summary.get("total_return", np.nan)),
                test_return=float(test_summary.get("total_return", np.nan)),
                train_win_rate=float(train_summary.get("win_rate", np.nan)),
                test_win_rate=float(test_summary.get("win_rate", np.nan)),
                train_dd=float(train_summary.get("max_drawdown", np.nan)),
                test_dd=float(test_summary.get("max_drawdown", np.nan)),
                train_trade_count=int(train_summary.get("trade_count", 0)),
                test_trade_count=int(test_summary.get("trade_count", 0)),
            )
        )
        fold_id += 1
        start += STEP_DAYS

    return folds


def _oos_holdout(symbol: str, featured: pd.DataFrame, params: RuleParams) -> dict[str, object]:
    days = pd.Series(featured["time_key"].dt.normalize().unique()).sort_values().reset_index(drop=True)
    total_days = len(days)
    oos_days = max(TEST_DAYS, int(total_days * 0.2))
    oos_days = min(oos_days, total_days - 1)

    split_idx = total_days - oos_days
    is_days = set(days.iloc[:split_idx])
    oos_set = set(days.iloc[split_idx:])

    is_df = featured[featured["time_key"].dt.normalize().isin(is_days)]
    oos_df = featured[featured["time_key"].dt.normalize().isin(oos_set)]

    is_summary = _run_slice(is_df, symbol, params)
    oos_summary = _run_slice(oos_df, symbol, params)

    is_sharpe = float(is_summary.get("sharpe", np.nan))
    oos_sharpe = float(oos_summary.get("sharpe", np.nan))
    sharpe_retention = np.nan
    if np.isfinite(is_sharpe) and abs(is_sharpe) > 1e-9:
        sharpe_retention = oos_sharpe / is_sharpe

    is_return = float(is_summary.get("total_return", np.nan))
    oos_return = float(oos_summary.get("total_return", np.nan))
    return_retention = np.nan
    if np.isfinite(is_return) and abs(is_return) > 1e-9:
        return_retention = oos_return / is_return

    return {
        "symbol": symbol,
        "is_start": pd.Timestamp(days.iloc[0]).date().isoformat(),
        "is_end": pd.Timestamp(days.iloc[split_idx - 1]).date().isoformat(),
        "oos_start": pd.Timestamp(days.iloc[split_idx]).date().isoformat(),
        "oos_end": pd.Timestamp(days.iloc[-1]).date().isoformat(),
        "is_days": int(split_idx),
        "oos_days": int(total_days - split_idx),
        "is_sharpe": is_sharpe,
        "oos_sharpe": oos_sharpe,
        "is_total_return": is_return,
        "oos_total_return": oos_return,
        "is_win_rate": float(is_summary.get("win_rate", np.nan)),
        "oos_win_rate": float(oos_summary.get("win_rate", np.nan)),
        "is_max_drawdown": float(is_summary.get("max_drawdown", np.nan)),
        "oos_max_drawdown": float(oos_summary.get("max_drawdown", np.nan)),
        "is_trade_count": int(is_summary.get("trade_count", 0)),
        "oos_trade_count": int(oos_summary.get("trade_count", 0)),
        "sharpe_retention": float(sharpe_retention) if np.isfinite(sharpe_retention) else np.nan,
        "return_retention": float(return_retention) if np.isfinite(return_retention) else np.nan,
    }


def _robust_score(folds_df: pd.DataFrame, oos_row: dict[str, object]) -> tuple[float, str]:
    if folds_df.empty:
        return 0.0, "NO_DATA"

    valid_folds = folds_df[folds_df["test_trade_count"] >= MIN_TEST_TRADES].copy()
    if valid_folds.empty:
        return 10.0, "VERY_WEAK"

    pos_ratio = float((valid_folds["test_total_return"] > 0).mean())
    sharpe_med = float(valid_folds["test_sharpe"].median())
    dd_worst = float(valid_folds["test_max_drawdown"].min())

    oos_ret = float(oos_row.get("oos_total_return", np.nan))
    oos_sharpe = float(oos_row.get("oos_sharpe", np.nan))
    oos_win = float(oos_row.get("oos_win_rate", np.nan))
    sharpe_retention = float(oos_row.get("sharpe_retention", np.nan))

    score = 0.0
    score += 30.0 * np.clip(pos_ratio, 0.0, 1.0)
    score += 20.0 * np.clip((sharpe_med + 1.0) / 3.0, 0.0, 1.0)
    score += 15.0 * np.clip((dd_worst + 0.03) / 0.03, 0.0, 1.0)
    score += 15.0 * np.clip((oos_ret + 0.02) / 0.06, 0.0, 1.0)
    score += 10.0 * np.clip((oos_sharpe + 0.5) / 2.0, 0.0, 1.0)
    score += 5.0 * np.clip((oos_win - 0.45) / 0.15, 0.0, 1.0)
    if np.isfinite(sharpe_retention):
        score += 5.0 * np.clip(sharpe_retention / 0.7, 0.0, 1.0)

    if score >= 75:
        label = "ROBUST"
    elif score >= 55:
        label = "MODERATE"
    else:
        label = "WEAK"
    return float(round(score, 2)), label


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fold_rows: list[dict[str, object]] = []
    oos_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for symbol, params in V3_BEST_PARAMS.items():
        featured = _prepare_featured(symbol, params)
        folds = _walkforward(symbol, featured, params)
        fold_df = pd.DataFrame([f.to_dict() for f in folds])
        oos_row = _oos_holdout(symbol, featured, params)
        score, label = _robust_score(fold_df, oos_row)

        if not fold_df.empty:
            fold_rows.extend(fold_df.to_dict("records"))
        oos_rows.append(oos_row)

        valid_fold_df = fold_df[fold_df["test_trade_count"] >= MIN_TEST_TRADES] if not fold_df.empty else pd.DataFrame()
        summary_rows.append(
            {
                "symbol": symbol,
                "folds_total": int(len(fold_df)),
                "folds_valid": int(len(valid_fold_df)),
                "wf_test_return_median": float(valid_fold_df["test_total_return"].median()) if not valid_fold_df.empty else np.nan,
                "wf_test_sharpe_median": float(valid_fold_df["test_sharpe"].median()) if not valid_fold_df.empty else np.nan,
                "wf_positive_fold_ratio": float((valid_fold_df["test_total_return"] > 0).mean()) if not valid_fold_df.empty else np.nan,
                "wf_worst_test_drawdown": float(valid_fold_df["test_max_drawdown"].min()) if not valid_fold_df.empty else np.nan,
                "oos_total_return": float(oos_row["oos_total_return"]),
                "oos_sharpe": float(oos_row["oos_sharpe"]),
                "oos_win_rate": float(oos_row["oos_win_rate"]),
                "oos_max_drawdown": float(oos_row["oos_max_drawdown"]),
                "sharpe_retention": float(oos_row["sharpe_retention"]) if np.isfinite(float(oos_row["sharpe_retention"])) else np.nan,
                "robust_score": score,
                "robust_label": label,
            }
        )

    fold_out = OUT_DIR / f"walkforward_folds_{timestamp}.csv"
    oos_out = OUT_DIR / f"oos_holdout_{timestamp}.csv"
    summary_out = OUT_DIR / f"robust_summary_{timestamp}.csv"

    pd.DataFrame(fold_rows).to_csv(fold_out, index=False)
    pd.DataFrame(oos_rows).to_csv(oos_out, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_out, index=False)

    print(f"[DONE] Walk-forward folds: {fold_out.as_posix()}")
    print(f"[DONE] OOS holdout:      {oos_out.as_posix()}")
    print(f"[DONE] Robust summary:   {summary_out.as_posix()}")
    print("")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
