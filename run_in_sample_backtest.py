"""Price + equity chart for the stack **pre-TEST_END** window (in-sample / holdout), not post-TEST_END OOS.

Uses the same simulation as ``run_oos_backtest.py`` but defaults to:
  - OOS_END = TEST_END (2025-01-01) — end **before** default out-of-sample
  - OOS_START = TRAIN_END (2023-01-01) — aligns with stack calendar; avoids multi-hour ``add_pa_features`` on full CSV
  - OOS_RESULTS_DIR = results/in_sample_upto_test_end
  - Chart title mentions in-sample

**Full CSV from first bar:** set ``OOS_START=2000-01-01`` (expect ~10^5+ rows: PA rules alone can take ~1h/symbol on a laptop).

Override any value via environment before launch.

Examples:
  python3 run_in_sample_backtest.py
  OOS_START=2020-03-01 OOS_END=2025-01-01 python3 run_in_sample_backtest.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.trainers.constants import TEST_END, TRAIN_END

os.environ.setdefault("OOS_END", str(TEST_END))
os.environ.setdefault("OOS_START", str(TRAIN_END))
os.environ.setdefault("OOS_RESULTS_DIR", os.path.join(_ROOT, "results", "in_sample_upto_test_end"))
os.environ.setdefault(
    "OOS_CHART_TITLE",
    "In-sample {symbol} | [{oos_start}, {oos_end})  (pre TEST_END={test_end})",
)

from backtests.oos_backtest import main

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    _os = os.environ.get("OOS_START", str(TRAIN_END))
    if str(_os) < "2023-01-01":
        print(
            "[in-sample] WARNING: OOS_START is before 2023-01-01 — PA feature prep can take ~1h+ per symbol "
            "for hundreds of thousands of 5m bars. Prefer default (TRAIN_END) or a later OOS_START.\n",
            flush=True,
        )
    main()
