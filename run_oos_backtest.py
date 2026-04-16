"""Thin entrypoint; implementation lives in ``backtests/oos_backtest.py``."""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backtests.oos_backtest import main

if __name__ == "__main__":
    # When piping to `tee`, default block-buffering delays stdout vs stderr (warnings).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    main()
