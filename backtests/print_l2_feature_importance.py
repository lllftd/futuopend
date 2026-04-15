"""Print L2 LightGBM feature importance (gain) top-30 per head. Run: PYTHONPATH=. python3 backtests/print_l2_feature_importance.py"""
from __future__ import annotations

import pickle

import lightgbm as lgb
import numpy as np

from core.trainers.constants import L2_META_FILE, MODEL_DIR


def main() -> None:
    with open(f"{MODEL_DIR}/{L2_META_FILE}", "rb") as f:
        meta = pickle.load(f)
    feat = list(meta["feature_cols"])
    mf = meta.get("model_files", {})
    print(f"L2 schema_version={meta.get('schema_version')}  n_features={len(feat)}\n")

    head_names = [k for k in mf if not k.endswith("_calibrator")]
    priority = ["trade_gate", "long_gate", "short_gate", "signed_edge", "size", "mfe", "mae"]
    ordered = [h for h in priority if h in head_names] + sorted(h for h in head_names if h not in priority)
    for name in ordered:
        fname = mf.get(name)
        if not fname or not str(fname).endswith(".txt"):
            continue
        booster = lgb.Booster(model_file=f"{MODEL_DIR}/{fname}")
        imp = booster.feature_importance(importance_type="gain")
        order = np.argsort(imp)[::-1]
        print("=" * 72)
        print(f"[{name}] top-30 by gain")
        print("=" * 72)
        for rank, i in enumerate(order[:30], 1):
            star = " *" if str(feat[i]).startswith("l1c_") else ""
            print(f"{rank:2d}  {imp[i]:12.1f}  {feat[i]}{star}")
        l1c_ranks = []
        for i in range(len(feat)):
            if str(feat[i]).startswith("l1c_"):
                r = int(np.where(order == i)[0][0]) + 1
                l1c_ranks.append((r, feat[i], imp[i]))
        l1c_ranks.sort(key=lambda x: x[0])
        if l1c_ranks:
            parts = [f"{c} r={r} g={g:.0f}" for r, c, g in l1c_ranks]
            print("  l1c_*:", "; ".join(parts))
        print()


if __name__ == "__main__":
    main()
