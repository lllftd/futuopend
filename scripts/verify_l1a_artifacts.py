#!/usr/bin/env python3
"""Check L1a checkpoint + l1a_outputs.pkl contract (schema, columns, files on disk).

Run from repo root: PYTHONPATH=. python3 scripts/verify_l1a_artifacts.py
Exit 0 = OK, 1 = failure.
"""
from __future__ import annotations

import os
import pickle
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.trainers.constants import (  # noqa: E402
    L1A_META_FILE,
    L1A_MODEL_FILE,
    L1A_OUTPUT_CACHE_FILE,
    L1A_SCHEMA_VERSION,
    MODEL_DIR,
)
from core.trainers.l1a.train import l1a_output_columns_with_embed_dim  # noqa: E402


def main() -> int:
    model_path = os.path.join(MODEL_DIR, L1A_MODEL_FILE)
    meta_path = os.path.join(MODEL_DIR, L1A_META_FILE)
    cache_path = os.path.join(MODEL_DIR, L1A_OUTPUT_CACHE_FILE)
    for p, label in [
        (model_path, L1A_MODEL_FILE),
        (meta_path, L1A_META_FILE),
        (cache_path, L1A_OUTPUT_CACHE_FILE),
    ]:
        if not os.path.isfile(p):
            print(f"FAIL: missing {label} -> {p}")
            return 1

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    sv = meta.get("schema_version")
    if sv != L1A_SCHEMA_VERSION:
        print(f"FAIL: meta schema_version={sv!r} expected {L1A_SCHEMA_VERSION!r}")
        return 1

    import torch

    try:
        sd = torch.load(model_path, map_location="cpu")
        if not isinstance(sd, dict) or not sd:
            print(f"FAIL: {L1A_MODEL_FILE} is not a non-empty state_dict")
            return 1
    except Exception as ex:
        print(f"FAIL: could not load {L1A_MODEL_FILE}: {ex}")
        return 1

    with open(cache_path, "rb") as f:
        out = pickle.load(f)
    if not hasattr(out, "columns"):
        print("FAIL: l1a_outputs.pkl is not a DataFrame")
        return 1

    ed = int(meta.get("embed_dim", 8))
    expected = set(l1a_output_columns_with_embed_dim(ed))
    got_l1a = {c for c in out.columns if str(c).startswith("l1a_")}
    missing = sorted(expected - got_l1a)
    extra = sorted(got_l1a - expected)
    if missing:
        print(f"FAIL: cache missing l1a columns: {missing[:12]}{'...' if len(missing) > 12 else ''}")
        return 1
    if extra:
        print(f"WARN: cache has extra l1a columns vs contract: {extra[:8]}{'...' if len(extra) > 8 else ''}")

    print("OK: L1a artifacts consistent")
    print(f"  schema={L1A_SCHEMA_VERSION}  embed_dim={ed}  l1a_cols={len(expected)}  rows={len(out):,}")
    print(f"  meta={meta_path}")
    print(f"  cache={cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
