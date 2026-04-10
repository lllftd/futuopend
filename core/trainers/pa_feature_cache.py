from __future__ import annotations

import os
import pickle
import tempfile
import time as _time

import pandas as pd

from core import indicators as indicators_module
from core import pa_rules as pa_rules_module
from core.indicators import atr as compute_atr
from core.pa_rules import add_pa_features


PA_FEATURE_CACHE_SCHEMA = 1


def _pa_cache_dir(data_dir: str) -> str:
    cache_dir = os.environ.get("PA_FEATURE_CACHE_DIR", "").strip()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    cache_dir = os.path.join(data_dir, ".pa_feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _pa_cache_path(symbol: str, data_dir: str, timeframe: str) -> str:
    safe_tf = timeframe.replace("/", "_")
    return os.path.join(_pa_cache_dir(data_dir), f"{symbol}_{safe_tf}.pkl")


def _pa_cache_meta(raw_path: str, timeframe: str) -> dict[str, int | str]:
    raw_stat = os.stat(raw_path)
    return {
        "schema": PA_FEATURE_CACHE_SCHEMA,
        "timeframe": timeframe,
        "raw_mtime_ns": raw_stat.st_mtime_ns,
        "pa_rules_mtime_ns": os.stat(pa_rules_module.__file__).st_mtime_ns,
        "indicators_mtime_ns": os.stat(indicators_module.__file__).st_mtime_ns,
    }


def load_or_build_pa_features(
    symbol: str,
    data_dir: str,
    *,
    timeframe: str = "5min",
) -> pd.DataFrame:
    raw_path = os.path.join(data_dir, f"{symbol}.csv")
    cache_path = _pa_cache_path(symbol, data_dir, timeframe)
    expected_meta = _pa_cache_meta(raw_path, timeframe)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if (
                isinstance(payload, dict)
                and payload.get("meta") == expected_meta
                and isinstance(payload.get("df"), pd.DataFrame)
            ):
                df_cached = payload["df"]
                print(
                    f"  [{symbol}] Loaded cached PA features ({len(df_cached):,} rows) from {cache_path}",
                    flush=True,
                )
                return df_cached
        except Exception as exc:
            print(f"  [{symbol}] Ignoring stale/corrupt PA cache {cache_path}: {exc}", flush=True)

    raw = pd.read_csv(raw_path)
    raw["time_key"] = pd.to_datetime(raw["time_key"])
    raw = raw.sort_values("time_key").reset_index(drop=True)

    atr_1m = compute_atr(raw, length=14)
    print(f"  [{symbol}] Computing PA features on {len(raw):,} bars…", flush=True)
    t0 = _time.time()
    df_pa = add_pa_features(raw, atr_1m, timeframe=timeframe)
    print(f"  [{symbol}] Done in {_time.time()-t0:.1f}s → {df_pa.shape[1]} cols", flush=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{symbol}_{timeframe}_",
        suffix=".tmp",
        dir=_pa_cache_dir(data_dir),
    )
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump({"meta": expected_meta, "df": df_pa}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        print(f"  [{symbol}] Cached PA features → {cache_path}", flush=True)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return df_pa
