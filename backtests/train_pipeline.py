import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

from core.trainers.constants import (
    CAL_END,
    L1A_META_FILE,
    L1A_OUTPUT_CACHE_FILE,
    L1B_META_FILE,
    L1B_OUTPUT_CACHE_FILE,
    L1C_META_FILE,
    L1C_OUTPUT_CACHE_FILE,
    L2_META_FILE,
    L2_OUTPUT_CACHE_FILE,
    L3_META_FILE,
    MODEL_DIR,
    PREPARED_DATASET_CACHE_FILE,
    TEST_END,
)
from core.trainers.data_prep import prepare_dataset as prepare_lgbm_data
from core.trainers.feature_registry import parse_prep_layer_targets, prep_needs_tcn_derivatives
from core.trainers.tcn_data_prep import ensure_tcn_state_classifier_checkpoint
from core.trainers.l1a import (
    infer_l1a_market_encoder,
    load_l1a_market_encoder,
    train_l1a_market_encoder,
)
from core.trainers.l1b import (
    infer_l1b_market_descriptor,
    load_l1b_market_descriptor,
    train_l1b_market_descriptor,
)
from core.trainers.l2 import infer_l2_trade_decision, load_l2_trade_decision, train_l2_trade_decision
from core.trainers.l1c import train_l1c_direction
from core.trainers.l3 import train_l3_exit_manager
from core.trainers.lgbm_utils import configure_training_runtime, _lgbm_n_jobs
from core.trainers.stack_v2_common import load_output_cache, save_output_cache


class Logger:
    """Tee stdout to terminal and the layer log file.

    Real console handle is kept as ``terminal``; trainers use ``lgbm_utils._tqdm_stream()`` so
    tqdm bars render on the same stream as ordinary prints (and still skip logs/*.log spam).
    Warnings and tracebacks stay on stderr.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.original_stdout = sys.stdout
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        if message:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()

    def close(self):
        sys.stdout = self.original_stdout
        self.log.close()

# Fixed layer log filenames (overwrite each run) under logs/
_LAYER_LOG_FILES = {
    "layer1a": "layer1a.log",
    "layer1b": "layer1b.log",
    "layer1c": "layer1c.log",
    "layer2": "layer2.log",
    "layer3": "layer3.log",
}


def setup_logger(layer_key: str):
    """Tee stdout/stderr to terminal and exactly one overwrite log per stage."""
    if layer_key not in _LAYER_LOG_FILES:
        raise ValueError(f"setup_logger: unknown layer_key={layer_key!r}; expected one of {sorted(_LAYER_LOG_FILES)}")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, _LAYER_LOG_FILES[layer_key])
    lg = Logger(log_file)
    print(f"\n>> Layer log (overwrite): {log_file}")
    return lg


def _prepared_dataset_cache_path() -> str:
    return os.path.join(MODEL_DIR, PREPARED_DATASET_CACHE_FILE)


def _save_prepared_dataset_cache(df, feat_cols) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = _prepared_dataset_cache_path()
    with open(path, "wb") as f:
        pickle.dump({"df": df, "feat_cols": list(feat_cols)}, f)
    return path


def _load_prepared_dataset_cache():
    path = _prepared_dataset_cache_path()
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "df" not in obj or "feat_cols" not in obj:
        raise TypeError(f"Prepared dataset cache {path} is malformed.")
    return obj["df"], list(obj["feat_cols"])


def _prepare_or_load_lgbm_dataset(
    symbols: list[str],
    *,
    prefer_cache: bool,
    layer1a_frozen_prep_hint: bool = False,
):
    cache_path = _prepared_dataset_cache_path()
    force_rebuild = os.environ.get("PREPARED_DATASET_CACHE_REBUILD", "").strip().lower() in {"1", "true", "yes"}
    if prefer_cache and not force_rebuild and os.path.exists(cache_path):
        print(f"\n[*] Loading prepared dataset cache from {cache_path} ...")
        if layer1a_frozen_prep_hint:
            print(
                "    (layer1a + LAYER1A_USE_PREPARED_CACHE: PA+prep skipped — set "
                "PREPARED_DATASET_CACHE_REBUILD=1 after feature/code changes.)",
                flush=True,
            )
        df, feat_cols = _load_prepared_dataset_cache()
        if os.environ.get("PREP_DRIFT_ON_CACHE_LOAD", "").strip().lower() in {"1", "true", "yes"}:
            from core.trainers.feature_drift import report_prep_drift_vs_saved

            report_prep_drift_vs_saved(df)
        return df, feat_cols
    print(f"\n[*] Preparing LGBM dataset...")
    if prep_needs_tcn_derivatives(parse_prep_layer_targets()):
        ensure_tcn_state_classifier_checkpoint()
    else:
        print(
            "[*] Skipping TCN checkpoint ensure (PREPARED_DATASET_LAYER_TARGETS has no l1c).",
            flush=True,
        )
    df, feat_cols = prepare_lgbm_data(symbols)
    saved = _save_prepared_dataset_cache(df, feat_cols)
    print(f"[*] Prepared dataset cache saved -> {saved}")
    return df, feat_cols


def _artifact_inferred_l1a_outputs(df):
    model, meta = load_l1a_market_encoder()
    outputs = infer_l1a_market_encoder(model, meta, df)
    save_output_cache(outputs, L1A_OUTPUT_CACHE_FILE)
    return outputs


def _artifact_inferred_l1b_outputs(df):
    models, meta = load_l1b_market_descriptor()
    outputs = infer_l1b_market_descriptor(models, meta, df)
    save_output_cache(outputs, L1B_OUTPUT_CACHE_FILE)
    return outputs


def _artifact_inferred_l2_outputs(df, l1a_outputs, l1b_outputs, l1c_outputs=None):
    models, meta = load_l2_trade_decision()
    outputs = infer_l2_trade_decision(models, meta, df, l1a_outputs, l1b_outputs, l1c_outputs)
    save_output_cache(outputs, L2_OUTPUT_CACHE_FILE)
    return outputs


def _print_feature_drift_summary(name: str, frame: pd.DataFrame, cols: list[str]) -> None:
    if "time_key" not in frame.columns:
        return
    ts = pd.to_datetime(frame["time_key"])
    cur = ts >= pd.Timestamp(TEST_END)
    cur_n = int(np.sum(cur))
    if cur_n < 200:
        return
    ref_mode = (os.environ.get("DRIFT_REF_MODE", "train_window") or "train_window").strip().lower()
    baseline_path = os.path.join(MODEL_DIR, f"drift_ref_{name}.pkl")
    ref_stats: dict[str, tuple[float, float, int]] = {}
    ref_label = "train_window"
    ref_n = 0
    if ref_mode in {"prev_holdout", "previous_holdout"} and os.path.exists(baseline_path):
        try:
            with open(baseline_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict):
                for col, row in cached.items():
                    if isinstance(row, dict):
                        q50 = float(row.get("q50", np.nan))
                        sd = float(row.get("sd", np.nan))
                        n = int(row.get("n", 0))
                        if np.isfinite(q50) and np.isfinite(sd) and n >= 50:
                            ref_stats[str(col)] = (q50, sd, n)
                if ref_stats:
                    ref_label = "prev_holdout"
                    ref_n = int(np.median([r[2] for r in ref_stats.values()]))
        except Exception:
            ref_stats = {}
    if not ref_stats:
        ref = (ts >= pd.Timestamp(CAL_END)) & (ts < pd.Timestamp(TEST_END))
        ref_n = int(np.sum(ref))
        if ref_n < 200:
            return
        for col in cols[:10]:
            if col not in frame.columns:
                continue
            x_ref = pd.to_numeric(frame.loc[ref, col], errors="coerce").to_numpy(dtype=np.float64)
            x_ref = x_ref[np.isfinite(x_ref)]
            if x_ref.size < 50:
                continue
            ref_stats[col] = (float(np.quantile(x_ref, 0.5)), float(np.std(x_ref)), int(x_ref.size))
    print(f"\n[*] drift summary ({name}) ref_mode={ref_label} ref≈{ref_n:,} cur={cur_n:,}", flush=True)
    cur_baseline: dict[str, dict[str, float | int]] = {}
    for col in cols[:10]:
        if col not in frame.columns or col not in ref_stats:
            continue
        x_cur = pd.to_numeric(frame.loc[cur, col], errors="coerce").to_numpy(dtype=np.float64)
        x_cur = x_cur[np.isfinite(x_cur)]
        if x_cur.size < 50:
            continue
        q50_ref, sd_ref, _ = ref_stats[col]
        q50_cur = float(np.quantile(x_cur, 0.5))
        drift = abs(q50_cur - q50_ref) / max(sd_ref, 1e-6)
        print(f"    {col}: median_z_drift={drift:.3f}  q50_ref={q50_ref:.4f}  q50_cur={q50_cur:.4f}", flush=True)
        cur_baseline[col] = {"q50": q50_cur, "sd": float(np.std(x_cur)), "n": int(x_cur.size)}
    if cur_baseline:
        try:
            with open(baseline_path, "wb") as f:
                pickle.dump(cur_baseline, f)
        except Exception:
            pass


def _load_threshold_registry_snapshot(meta_file: str) -> dict[str, float]:
    path = os.path.join(MODEL_DIR, meta_file)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            meta = pickle.load(f)
    except Exception:
        return {}
    entries = ((meta.get("threshold_registry") or {}).get("entries") or [])
    snap: dict[str, float] = {}
    for e in entries:
        name = str(e.get("name", "")).strip()
        val = e.get("value")
        if not name:
            continue
        try:
            fv = float(val)
        except Exception:
            continue
        if np.isfinite(fv):
            snap[name] = fv
    return snap


def _print_threshold_registry_drift(layer: str, before: dict[str, float], after: dict[str, float]) -> None:
    if not after:
        return
    keys = sorted(set(before.keys()) & set(after.keys()))
    if not keys:
        return
    print(f"\n[*] threshold drift summary ({layer})", flush=True)
    shown = 0
    for k in keys:
        b = float(before[k])
        a = float(after[k])
        denom = max(abs(b), 1e-8)
        rel = abs(a - b) / denom
        if rel < 0.01:
            continue
        print(f"    {k}: prev={b:.6g} cur={a:.6g} rel_change={rel:.2%}", flush=True)
        shown += 1
        if shown >= 12:
            break
    if shown == 0:
        print("    no material threshold changes (>1%)", flush=True)


def _resolve_l1c_outputs_for_pipeline(df, feat_cols: list[str], sf: str):
    """Train L1c after L1a/L1b when running those stages; otherwise load cache if present."""
    import os

    skip = os.environ.get("L1C_SKIP_PIPELINE", "").strip().lower() in {"1", "true", "yes"}
    cache_path = os.path.join(MODEL_DIR, L1C_OUTPUT_CACHE_FILE)
    if skip:
        if os.path.exists(cache_path):
            print(f"\n[*] L1C_SKIP_PIPELINE: loading {L1C_OUTPUT_CACHE_FILE} ...")
            return load_output_cache(L1C_OUTPUT_CACHE_FILE)
        print("\n[*] L1C_SKIP_PIPELINE: no L1c cache; L2 will run without l1c_* features.")
        return None
    if sf in ("layer1a", "layer1b"):
        l1c_thr_before = _load_threshold_registry_snapshot(L1C_META_FILE)
        logger = setup_logger("layer1c")
        try:
            print("\n[1c] --- Training L1c (causal direction) ---")
            train_l1c_direction(df, feat_cols)
        finally:
            logger.close()
        _print_threshold_registry_drift("l1c", l1c_thr_before, _load_threshold_registry_snapshot(L1C_META_FILE))
        return load_output_cache(L1C_OUTPUT_CACHE_FILE)
    if os.path.exists(cache_path):
        print(f"\n[*] start-from={sf}: loading L1c outputs from {L1C_OUTPUT_CACHE_FILE} ...")
        return load_output_cache(L1C_OUTPUT_CACHE_FILE)
    print("\n[*] No L1c output cache; training L2 without l1c_* features (train L1c separately if needed).")
    return None


def run_lgbm_layers(start_from: str = "layer1"):
    configure_training_runtime()
    n_th = torch.get_num_threads()
    print(f"\n  Thread budget: PyTorch intra-op={n_th}  LightGBM n_jobs={_lgbm_n_jobs()}")

    print("=" * 70)
    print("  Dual-View Stack: L1a + L1b + L2 + L3")
    print("=" * 70)

    sf = start_from.strip().lower()
    if sf == "layer1":
        sf = "layer1a"

    if sf == "layer1c":
        df, feat_cols = _prepare_or_load_lgbm_dataset(
            ["QQQ", "SPY"],
            prefer_cache=True,
            layer1a_frozen_prep_hint=False,
        )
        l1c_thr_before = _load_threshold_registry_snapshot(L1C_META_FILE)
        logger = setup_logger("layer1c")
        try:
            print("\n[1c] --- Training L1c (causal direction) only ---")
            train_l1c_direction(df, feat_cols)
        finally:
            logger.close()
        _print_threshold_registry_drift("l1c", l1c_thr_before, _load_threshold_registry_snapshot(L1C_META_FILE))
        print("\n" + "=" * 70)
        print("  DONE — L1c artifacts saved in lgbm_models/")
        print("=" * 70)
        return

    _l1a_use_prep_cache = os.environ.get("LAYER1A_USE_PREPARED_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    df, feat_cols = _prepare_or_load_lgbm_dataset(
        ["QQQ", "SPY"],
        prefer_cache=(sf != "layer1a") or _l1a_use_prep_cache,
        layer1a_frozen_prep_hint=(sf == "layer1a" and _l1a_use_prep_cache),
    )

    if sf == "layer3":
        print("\n[*] start-from=layer3: load L1a + L2 output caches, train L3 only.")
        l1a_outputs = load_output_cache(L1A_OUTPUT_CACHE_FILE)
        l2_outputs = load_output_cache(L2_OUTPUT_CACHE_FILE)
        l3_thr_before = _load_threshold_registry_snapshot(L3_META_FILE)
        logger = setup_logger("layer3")
        try:
            print("\n[3] --- Training L3 (Exit Manager) ---")
            train_l3_exit_manager(df, l1a_outputs, l2_outputs)
        finally:
            logger.close()
        _print_threshold_registry_drift("l3", l3_thr_before, _load_threshold_registry_snapshot(L3_META_FILE))
        print("\n" + "=" * 70)
        print("  DONE — L3 models saved in lgbm_models/")
        print("=" * 70)
        return

    if sf == "layer1a":
        l1a_thr_before = _load_threshold_registry_snapshot(L1A_META_FILE)
        logger = setup_logger("layer1a")
        try:
            print("\n[1a] --- Training L1a (Sequence Market Encoder) ---")
            train_l1a_market_encoder(df, feat_cols)
        finally:
            logger.close()
        _print_threshold_registry_drift("l1a", l1a_thr_before, _load_threshold_registry_snapshot(L1A_META_FILE))
        print("\n[*] Recomputing downstream-facing L1a outputs from frozen artifact ...")
        l1a_outputs = _artifact_inferred_l1a_outputs(df)
    else:
        print(f"\n[*] start-from={sf}: loading L1a outputs from {L1A_OUTPUT_CACHE_FILE} ...")
        l1a_outputs = load_output_cache(L1A_OUTPUT_CACHE_FILE)
    _print_feature_drift_summary("l1a", l1a_outputs, [c for c in l1a_outputs.columns if c.startswith("l1a_")])

    if sf in ("layer1a", "layer1b"):
        l1b_thr_before = _load_threshold_registry_snapshot(L1B_META_FILE)
        logger = setup_logger("layer1b")
        try:
            print("\n[1b] --- Training L1b (Tabular Market Descriptor) ---")
            train_l1b_market_descriptor(df, feat_cols)
        finally:
            logger.close()
        _print_threshold_registry_drift("l1b", l1b_thr_before, _load_threshold_registry_snapshot(L1B_META_FILE))
        print("\n[*] Recomputing downstream-facing L1b outputs from frozen artifact ...")
        l1b_outputs = _artifact_inferred_l1b_outputs(df)
    else:
        print(f"\n[*] start-from={sf}: loading L1b outputs from {L1B_OUTPUT_CACHE_FILE} ...")
        l1b_outputs = load_output_cache(L1B_OUTPUT_CACHE_FILE)
    _print_feature_drift_summary("l1b", l1b_outputs, [c for c in l1b_outputs.columns if c.startswith("l1b_")])

    l1c_outputs = _resolve_l1c_outputs_for_pipeline(df, feat_cols, sf)

    logger = setup_logger("layer2")
    l2_thr_before = _load_threshold_registry_snapshot(L2_META_FILE)
    try:
        print("\n[2] --- Training L2 (Trade Decision) ---")
        train_l2_trade_decision(df, l1a_outputs, l1b_outputs, l1c_outputs)
    finally:
        logger.close()
    _print_threshold_registry_drift("l2", l2_thr_before, _load_threshold_registry_snapshot(L2_META_FILE))
    print("\n[*] Recomputing downstream-facing L2 outputs from frozen artifact ...")
    l2_outputs = _artifact_inferred_l2_outputs(df, l1a_outputs, l1b_outputs, l1c_outputs)
    _print_feature_drift_summary("l2", l2_outputs, [c for c in l2_outputs.columns if c.startswith("l2_")])

    if sf == "layer2":
        print("\n" + "=" * 70)
        print("  DONE — L2 only (start-from=layer2). Run layer3 separately if needed.")
        print("=" * 70)
        return

    logger = setup_logger("layer3")
    l3_thr_before = _load_threshold_registry_snapshot(L3_META_FILE)
    try:
        print("\n[3] --- Training L3 (Exit Manager) ---")
        train_l3_exit_manager(df, l1a_outputs, l2_outputs)
    finally:
        logger.close()
    _print_threshold_registry_drift("l3", l3_thr_before, _load_threshold_registry_snapshot(L3_META_FILE))

    print("\n" + "=" * 70)
    print("  DONE — Models saved in lgbm_models/")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Dual-view training pipeline (L1a/L1b/L2/L3)")
    parser.add_argument(
        "--start-from",
        type=str,
        choices=["layer1", "layer1a", "layer1b", "layer1c", "layer2", "layer3"],
        default="layer1",
        help="layer1/layer1a: L1a→L1b→L1c→L2→L3 (set L1C_SKIP_PIPELINE=1 to skip L1c train). "
        "layer1b: needs l1a_outputs.pkl, then L1b→L1c→L2→L3. layer1c: prepared cache + L1c only. "
        "layer2: needs l1a/l1b caches; loads l1c_outputs.pkl if present; trains L2 only (no L3). "
        "layer3: needs l1a_outputs.pkl + l2_outputs.pkl, L3 only.",
    )
    args = parser.parse_args()
    run_lgbm_layers(start_from=args.start_from)


if __name__ == "__main__":
    main()
