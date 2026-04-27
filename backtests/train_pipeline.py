"""Stack training pipeline (L1a/L1b/L2/L3).

Per-stage layer logs: ``layer1a.log``, ``layer1b.log``. **Unified L2 + L3** (one PyTorch model + policy registry) uses a
**single** tee file ``logs/l2l3_unified.log``: L2 training prints, L2 ``print_l2_unified_meta_readable`` summary, L3
registry one-liner; full meta stays in ``lgbm_models/*.pkl`` (stdout during ``setup_logger("unified")``).

**tqdm** (epoch bars, L3 bar scan, …) is sent to the **real terminal** via ``core.training.common.lgbm_utils._tqdm_stream``,
not to ``l2l3_unified.log``. For a full stream capture, redirect: ``... 2>&1 | tee run.log``.

OOS-only diagnostics: ``logs/oos_unified.log`` (append; exit dist / drift / ablation when env flags are on).
"""
import argparse
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from core.training.common.constants import (
    CAL_END,
    L1A_META_FILE,
    L1A_OUTPUT_CACHE_FILE,
    L1B_META_FILE,
    L1B_OUTPUT_CACHE_FILE,
    L2_META_FILE,
    L2_OUTPUT_CACHE_FILE,
    L3_META_FILE,
    MODEL_DIR,
    PREPARED_DATASET_CACHE_FILE,
    TEST_END,
)
from core.training.prep.data_prep import prepare_dataset as prepare_lgbm_data
from core.training.prep.feature_registry import parse_prep_layer_targets, prep_needs_tcn_derivatives
from core.training.prep.tcn_data_prep import ensure_tcn_state_classifier_checkpoint
from core.training.l1a import (
    infer_l1a_market_encoder,
    load_l1a_market_encoder,
    train_l1a_market_encoder,
)
from core.training.l1b import (
    infer_l1b_market_descriptor,
    load_l1b_market_descriptor,
    train_l1b_market_descriptor,
)
from core.training.l2 import infer_l2_trade_decision, load_l2_trade_decision, train_l2_trade_decision
from core.training.unified.policy_meta import train_l3_exit_manager
from core.training.common.lgbm_utils import configure_training_runtime, _lgbm_n_jobs
from core.training.l1b.l1a_bridge import (
    attach_l1a_outputs_to_df,
    configure_l1b_l1a_allowlist_from_tier,
    extend_feat_cols_with_l1a,
    l1a_feature_cols_from_l1b_meta,
    l1b_l1a_feature_names_for_tier,
    l1b_l1a_feature_tier,
    l1b_l1a_inputs_enabled,
    meta_expects_l1a_features,
)
from core.training.common.stack_v2_common import load_output_cache, save_output_cache


def _prepare_df_and_feat_for_l1b_train(df, feat_cols, l1a_outputs):
    """When ``L1B_USE_L1A_FEATURES=1``, merge L1a cache columns and extend ``feat_cols`` / allowlist."""
    if not l1b_l1a_inputs_enabled() or l1b_l1a_feature_tier() == "none":
        return df, feat_cols
    if l1a_outputs is None:
        raise ValueError("L1b L1a features enabled but l1a_outputs is None (load l1a_outputs.pkl first).")
    configure_l1b_l1a_allowlist_from_tier()
    tier_cols = l1b_l1a_feature_names_for_tier()
    df_m = attach_l1a_outputs_to_df(df, l1a_outputs, cols=tier_cols)
    feat_m = extend_feat_cols_with_l1a(feat_cols, df_m)
    return df_m, feat_m


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
    "unified": "l2l3_unified.log",
}
# Human label for 开始/结束 lines
_LAYER_LOG_LABEL = {
    "layer1a": "L1a",
    "layer1b": "L1b",
    "layer1c": "L1c",
    "layer2": "L2",
    "layer3": "L3",
    "unified": "L2+L3",
}


def _layer_log_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _print_layer_stage_boundary(layer_key: str, *, is_start: bool) -> None:
    """One line at the start or end of a stage; written to the layer teed log and the terminal."""
    name = _LAYER_LOG_LABEL.get(layer_key, layer_key)
    tag = "开始" if is_start else "结束"
    print(f"  [{name}] {tag}  {_layer_log_timestamp()}", flush=True)


def setup_logger(layer_key: str):
    """Tee stdout to the layer log file. First body line is ``[Lx] 开始  <timestamp>``; call ``close_layer_log`` to close."""
    if layer_key not in _LAYER_LOG_FILES:
        raise ValueError(f"setup_logger: unknown layer_key={layer_key!r}; expected one of {sorted(_LAYER_LOG_FILES)}")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, _LAYER_LOG_FILES[layer_key])
    lg = Logger(log_file)
    print(f"\n>> Layer log (overwrite): {log_file}")
    _print_layer_stage_boundary(layer_key, is_start=True)
    if layer_key in ("layer2", "layer3", "unified"):
        print(
            "  (本文件为 stdout: 不含 tqdm 进度条, 进度在终端; 见 lgbm_utils._tqdm_stream)",
            flush=True,
        )
    return lg


def close_layer_log(lg: Logger, layer_key: str) -> None:
    """Emit ``[Lx] 结束 <timestamp>`` then restore stdout (must match ``setup_logger`` key)."""
    _print_layer_stage_boundary(layer_key, is_start=False)
    lg.close()


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
            from core.training.prep.feature_drift import report_prep_drift_vs_saved

            report_prep_drift_vs_saved(df)
        return df, feat_cols
    print(f"\n[*] Preparing LGBM dataset...")
    if prep_needs_tcn_derivatives(parse_prep_layer_targets()):
        ensure_tcn_state_classifier_checkpoint()
    else:
        print(
            "[*] Skipping TCN checkpoint ensure (prep targets do not require TCN derivatives; "
            "add `tcn` or legacy `l1c` to PREPARED_DATASET_LAYER_TARGETS to enable).",
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
    df_work = df
    if meta_expects_l1a_features(meta):
        l1a_model, l1a_meta = load_l1a_market_encoder()
        l1a_out = infer_l1a_market_encoder(l1a_model, l1a_meta, df.copy())
        l1_cols = l1a_feature_cols_from_l1b_meta(meta)
        df_work = attach_l1a_outputs_to_df(df, l1a_out, cols=l1_cols if l1_cols else None)
    outputs = infer_l1b_market_descriptor(models, meta, df_work)
    save_output_cache(outputs, L1B_OUTPUT_CACHE_FILE)
    return outputs


def _artifact_inferred_l2_outputs(df, l1a_outputs, l1b_outputs):
    models, meta = load_l2_trade_decision()
    outputs = infer_l2_trade_decision(models, meta, df, l1a_outputs, l1b_outputs)
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


def run_lgbm_layers(start_from: str = "layer1"):
    configure_training_runtime()
    n_th = torch.get_num_threads()
    print(f"\n  Thread budget: PyTorch intra-op={n_th}  LightGBM n_jobs={_lgbm_n_jobs()}")

    print("=" * 70)
    print("  Stack: L1a + L1b → L2 (unified + Phase-2) → L3 policy meta auto-written after L2 (set L2_AUTO_L3_META=0 to skip)")
    print("=" * 70)

    sf = start_from.strip().lower()
    if sf == "layer1":
        sf = "layer1a"

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
        print("\n[*] start-from=layer3: load L1a + l2_outputs cache; write L3 policy registry only (no L2 retrain).")
        l1a_outputs = load_output_cache(L1A_OUTPUT_CACHE_FILE)
        l2_outputs = load_output_cache(L2_OUTPUT_CACHE_FILE)
        l3_thr_before = _load_threshold_registry_snapshot(L3_META_FILE)
        logger = setup_logger("unified")
        try:
            print("\n[3] --- L3 policy feature registry ---")
            train_l3_exit_manager(df, l1a_outputs, l2_outputs)
            _print_threshold_registry_drift("l3", l3_thr_before, _load_threshold_registry_snapshot(L3_META_FILE))
        finally:
            close_layer_log(logger, "unified")
        print("\n" + "=" * 70)
        print("  DONE — L3 policy metadata (l2_unified_exit) saved under lgbm_models/")
        print("=" * 70)
        return

    if sf == "layer1a":
        l1a_thr_before = _load_threshold_registry_snapshot(L1A_META_FILE)
        logger = setup_logger("layer1a")
        try:
            print("\n[1a] --- Training L1a (Sequence Market Encoder) ---")
            train_l1a_market_encoder(df, feat_cols)
        finally:
            close_layer_log(logger, "layer1a")
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
            df_l1b, feat_l1b = _prepare_df_and_feat_for_l1b_train(df, feat_cols, l1a_outputs)
            train_l1b_market_descriptor(df_l1b, feat_l1b)
        finally:
            close_layer_log(logger, "layer1b")
        _print_threshold_registry_drift("l1b", l1b_thr_before, _load_threshold_registry_snapshot(L1B_META_FILE))
        print("\n[*] Recomputing downstream-facing L1b outputs from frozen artifact ...")
        l1b_outputs = _artifact_inferred_l1b_outputs(df)
    else:
        print(f"\n[*] start-from={sf}: loading L1b outputs from {L1B_OUTPUT_CACHE_FILE} ...")
        l1b_outputs = load_output_cache(L1B_OUTPUT_CACHE_FILE)
    _print_feature_drift_summary("l1b", l1b_outputs, [c for c in l1b_outputs.columns if c.startswith("l1b_")])

    # Single tee for L2 train + L2 drift/recompute + optional L3 registry (readable/slim log; full meta in pkl).
    logger = setup_logger("unified")
    l2_thr_before = _load_threshold_registry_snapshot(L2_META_FILE)
    try:
        print("\n[2] --- Training L2 (Trade Decision) ---")
        train_l2_trade_decision(df, l1a_outputs, l1b_outputs)
        _print_threshold_registry_drift("l2", l2_thr_before, _load_threshold_registry_snapshot(L2_META_FILE))
        print("\n[*] Recomputing downstream-facing L2 outputs from frozen artifact ...")
        l2_outputs = _artifact_inferred_l2_outputs(df, l1a_outputs, l1b_outputs)
        _print_feature_drift_summary("l2", l2_outputs, [c for c in l2_outputs.columns if c.startswith("l2_")])

        # L3: policy registry only; set L2_AUTO_L3_META=0 to skip.
        l3_autorun = (os.environ.get("L2_AUTO_L3_META", "1") or "1").strip().lower() not in ("0", "false", "no", "off")
        if l3_autorun:
            l3_thr_before = _load_threshold_registry_snapshot(L3_META_FILE)
            print("\n[3] --- L3 policy feature registry (auto after L2) ---")
            train_l3_exit_manager(df, l1a_outputs, l2_outputs)
            _print_threshold_registry_drift("l3", l3_thr_before, _load_threshold_registry_snapshot(L3_META_FILE))
        else:
            print(
                "\n  [pipeline] L2_AUTO_L3_META=0: skipping L3 policy registry; run ``--start-from layer3`` to write l3_*.pkl later.",
                flush=True,
            )
    finally:
        close_layer_log(logger, "unified")

    if sf == "layer2":
        print("\n" + "=" * 70)
        print("  DONE — L2 + L3 policy registry (l2_unified_exit).  Artifacts in lgbm_models/")
        print("=" * 70)
        return

    print("\n" + "=" * 70)
    print("  DONE — Models saved under lgbm_models/")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Dual-view training pipeline (L1a/L1b/L2/L3)")
    parser.add_argument(
        "--start-from",
        type=str,
        choices=["layer1", "layer1a", "layer1b", "layer2", "layer3"],
        default="layer1",
        help="layer1/layer1a: L1a→L1b→L2→L3. layer1b: needs l1a_outputs.pkl, then L1b→L2→L3. "
        "layer2: needs l1a/l1b caches; trains L2 then writes L3 policy registry (L2_AUTO_L3_META=0 to skip; same as old layer3 I/O). "
        "layer3: refresh L3 policy registry only (no L2; needs l1a + l2_output caches). "
        "(Archived L1c: archive/train_layer1c_only.py + archive/l1c/.)",
    )
    args = parser.parse_args()
    run_lgbm_layers(start_from=args.start_from)


if __name__ == "__main__":
    main()
