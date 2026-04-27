from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import torch

from core.training.common.val_metrics_extra import pearson_corr


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(m)) < 3:
        return float("nan")
    s = pd.Series(a[m]).corr(pd.Series(b[m]), method="spearman")
    return float(s) if np.isfinite(s) else float("nan")


def evaluate_l1c(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Val metrics for regression L1c: z-target fit + direction ranking quality."""
    model.eval()
    preds: list[np.ndarray] = []
    yz_l: list[np.ndarray] = []
    y_ret_l: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device)
            yz = batch[1].cpu().numpy().ravel()
            yr = batch[3].cpu().numpy().ravel()
            out = model(xb)
            if isinstance(out, dict):
                pred = out["direction_pred"].view(-1).cpu().numpy()
            else:
                pred = out.view(-1).cpu().numpy()
            preds.append(pred.astype(np.float64))
            yz_l.append(yz.astype(np.float64))
            y_ret_l.append(yr.astype(np.float64))
    pred_z = np.concatenate(preds)
    y_z = np.concatenate(yz_l)
    y_ret = np.concatenate(y_ret_l)
    fin = np.isfinite(pred_z) & np.isfinite(y_z) & np.isfinite(y_ret)
    pred_z = pred_z[fin]
    y_z = y_z[fin]
    y_ret = y_ret[fin]
    if pred_z.size == 0:
        return {"n": 0}

    err = pred_z - y_z
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    try:
        from sklearn.metrics import r2_score

        r2 = float(r2_score(y_z, pred_z)) if len(np.unique(y_z)) > 1 else float("nan")
    except Exception:
        r2 = float("nan")
    spearman_ret = _safe_spearman(pred_z, y_ret)
    spearman_z = _safe_spearman(pred_z, y_z)
    pearson_ret = pearson_corr(pred_z, y_ret)
    sign_acc = float(np.mean((pred_z > 0) == (y_ret > 0)))

    q_edges = np.quantile(pred_z, [0.2, 0.4, 0.6, 0.8]) if pred_z.size else np.array([0, 0, 0, 0], dtype=np.float64)
    q_rows: list[dict[str, float]] = []
    for i in range(5):
        lo = -np.inf if i == 0 else q_edges[i - 1]
        hi = np.inf if i == 4 else q_edges[i]
        mask = (pred_z >= lo) & (pred_z < hi) if i < 4 else (pred_z >= lo)
        qn = int(np.sum(mask))
        q_rows.append(
            {
                "quintile": int(i + 1),
                "n": qn,
                "mean_pred_z": float(np.mean(pred_z[mask])) if qn > 0 else float("nan"),
                "mean_ret": float(np.mean(y_ret[mask])) if qn > 0 else float("nan"),
                "sign_acc": float(np.mean((pred_z[mask] > 0) == (y_ret[mask] > 0))) if qn > 0 else float("nan"),
            }
        )

    return {
        "mae_z": mae,
        "rmse_z": rmse,
        "r2_z": r2,
        "spearman_pred_raw_return": float(spearman_ret) if np.isfinite(spearman_ret) else float("nan"),
        "spearman_pred_target_z": float(spearman_z) if np.isfinite(spearman_z) else float("nan"),
        "pearson_pred_raw_return": float(np.nan_to_num(pearson_ret, nan=0.0)),
        "sign_accuracy_raw_return": sign_acc,
        "pred_z_mean": float(np.mean(pred_z)),
        "pred_z_std": float(np.std(pred_z)),
        "ret_mean": float(np.mean(y_ret)),
        "ret_std": float(np.std(y_ret)),
        "pred_quintiles": q_rows,
        "n": int(len(pred_z)),
    }


def print_l1c_eval_report(metrics: dict[str, Any]) -> None:
    if int(metrics.get("n", 0)) <= 0:
        print("  [L1c] val: no rows", flush=True)
        return
    print(
        f"  [L1c] val regression: MAE={metrics.get('mae_z', float('nan')):.4f}  "
        f"RMSE={metrics.get('rmse_z', float('nan')):.4f}  R2={metrics.get('r2_z', float('nan')):.4f}",
        flush=True,
    )
    print(
        f"  [L1c] val direction quality: spearman(pred,ret)={metrics.get('spearman_pred_raw_return', float('nan')):.4f}  "
        f"spearman(pred,z)={metrics.get('spearman_pred_target_z', float('nan')):.4f}  "
        f"pearson(pred,ret)={metrics.get('pearson_pred_raw_return', float('nan')):.4f}  "
        f"sign_acc={metrics.get('sign_accuracy_raw_return', float('nan')):.4f}",
        flush=True,
    )
    print(
        f"  [L1c] pred_z stats: mean={metrics.get('pred_z_mean', float('nan')):.4f}  std={metrics.get('pred_z_std', float('nan')):.4f}  "
        f"ret mean={metrics.get('ret_mean', float('nan')):.6f}  ret std={metrics.get('ret_std', float('nan')):.6f}",
        flush=True,
    )
    quint_rows = metrics.get("pred_quintiles") or []
    if quint_rows:
        print("  [L1c] pred_z quintiles (monotonicity check):", flush=True)
        for row in quint_rows:
            print(
                f"    Q{int(row['quintile'])}: n={int(row['n']):,}  mean_pred_z={row['mean_pred_z']:.4f}  "
                f"mean_ret={row['mean_ret']:.6f}  sign_acc={row['sign_acc']:.4f}",
                flush=True,
            )
