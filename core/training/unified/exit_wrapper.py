"""
OOS exit via shared ``UnifiedL2L3Net``: **decision score = close_prob** only
(``sigmoid(exit_close_logit)``). Urgency/value heads are trained to shape the backbone but are not
combined at inference (see ``predict_detailed`` for diagnostics).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import torch

from core.training.unified.position_features import build_position_matrix, build_position_matrix_from_dataframe


def _as_numpy_X(X: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def _index_map(cols: list[str]) -> dict[str, int]:
    return {c: i for i, c in enumerate(cols)}


def _env_oos_drift() -> bool:
    return (os.environ.get("L2_UNIFIED_OOS_DRIFT", "") or "").strip().lower() in ("1", "true", "yes", "on")


def log_exit_distribution(close_probs: np.ndarray, *, prefix: str = "[OOS exit]") -> None:
    """Log mean, std, percentiles and fraction > 0.5 for a batch of close probabilities."""
    a = np.asarray(close_probs, dtype=np.float64).ravel()
    if a.size < 1:
        line = f"  {prefix} n=0 (skip)"
        print(line, flush=True)
        try:
            from core.training.logging.metrics_file_log import append_oos_unified_log

            append_oos_unified_log(line)
        except Exception:  # noqa: BLE001
            pass
        return
    line = (
        f"  {prefix} n={a.size} mean={a.mean():.3f} std={a.std():.3f} "
        f"p10={float(np.percentile(a, 10)):.3f} p50={float(np.percentile(a, 50)):.3f} "
        f"p90={float(np.percentile(a, 90)):.3f} frac_gt_0.5={float((a > 0.5).mean()):.3f}"
    )
    print(line, flush=True)
    try:
        from core.training.logging.metrics_file_log import append_oos_unified_log

        append_oos_unified_log(line)
    except Exception:  # noqa: BLE001
        pass


class UnifiedExitWrapper:
    """Map policy rows to market/regime/position columns and run ``return_all=True`` forward."""

    def __init__(
        self,
        unified_model: torch.nn.Module,
        l3_feature_cols: list[str],
        l2_market_col_names: list[str],
        l2_regime_col_names: list[str],
        *,
        n_pos: int,
        device: str | torch.device = "cpu",
        train_input_stats: dict[str, Any] | None = None,
        value_target_norm: dict[str, Any] | None = None,
    ) -> None:
        self.model = unified_model.eval()
        self.l3_feature_cols: list[str] = list(l3_feature_cols)
        self._l3_i = _index_map(self.l3_feature_cols)
        self._m_names = list(l2_market_col_names)
        self._r_names = list(l2_regime_col_names)
        self.m_ix = [self._l3_i[c] for c in self._m_names if c in self._l3_i]
        self.r_ix = [self._l3_i[c] for c in self._r_names if c in self._l3_i]
        self.n_pos = int(n_pos)
        self.device = torch.device(device)
        self._train_input_stats: dict[str, Any] | None = train_input_stats
        self._value_target_norm: dict[str, Any] | None = value_target_norm
        self.model.to(self.device)

    def _maybe_log_input_drift(self, t_m: torch.Tensor, t_r: torch.Tensor, t_p: torch.Tensor) -> None:
        st = self._train_input_stats
        if st is None or not _env_oos_drift():
            return
        m_mean = st.get("train_market_mean") or []
        m_std = st.get("train_market_std") or []
        r_mean = st.get("train_regime_mean") or []
        r_std = st.get("train_regime_std") or []
        p_mean = st.get("train_position_mean") or []
        p_std = st.get("train_position_std") or []

        def _one(name: str, x: torch.Tensor, ref_m: list, ref_s: list) -> dict[str, float | int]:
            if not ref_m or x.shape[1] < 1:
                return {"n_drifted": 0, "max_z": 0.0}
            rm = torch.as_tensor(np.asarray(ref_m, dtype=np.float32), device=x.device, dtype=torch.float32)
            rs = torch.as_tensor(np.asarray(ref_s, dtype=np.float32), device=x.device, dtype=torch.float32).clamp_min(
                1e-8
            )
            if rm.numel() != x.shape[1]:
                return {"n_drifted": 0, "max_z": 0.0}
            z = ((x.mean(0) - rm) / rs).abs()
            n_d = int((z > 3.0).sum().item())
            if n_d > 0:
                msg = (
                    f"  [OOS drift] {name}: {n_d}/{x.shape[1]} cols >3σ from train mean "
                    f"(max_z={float(z.max().item()):.3f})"
                )
                print(msg, flush=True)
                try:
                    from core.training.logging.metrics_file_log import append_oos_unified_log

                    append_oos_unified_log(msg)
                except Exception:  # noqa: BLE001
                    pass
            return {"n_drifted": n_d, "max_z": float(z.max().item())}

        _one("market", t_m, m_mean, m_std)
        _one("regime", t_r, r_mean, r_std)
        _one("position", t_p, p_mean, p_std)

    @torch.inference_mode()
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:  # noqa: A003 — sklearn-like API
        """Return ``close_prob`` per row (same tensor as training target for the exit BCE head)."""
        det = self.predict_detailed(X)
        return det["exit_score"]

    @torch.inference_mode()
    def predict_detailed(self, X: np.ndarray | pd.DataFrame) -> dict[str, np.ndarray]:
        n = len(X) if isinstance(X, pd.DataFrame) else int(np.asarray(X).shape[0])
        if n == 0:
            return {
                "exit_score": np.zeros(0, dtype=np.float64),
                "close_prob": np.zeros(0, dtype=np.float64),
                "urgency": np.zeros(0, dtype=np.float64),
                "remaining_pnl": np.zeros(0, dtype=np.float64),
                "remaining_bars": np.zeros(0, dtype=np.float64),
                "pnl_quantiles": np.zeros((0, 5), dtype=np.float64),
            }
        if isinstance(X, pd.DataFrame):
            xmat = _as_numpy_X(X)
            pos = build_position_matrix_from_dataframe(X, self.n_pos)
        else:
            xn = np.asarray(X, dtype=np.float32, order="C")
            if xn.shape[1] != len(self.l3_feature_cols):
                raise ValueError(
                    f"UnifiedExitWrapper: expected {len(self.l3_feature_cols)} L3 feature columns, got {xn.shape[1]}"
                )
            xmat = xn
            pos, _ = build_position_matrix(xmat, self.l3_feature_cols, self.n_pos)
        m_dim = int(self.model.n_market)
        r_dim = int(self.model.n_regime)
        mx = np.zeros((n, m_dim), dtype=np.float32)
        rx = np.zeros((n, r_dim), dtype=np.float32)
        for j, c in enumerate(self._m_names):
            if j < m_dim and c in self._l3_i:
                mx[:, j] = xmat[:, self._l3_i[c]]
        for j, c in enumerate(self._r_names):
            if j < r_dim and c in self._l3_i:
                rx[:, j] = xmat[:, self._l3_i[c]]
        t_m = torch.as_tensor(mx, device=self.device, dtype=torch.float32)
        t_r = torch.as_tensor(rx, device=self.device, dtype=torch.float32)
        t_p = torch.as_tensor(pos, device=self.device, dtype=torch.float32)
        self._maybe_log_input_drift(t_m, t_r, t_p)
        out = self.model(t_m, t_r, t_p, return_all=True)
        close_p = (
            torch.sigmoid(out["exit_close_logit"].squeeze(-1)).detach().float().cpu().numpy().astype(np.float64, copy=False)
        )
        rem = out["value_remaining_pnl"].detach().float().cpu().numpy().ravel()
        bars = out["value_remaining_bars"].detach().float().cpu().numpy().ravel()
        qu = out["value_pnl_quantiles"].detach().float().cpu().numpy()
        urgency = torch.sigmoid(out["exit_urgency_logit"].squeeze(-1)).detach().float().cpu().numpy().astype(
            np.float64, copy=False
        )
        vn = self._value_target_norm
        if vn and vn.get("enabled", True):
            # Point / quantile head targets are all «remaining PnL» in Phase2 — use remaining_pnl μ/σ (not bars).
            pm = float(vn.get("remaining_pnl_mean", 0.0))
            ps = max(float(vn.get("remaining_pnl_std", 1.0)), 1e-8)
            bm = float(vn.get("remaining_bars_mean", 0.0))
            bs = max(float(vn.get("remaining_bars_std", 1.0)), 1e-8)
            rem = rem * ps + pm
            bars = bars * bs + bm
            qu = qu * ps + pm
        # Countdown-to-exit in bars is non-negative; model / denorm can dip slightly below 0.
        bars = np.maximum(np.asarray(bars, dtype=np.float64), 0.0)
        return {
            "exit_score": close_p,
            "close_prob": close_p,
            "urgency": np.asarray(urgency, dtype=np.float64).ravel(),
            "remaining_pnl": rem.astype(np.float64, copy=False),
            "remaining_bars": bars,
            "pnl_quantiles": qu,
        }

    @torch.inference_mode()
    def route_contribution(self, X: np.ndarray | pd.DataFrame) -> dict[str, float]:
        """Per-route ablation: mean |Δ exit_close_logit| when zeroing one encoder input group."""
        if isinstance(X, pd.DataFrame):
            xmat = _as_numpy_X(X)
            pos = build_position_matrix_from_dataframe(X, self.n_pos)
        else:
            xn = np.asarray(X, dtype=np.float32, order="C")
            if xn.shape[1] != len(self.l3_feature_cols):
                raise ValueError(
                    f"UnifiedExitWrapper: expected {len(self.l3_feature_cols)} L3 feature columns, got {xn.shape[1]}"
                )
            xmat = xn
            pos, _ = build_position_matrix(xmat, self.l3_feature_cols, self.n_pos)
        n = int(xmat.shape[0])
        m_dim = int(self.model.n_market)
        r_dim = int(self.model.n_regime)
        mx = np.zeros((n, m_dim), dtype=np.float32)
        rx = np.zeros((n, r_dim), dtype=np.float32)
        for j, c in enumerate(self._m_names):
            if j < m_dim and c in self._l3_i:
                mx[:, j] = xmat[:, self._l3_i[c]]
        for j, c in enumerate(self._r_names):
            if j < r_dim and c in self._l3_i:
                rx[:, j] = xmat[:, self._l3_i[c]]
        t_m = torch.as_tensor(mx, device=self.device, dtype=torch.float32)
        t_r = torch.as_tensor(rx, device=self.device, dtype=torch.float32)
        t_p = torch.as_tensor(pos, device=self.device, dtype=torch.float32)
        base = self.model(t_m, t_r, t_p, return_all=True)
        bl = base["exit_close_logit"].squeeze(-1)
        contribs: dict[str, float] = {}
        for name, (zm, zr, zp) in (
            ("market", (torch.zeros_like(t_m), t_r, t_p)),
            ("regime", (t_m, torch.zeros_like(t_r), t_p)),
            ("position", (t_m, t_r, torch.zeros_like(t_p))),
        ):
            ab = self.model(zm, zr, zp, return_all=True)
            d = (bl - ab["exit_close_logit"].squeeze(-1)).abs().mean()
            contribs[name] = round(float(d.item()), 4)
        return contribs
