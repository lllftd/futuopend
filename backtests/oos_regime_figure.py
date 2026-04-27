"""
Shared: 1m K 线 + 3 行诊断：A 价格 + **L1a 持仓区间 regime 色块底纹(与色例同色系)**、B 累计&回撤(与 A 同 K bar 横轴)、C 本窗各 regime 收益细柱直方(密度)。
A 上 L1a 色例与底纹、C 中柱色一致。统计摘要置于图最下方(大字)。

Walkforward: ``fold_*/regime_charts/{sym}_week_*.png``。
OOS: ``oos_chart_*.png`` 单文件。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

# Mirrors oos backtest OOS_VOL_REGIME_ROUTER_MAP
REGIME_ROUTER: dict[int, str] = {
    0: "SHORT_STRADDLE_R0",
    1: "LONG_STRADDLE_R1",
    2: "GAMMA_SCALP_R2",
    3: "SHORT_STRADDLE_R3",
    4: "IRON_CONDOR_R4",
}

REGIME_COLOR: dict[int, tuple[float, float, float]] = {
    0: (0.2, 0.45, 0.85),
    1: (0.95, 0.55, 0.15),
    2: (0.25, 0.75, 0.35),
    3: (0.55, 0.35, 0.75),
    4: (0.85, 0.35, 0.35),
}

# 叠在 R0–R4 半透明色块 / 多色直方上时，文字统一用白描边，避免与任意 regime 底纹同色系
PE_ON_REGIME_TINT: list = [pe.withStroke(linewidth=2.4, foreground="1.0", alpha=0.92)]


def _set_text_path_effects_legends(leg, effects: list | None = None) -> None:
    if leg is None:
        return
    eff = effects if effects is not None else PE_ON_REGIME_TINT
    for t in leg.get_texts():
        t.set_path_effects(eff)
    t0 = leg.get_title()
    if t0 is not None:
        t0.set_path_effects(eff)


# 全图统一配色（与上表一致）
C = {
    "price_wick": "0.25",
    "cum_ret": "#e65100",
    "drawdown": "#c0392b",
    "dd_fill": "#e74c3c",
    "win": "#2ecc71",
    "loss": "#e74c3c",
    "winrate": "#2980b9",
    "z_act": "#d35400",
    "z_pred": "#1f6dad",
    "grid": "#bdc3c7",
    "event_line": "#c0392b",
}

# 仅当 ``event_markers is None`` 时用于标注；在 OHLC 时间窗外交叠的项会自动跳过
_DEFAULT_EVENT_MARKERS: dict[str, str] = {
    "2025-01-27": "DeepSeek 冲击",
    "2025-03-04": "关税预期升温",
    "2025-04-02": "Liberation Day",
    "2025-04-07": "深回撤日",
    "2025-04-09": "关税暂停 90d",
}


def _resolve_csv(data_dir: Path, symbol: str) -> Path:
    for name in (f"{symbol}_labeled_v2.csv", f"{symbol.lower()}_labeled_v2.csv", f"{symbol}.csv"):
        p = data_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No CSV for {symbol} under {data_dir}")


def df_price_to_1m_ohlc(df: pd.DataFrame) -> pd.DataFrame | None:
    """In-memory OOS bar frame (time_key + OHLC) → 按行保留的分钟 OHLC，不 resample；缺列则 None。"""
    req = ("time_key", "open", "high", "low", "close")
    if not all(c in df.columns for c in req):
        return None
    d = df[list(req)].copy()
    d["time_key"] = pd.to_datetime(d["time_key"])
    for c in ("open", "high", "low", "close"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["time_key", "open", "high", "low", "close"], how="any")
    if d.empty:
        return None
    d = d.sort_values("time_key").drop_duplicates(subset=["time_key"], keep="last")
    return d.reset_index(drop=True)


def _load_ohlc_1m_from_csv(path: Path, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    use = ["time_key", "open", "high", "low", "close"]
    df = pd.read_csv(path, usecols=lambda c: c in set(use))
    df["time_key"] = pd.to_datetime(df["time_key"])
    df = df[(df["time_key"] >= t0) & (df["time_key"] < t1)]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time_key", "open", "high", "low", "close"], how="any")
    if df.empty:
        return df
    df = df.sort_values("time_key").drop_duplicates(subset=["time_key"], keep="last")
    return df.reset_index(drop=True)


def _regime_rgba(rid: int, *, alpha: float) -> tuple[float, float, float, float]:
    c = REGIME_COLOR.get(int(rid), (0.5, 0.5, 0.5))
    return (c[0], c[1], c[2], alpha)


def _merge_regime_segments(trades_sym: pd.DataFrame) -> list[dict[str, Any]]:
    t = trades_sym.sort_values("entry_time").reset_index(drop=True)
    if t.empty:
        return []
    segs: list[dict[str, Any]] = []
    cur: dict | None = None
    for _, row in t.iterrows():
        es = pd.Timestamp(row["entry_time"])
        xs = pd.Timestamp(row["exit_time"])
        rid = int(row["entry_regime_id"])
        if cur is None:
            cur = {
                "start": es,
                "end": xs,
                "reg": rid,
                "returns": [float(row["return"])],
                "exit_probs": [float(row.get("exit_prob", np.nan))],
                "deadlines": [float(row.get("deadline_pnl_atr", np.nan))],
                "strategies": [str(row.get("strategy_type", ""))],
            }
            continue
        if rid == cur["reg"] and es >= pd.Timestamp(cur["end"]) - pd.Timedelta(seconds=1):
            cur["end"] = max(pd.Timestamp(cur["end"]), xs)
            cur["returns"].append(float(row["return"]))
            cur["exit_probs"].append(float(row.get("exit_prob", np.nan)))
            cur["deadlines"].append(float(row.get("deadline_pnl_atr", np.nan)))
            cur["strategies"].append(str(row.get("strategy_type", "")))
        else:
            segs.append(cur)
            cur = {
                "start": es,
                "end": xs,
                "reg": rid,
                "returns": [float(row["return"])],
                "exit_probs": [float(row.get("exit_prob", np.nan))],
                "deadlines": [float(row.get("deadline_pnl_atr", np.nan))],
                "strategies": [str(row.get("strategy_type", ""))],
            }
    if cur is not None:
        segs.append(cur)
    return segs


def _regime_id_for_timestamp(segs: list[dict[str, Any]], t: pd.Timestamp) -> int | None:
    for seg in segs:
        if pd.Timestamp(seg["start"]) <= t <= pd.Timestamp(seg["end"]):
            return int(seg["reg"])
    return None


def _bar_index_for_times(ohlc: pd.DataFrame, times: pd.Series) -> np.ndarray:
    """将时刻映射到**最近的不晚于该时刻**的 K 线索引 (0..n-1)。"""
    n = len(ohlc)
    if n == 0:
        return np.array([], dtype=np.float64)
    tk = ohlc["time_key"].to_numpy(dtype="datetime64[ns]")
    ts = pd.to_datetime(times).to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(tk, ts, side="right") - 1
    idx = np.clip(idx, 0, n - 1).astype(np.float64)
    return idx


def _draw_regime_tints_bar_index(
    ax,
    ohlc: pd.DataFrame,
    segs: list[dict[str, Any]],
    *,
    alpha: float,
) -> None:
    """A 子图底纹：在 **K 线索号**上为「有持仓的 bar」着与图例同色的半透明条带（x=data, y=axes 满高）。"""
    n = len(ohlc)
    if n == 0 or not segs:
        return
    tr_xy = blended_transform_factory(ax.transData, ax.transAxes)
    for i0, i1, rid in _consecutive_regime_bar_runs(ohlc, segs, n):
        c = _regime_rgba(int(rid), alpha=alpha)
        ax.axvspan(
            i0 - 0.5,
            i1 + 0.5,
            ymin=0.0,
            ymax=1.0,
            facecolor=c,
            edgecolor="none",
            clip_on=True,
            zorder=0.25,
            transform=tr_xy,
        )


def _draw_candles_bar_index(ax, ohlc: pd.DataFrame) -> None:
    """横轴 = K 线序号，去掉无 K 的日历空白；实体宽度约 0.6。"""
    w = 0.6
    for i in range(len(ohlc)):
        row = ohlc.iloc[i]
        o, h, l_ol, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        col = "#2ca02c" if c >= o else "#d62728"
        ax.plot([i, i], [l_ol, h], color="0.25", linewidth=0.45, zorder=3)
        y0, y1 = (o, c) if c >= o else (c, o)
        ax.add_patch(
            Rectangle(
                (i - w / 2, y0),
                w,
                y1 - y0,
                facecolor=col,
                edgecolor="0.25",
                linewidth=0.3,
                zorder=3,
            )
        )


def _apply_bar_index_xaxis(axes, ohlc: pd.DataFrame) -> None:
    """共享 K bar 序号为 x，刻度显示对应时间 (稀疏)。``ohlc`` 须已按 time_key 升序，左=早、右=晚。"""
    n = len(ohlc)
    if n == 0:
        return
    n_ticks = min(10, n)
    tick_pos = np.unique(np.round(np.linspace(0, n - 1, n_ticks)).astype(int))
    tick_pos = np.sort(tick_pos)
    t0 = pd.Timestamp(ohlc["time_key"].iloc[0])
    t1 = pd.Timestamp(ohlc["time_key"].iloc[-1])
    same_year = t0.year == t1.year
    fmt = "%m-%d %H:%M" if same_year else "%Y-%m-%d"
    tick_labs = [pd.Timestamp(ohlc["time_key"].iloc[i]).strftime(fmt) for i in tick_pos]
    for ax in axes:
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labs, rotation=12, ha="right", fontsize=7)


def iter_five_day_windows(
    t0: pd.Timestamp | str,
    t1: pd.Timestamp | str,
    *,
    days: int = 5,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """非重叠、左闭右开 ``[a,b)``，步长为 ``days`` 个**自然日**。"""
    a0 = pd.Timestamp(t0).floor("D")
    b0 = pd.Timestamp(t1)
    if b0 <= a0:
        return []
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = a0
    step = pd.Timedelta(days=days)
    while cur < b0:
        nxt = min(cur + step, b0)
        if nxt > cur:
            out.append((cur, nxt))
        cur = nxt
    return out


def _consecutive_regime_bar_runs(
    ohlc: pd.DataFrame, segs: list[dict[str, Any]], n_bars: int
) -> list[tuple[int, int, int]]:
    """K 线序号上、着色相同的连续区间 (i0, i1 含) 与 regime id；无 regime 的 bar 打断连续性。"""
    rids: list[int | None] = []
    for i in range(n_bars):
        t_i = pd.Timestamp(ohlc["time_key"].iloc[i])
        rids.append(_regime_id_for_timestamp(segs, t_i))
    out: list[tuple[int, int, int]] = []
    i = 0
    while i < n_bars:
        r = rids[i]
        if r is None:
            i += 1
            continue
        j = i + 1
        while j < n_bars and rids[j] == r:
            j += 1
        out.append((i, j - 1, int(r)))
        i = j
    return out


def _exit_pnl_in_window_bars_pct(ts: pd.DataFrame, t_lo: pd.Timestamp, t_hi: pd.Timestamp) -> float:
    """本窗内 exit 落在 [t_lo, t_hi] 的成交收益合计 (% 点)。"""
    if ts.empty or "exit_time" not in ts.columns or "return" not in ts.columns:
        return 0.0
    t = ts.copy()
    ex = pd.to_datetime(t["exit_time"])
    m = (ex >= t_lo) & (ex <= t_hi)
    s = t.loc[m, "return"].to_numpy(dtype=np.float64)
    if s.size == 0:
        return 0.0
    return float(np.nansum(s) * 100.0)


def _sharpe_trades(returns: np.ndarray) -> float:
    r = returns[np.isfinite(returns)]
    if r.size < 2:
        return float("nan")
    mu, sd = float(np.mean(r)), float(np.std(r, ddof=1))
    if sd <= 1e-12 or not np.isfinite(sd):
        return float("nan")
    return mu / sd * np.sqrt(float(r.size))


def _draw_regime_return_distribution(ax, ts: pd.DataFrame) -> str:
    """
    本窗内各 L1a regime 的**单笔**收益**密度**直方：横轴 = 收益率 (%)，纵轴 = 归一化密度 (density=True)、
    分箱较密 (柱窄)、同 bin 边；各 regime 半透叠画。
    """
    if ts.empty or "entry_regime_id" not in ts.columns:
        ax.text(
            0.5,
            0.5,
            "无成交",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="0.15",
            path_effects=PE_ON_REGIME_TINT,
        )
        ax.set_xlabel("单笔收益率 (%)", fontsize=8)
        ax.set_ylabel("密度", fontsize=8)
        return ""
    t = ts.copy()
    t["r_pct"] = t["return"].to_numpy(dtype=np.float64) * 100.0
    allr = t["r_pct"].to_numpy()
    if allr.size == 0 or not np.all(np.isfinite(allr)):
        ax.text(
            0.5,
            0.5,
            "无有效收益",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="0.15",
            path_effects=PE_ON_REGIME_TINT,
        )
        ax.set_xlabel("单笔收益率 (%)", fontsize=8)
        ax.set_ylabel("密度", fontsize=8)
        return ""

    lo, hi = float(np.min(allr)), float(np.max(allr))
    if hi - lo < 1e-8:
        lo, hi = lo - 0.5, hi + 0.5
    else:
        pad = 0.04 * (hi - lo)
        lo, hi = lo - pad, hi + pad
    # 更细分箱，柱体窄；上限随样本量略增，避免过粗
    n_bins = int(
        np.clip(9.0 * int(np.sqrt(max(len(allr), 1))), 60, 220)
    )
    edges = np.linspace(lo, hi, n_bins + 1)

    lines: list[str] = ["regime  n  win%  mean%  sum%  std%  ----------------"]
    drawn = 0
    for rid in range(5):
        sub = t[t["entry_regime_id"].astype(int) == int(rid)]
        r = sub["r_pct"].to_numpy(dtype=np.float64)
        if r.size == 0:
            continue
        w = float(np.mean(r > 0.0) * 100.0)
        lines.append(
            f"  R{rid}  {r.size:3d}  {w:5.1f}  {float(np.mean(r)):+6.2f}  {float(np.sum(r)):+6.2f}  {float(np.std(r, ddof=1) if r.size > 1 else 0.0):5.2f}"
        )
        c = REGIME_COLOR.get(int(rid), (0.5, 0.5, 0.5))
        rname = REGIME_ROUTER.get(int(rid), f"R{rid}").split("_")[0]
        ax.hist(
            r,
            bins=edges,
            density=True,
            alpha=0.32,
            color=c,
            label=f"R{rid} {rname} (n={r.size})",
            histtype="bar",
            rwidth=0.75,
            align="mid",
            edgecolor="0.3",
            linewidth=0.12,
            zorder=1 + int(rid),
        )
        drawn += 1

    if drawn == 0:
        ax.text(
            0.5,
            0.5,
            "本窗无 regime 分布",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="0.15",
            path_effects=PE_ON_REGIME_TINT,
        )

    ax.axvline(0.0, color="0.4", linewidth=0.6, zorder=0, linestyle="--")
    ax.set_xlabel("单笔收益率 (%)", fontsize=8)
    ax.set_ylabel("密度", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    if h:
        legh = ax.legend(
            h,
            l,
            loc="lower left",
            fontsize=5.5,
            ncol=1,
            framealpha=0.9,
            title="L1a regime",
        )
        for t in legh.get_texts():
            t.set_color("0.12")
        tti = legh.get_title()
        if tti is not None:
            tti.set_color("0.12")
        _set_text_path_effects_legends(legh)
    stat_txt = "\n".join(lines)
    ax.text(
        0.99,
        0.98,
        stat_txt,
        transform=ax.transAxes,
        fontsize=5.0,
        family="monospace",
        ha="right",
        va="top",
        color="0.12",
        path_effects=PE_ON_REGIME_TINT,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="0.97", alpha=0.9, edgecolor="0.45"),
    )
    return stat_txt


def render_regime_dashboard_figure(
    *,
    sym: str,
    ohlc: pd.DataFrame,
    ts: pd.DataFrame,
    out_path: Path,
    title_line: str = "",
    metrics_legend: str | None = None,
    roll_window: int = 50,
    event_markers: dict[str, str] | None = None,
) -> bool:
    """
    3 行：A 价格(含 L1a 持仓 regime 底纹，与 L1a 色例色一致)、B 累计+回撤(与 A 同 K bar 横轴)、C 各 regime 收益细柱直方(密度)。

    ``roll_window`` 保留在签名中以便兼容，已不再绘制滚动胜率子图。

    底部汇总：本窗(约 5 日)交易次数、胜率、合计收益、最大回撤(步进)、简易夏普；可选 ``metrics_legend`` 追加一行。
    """
    _ = roll_window  # 保留入参，兼容旧调用
    if ohlc is None or ohlc.empty:
        return False
    if "time_key" in ohlc.columns:
        ohlc = (
            ohlc.sort_values("time_key", ascending=True)
            .drop_duplicates(subset=["time_key"], keep="last")
            .reset_index(drop=True)
        )
    tnum = ohlc["time_key"]
    n_bars = len(ohlc)
    t0_win, t1_win = pd.Timestamp(tnum.min()), pd.Timestamp(tnum.max())
    segs = _merge_regime_segments(ts) if not ts.empty else []
    if event_markers is None:
        ev_src = _DEFAULT_EVENT_MARKERS
    else:
        ev_src = event_markers
    events_use = {k: v for k, v in ev_src.items() if t0_win <= pd.Timestamp(k) < t1_win}

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Arial Unicode MS",
        "Heiti TC",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    rc_style = {
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.color": C["grid"],
    }
    plt.rcParams.update(rc_style)

    fig = plt.figure(figsize=(20, 12), dpi=110)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.8, 1.7, 1.55], hspace=0.18)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax_hist = fig.add_subplot(gs[2, 0])
    for ax in (ax0, ax1, ax_hist):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax0.minorticks_off()
    ax1.minorticks_off()

    _REGIME_ALPHA = 0.30

    panel_tags = [
        "A. 价格 (1m K) & L1a 色块/色例(持仓 regime，与 C 同色系)",
        "B. 累计 PnL & 回撤带",
        "C. 各 regime 收益直方(细柱,横=return%、纵=密度)",
    ]

    _draw_regime_tints_bar_index(ax0, ohlc, segs, alpha=_REGIME_ALPHA)
    _draw_candles_bar_index(ax0, ohlc)
    ax0.set_ylabel("价格 (1m OHLC)", fontsize=9)
    ax0.set_title(title_line or f"{sym}  |  跨式+mtm  |  1m 诊断", fontsize=11)
    leg_patches: list[Patch] = []
    for rid in range(5):
        c = REGIME_COLOR.get(rid, (0.5, 0.5, 0.5))
        rname = REGIME_ROUTER.get(rid, f"R{rid}")
        short = f"R{rid} {rname.split('_')[0]}"
        leg_patches.append(
            Patch(facecolor=c, edgecolor="0.35", alpha=_REGIME_ALPHA, label=short)
        )
    leg0 = ax0.legend(
        handles=leg_patches,
        bbox_to_anchor=(1.0, 1.0),
        loc="upper right",
        fontsize=6.5,
        ncol=1,
        framealpha=0.92,
        title="L1a regime",
        title_fontsize=7,
    )
    if leg0:
        leg0.set_zorder(5)
        for t in leg0.get_texts():
            t.set_color("0.10")
        tti = leg0.get_title()
        if tti is not None:
            tti.set_color("0.10")
        _set_text_path_effects_legends(leg0)
    ylim0 = ax0.get_ylim()
    for dstr, lab in events_use.items():
        dt = pd.Timestamp(dstr)
        if not (t0_win <= dt < t1_win):
            continue
        ix = float(_bar_index_for_times(ohlc, pd.Series([dt]))[0])
        ax0.axvline(ix, color=C["event_line"], ls="--", lw=0.85, alpha=0.7, zorder=2.5)
        ax0.annotate(
            lab,
            xy=(ix, ylim0[1]),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=6.5,
            rotation=32,
            ha="left",
            va="bottom",
            color="#4a0e0e",
            fontweight="bold",
            clip_on=True,
            path_effects=PE_ON_REGIME_TINT,
        )
    ax0.set_ylim(ylim0)
    ax0.text(
        0.01,
        0.95,
        panel_tags[0],
        transform=ax0.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color="0.10",
        va="top",
        path_effects=PE_ON_REGIME_TINT,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="0.5"),
    )

    # 同色连续区段下缘：本段 exit 在本窗的合计收益 (%)
    if not ts.empty:
        y_lo, y_hi = ax0.get_ylim()
        yr = y_hi - y_lo
        ax0.set_ylim(y_lo - 0.16 * yr, y_hi)
        y_lo, y_hi = ax0.get_ylim()
        yr = y_hi - y_lo
        y_txt = y_lo + 0.022 * yr
        # 与 R0–R4 色块解耦的深色+白描边(各段盈亏统一处理)
        _col_win_txt = "#0b5345"
        _col_loss_txt = "#1a5276"
        _col_flat_txt = "#212f3d"
        runs = _consecutive_regime_bar_runs(ohlc, segs, n_bars)
        # 长窗区段极多，全部标会挤成一团；长窗/多段时提高门槛或下采样
        n_run = len(runs)
        min_w = 1
        if n_bars > 8_000 or n_run > 50:
            min_w = max(5, n_bars // 400)  # 约每窗最多 ~400 个标注槽位
        if n_run > 60:
            step = max(1, n_run // 45)
            runs = [runs[j] for j in range(0, n_run, step)]
        for k, (i0, i1, _rid) in enumerate(runs):
            if (i1 - i0 + 1) < min_w:
                continue
            t_lo = pd.Timestamp(ohlc["time_key"].iloc[i0])
            t_hi = pd.Timestamp(ohlc["time_key"].iloc[i1])
            pnl = _exit_pnl_in_window_bars_pct(ts, t_lo, t_hi)
            xc = 0.5 * (i0 + i1)
            if pnl > 1e-4:
                msg = f"盈 +{pnl:.2f}%"
                tcol = _col_win_txt
            elif pnl < -1e-4:
                msg = f"亏 {pnl:.2f}%"
                tcol = _col_loss_txt
            else:
                msg = "平 0.00%"
                tcol = _col_flat_txt
            y_use = y_txt + (0.0 if (k % 2) == 0 else 0.03 * yr) + (0.01 * yr) * ((k // 2) % 3)
            ax0.text(
                xc,
                y_use,
                msg,
                ha="center",
                va="bottom",
                fontsize=6.2 if n_bars > 10_000 else 7.0,
                color=tcol,
                fontweight="bold",
                zorder=4,
                path_effects=PE_ON_REGIME_TINT,
            )

    max_dd = float("nan")
    rets1 = np.array([], dtype=np.float64)
    if not ts.empty:
        tt0 = ts.sort_values("exit_time").reset_index(drop=True)
        rets1 = tt0["return"].to_numpy(dtype=np.float64)
        r0 = rets1 * 100.0
        cum0 = np.cumsum(r0)
        run0 = np.maximum.accumulate(cum0)
        max_dd = float(np.min(cum0 - run0))
        sh = _sharpe_trades(rets1)
    else:
        sh = float("nan")

    if not ts.empty:
        tt = ts.sort_values("exit_time").reset_index(drop=True)
        r = tt["return"].to_numpy(dtype=np.float64) * 100.0
        cum = np.cumsum(r)
        run_m = np.maximum.accumulate(cum)
        dd = cum - run_m
        ex_ix = _bar_index_for_times(ohlc, tt["exit_time"])
        ax1.plot(
            ex_ix,
            cum,
            color=C["cum_ret"],
            linewidth=1.5,
            drawstyle="steps-post",
            label="累计收益 % (步进)",
            zorder=3,
        )
        ax1.fill_between(
            ex_ix,
            0.0,
            dd,
            step="post",
            color=C["dd_fill"],
            alpha=0.38,
            zorder=1,
        )
        ax1.axhline(0.0, color="0.35", linewidth=0.55, linestyle="--", zorder=0)
        leg1 = ax1.legend(
            loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=7.5, framealpha=0.92, ncol=1
        )
        for t in leg1.get_texts():
            t.set_color("0.12")
        _set_text_path_effects_legends(leg1)
    ax1.set_ylabel("累计 & 回撤 (%)", fontsize=9)
    ax1.set_xlabel("K 线 bar 索引 (无K 的时段已压缩)", fontsize=9)
    ax1.text(
        0.01,
        0.95,
        panel_tags[1],
        transform=ax1.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color="0.10",
        va="top",
        path_effects=PE_ON_REGIME_TINT,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="0.5"),
    )

    _draw_regime_return_distribution(ax_hist, ts)
    ax_hist.text(
        0.01,
        0.97,
        panel_tags[2],
        transform=ax_hist.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color="0.10",
        va="top",
        ha="left",
        path_effects=PE_ON_REGIME_TINT,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="0.5"),
    )

    plt.setp(ax0.get_xticklabels(), visible=False)
    ax0.tick_params(axis="x", labelsize=8, rotation=10)
    ax1.tick_params(axis="x", labelsize=8, rotation=10)
    ax_hist.tick_params(axis="x", labelsize=8, rotation=0)
    _apply_bar_index_xaxis([ax0, ax1], ohlc)

    if not ts.empty and len(ts) >= 1:
        tt_s = ts.sort_values("exit_time")
        n_t = len(tt_s)
        win_r = (tt_s["return"] > 0).mean() * 100.0
        tot = float(tt_s["return"].sum() * 100.0)
        md_s = f"{max_dd:.1f}%" if np.isfinite(max_dd) else "n/a"
        sh_s = f"{sh:.2f}" if np.isfinite(sh) else "n/a"
        summary = (
            f"标的 {sym}  |  本窗交易: {n_t}  |  胜率: {win_r:.1f}%  |  本窗合计收益: {tot:+.2f}%  "
            f"|  最大回撤(步进): {md_s}  |  夏普(简·按笔): {sh_s}"
        )
    else:
        summary = f"本窗无成交  |  标的 {sym}"
    extra = (metrics_legend or "").strip()
    if extra:
        footer_block = f"{summary}\n{extra}"
    else:
        footer_block = summary
    n_lines = max(1, footer_block.count("\n") + 1)
    # 底边：多行指标与 C 子图 x 轴、summary 分块，用 va=bottom 自下往上排，避免与坐标轴/彼此重叠
    fig.text(
        0.5,
        0.01,
        footer_block,
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="0.18",
        transform=fig.transFigure,
        linespacing=1.18,
    )
    try:
        fig.align_ylabels([ax0, ax1, ax_hist])
    except (ValueError, RuntimeError):
        pass
    _bottom = min(0.44, 0.09 + 0.017 * n_lines)
    fig.subplots_adjust(top=0.92, bottom=_bottom, left=0.08, right=0.95)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=125)
    plt.close(fig)
    print(f"  [OOS plot] saved (regime dashboard) -> {out_path}", flush=True)
    return True


def plot_regime_candles_for_walkforward(
    sym: str,
    trades: pd.DataFrame,
    data_dir: Path,
    out_dir: Path,
    title_suffix: str = "",
    event_markers: dict[str, str] | None = None,
) -> bool:
    """
    按每 5 个自然日切窗，在 ``out_dir / regime_charts / {sym}_week_....png`` 出图。
    """
    for c in ("walkforward_fold",):
        if c in trades.columns:
            trades = trades.drop(columns=[c])
    ts = trades[trades["symbol"] == sym].copy() if "symbol" in trades.columns else trades
    if ts.empty:
        return False
    t0 = pd.Timestamp(ts["entry_time"].min()) - pd.Timedelta(minutes=5)
    t1 = pd.Timestamp(ts["exit_time"].max()) + pd.Timedelta(minutes=5)
    try:
        raw_path = _resolve_csv(data_dir, sym)
    except (OSError, FileNotFoundError) as e:
        print(f"  [regime fig] {sym}: {e}", flush=True)
        return False
    ohlc = _load_ohlc_1m_from_csv(raw_path, t0, t1)
    if ohlc.empty:
        print(f"  [regime fig] {sym}: no OHLC in range", flush=True)
        return False
    h = f"{title_suffix}  |  " if (title_suffix or "").strip() else ""
    t_lo = pd.Timestamp(ohlc["time_key"].min())
    t_hi = pd.Timestamp(ohlc["time_key"].max())
    n_ok = 0
    for w0, w1 in iter_five_day_windows(t_lo, t_hi, days=5):
        sub_oh = ohlc[(ohlc["time_key"] >= w0) & (ohlc["time_key"] < w1)]
        if sub_oh.empty:
            continue
        sub_ts = ts[(ts["exit_time"] >= w0) & (ts["exit_time"] < w1)].copy()
        wtag = f"week_{w0.strftime('%Y%m%d')}_{w1.strftime('%Y%m%d')}"
        chart_dir = out_dir / "regime_charts"
        chart_dir.mkdir(parents=True, exist_ok=True)
        out_p = chart_dir / f"{sym}_{wtag}.png"
        if render_regime_dashboard_figure(
            sym=sym,
            ohlc=sub_oh,
            ts=sub_ts,
            out_path=out_p,
            title_line=f"{h}{sym}  |  {wtag}  |  跨式+mtm  |  1m 诊断",
            metrics_legend=None,
            event_markers=event_markers,
        ):
            n_ok += 1
    if n_ok == 0:
        print(f"  [regime fig] {sym}: no weekly figure written", flush=True)
        return False
    return True


def plot_regime_candles_for_oos(
    symbol: str,
    df_price: pd.DataFrame,
    trades: pd.DataFrame,
    out_path: Path,
    *,
    full_title: str,
    stats_txt: str,
    event_markers: dict[str, str] | None = None,
) -> bool:
    """OOS: OHLC from in-memory ``df_price``; same file names ``oos_chart_{sym}.png``."""
    ohlc = df_price_to_1m_ohlc(df_price)
    if ohlc is None or ohlc.empty:
        return False
    if trades is None or trades.empty:
        return render_regime_dashboard_figure(
            sym=symbol,
            ohlc=ohlc,
            ts=pd.DataFrame(),
            out_path=out_path,
            title_line=full_title,
            metrics_legend=stats_txt,
            event_markers=event_markers,
        )
    return render_regime_dashboard_figure(
        sym=symbol,
        ohlc=ohlc,
        ts=trades,
        out_path=out_path,
        title_line=full_title,
        metrics_legend=stats_txt,
        event_markers=event_markers,
    )
