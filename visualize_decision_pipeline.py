"""
Visualize the full LightGBM + TCN ensemble decision pipeline.

Outputs:
  1. Decision flowchart (pipeline architecture)
  2. Feature importance (both models, top-30)
  3. Sample decision trees from LightGBM
  4. Confidence tier / position weight mapping
"""
from __future__ import annotations

import os
import pickle
import sys

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lgbm_models")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz_output")
os.makedirs(OUT_DIR, exist_ok=True)


def load_models():
    state_model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "state_classifier_4c.txt"))
    bo_model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "breakout_predictor_v2.txt"))
    with open(os.path.join(MODEL_DIR, "state_calibrators.pkl"), "rb") as f:
        state_cals = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "breakout_calibrator.pkl"), "rb") as f:
        bo_cal = pickle.load(f)
    return state_model, state_cals, bo_model, bo_cal


# ─────────────────────────────────────────────────────────────────
# 1. Decision Pipeline Flowchart
# ─────────────────────────────────────────────────────────────────

def draw_pipeline_flowchart():
    fig, ax = plt.subplots(figsize=(22, 30))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 30)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    colors = {
        "input":   "#1f6feb",
        "model":   "#238636",
        "logic":   "#da3633",
        "filter":  "#d29922",
        "output":  "#8b949e",
        "signal":  "#a371f7",
        "exec":    "#f78166",
    }

    def box(x, y, w, h, text, color, fontsize=10, bold=False, alpha=0.9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                              facecolor=color, edgecolor="white", linewidth=1.5, alpha=alpha)
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color="white", weight=weight,
                wrap=True, linespacing=1.4)

    def arrow(x1, y1, x2, y2, label="", color="white"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.3, my, label, fontsize=8, color="#8b949e", va="center")

    # Title
    ax.text(11, 29.3, "LightGBM + TCN Ensemble — Decision Pipeline",
            ha="center", va="center", fontsize=18, color="white", weight="bold")
    ax.text(11, 28.8, "Complete signal generation → execution → exit flow",
            ha="center", va="center", fontsize=11, color="#8b949e")

    # ── Layer 0: Data Input ──
    box(1, 27, 5, 1.2, "1-Min OHLCV Bars\n(SPY / QQQ)", colors["input"], 11, True)
    box(8, 27, 5, 1.2, "ATR(14) Computation\n→ Volatility Baseline", colors["input"], 10)
    box(15, 27, 5.5, 1.2, "add_all_pa_features()\n→ 140 PA Features (5m TF)", colors["input"], 10)
    arrow(6, 27.6, 8, 27.6)
    arrow(13, 27.6, 15, 27.6)

    # ── Layer 1: Model Inference ──
    box(1, 24.5, 6, 1.8, "LightGBM State Classifier\n3-Class: bull / bear / TR\nnum_leaves=127, 2000 rounds\n+ Isotonic Calibration × 3",
        colors["model"], 9, True)
    box(8.5, 24.5, 5, 1.8, "TCN State Classifier\n3-Class temporal model\nSeq window on 5m bars\n+ Softmax probabilities",
        colors["model"], 9, True)
    box(15, 24.5, 5.5, 1.8, "LightGBM Breakout Predictor\nBinary: success / fail\nnum_leaves=63, calibrated\n→ P(breakout succeeds)",
        colors["model"], 9, True)

    arrow(11, 27, 4, 26.3)
    arrow(17, 27, 17.75, 26.3)
    arrow(17, 27, 11, 26.3)

    # ── Layer 2: Ensemble ──
    box(3.5, 22, 8, 1.2, "Ensemble State Probabilities\nP(bull), P(bear), P(TR) = w·TCN + (1-w)·LightGBM\nstate_pred = argmax  |  state_conf = max",
        colors["model"], 9.5)
    arrow(4, 24.5, 7, 23.2)
    arrow(11, 24.5, 8, 23.2)

    box(15, 22.2, 5.5, 0.8, "bo_prob = P(breakout success)\n(Isotonic calibrated)", colors["model"], 9)
    arrow(17.75, 24.5, 17.75, 23)

    # ── Layer 3: Context Features ──
    box(0.5, 19.5, 10, 1.5,
        "5m Context Features\n"
        "prev_5m_ema_slope (trend strength)  |  prev_5m_body_ratio / body_mult (momentum)\n"
        "prev_5m_high / low (range)  |  prev_5m_ema20 (structure)  |  prev_5m_atr (vol)",
        colors["input"], 8.5)
    box(11.5, 19.5, 9, 1.5,
        "PA Pattern Signals\n"
        "H1/H2 (bull PB)  |  L1/L2 (bear PB)  |  wedge_up/down  |  overshoot\n"
        "bo_up/down (breakout)  |  mtr_bull/bear  |  buy/sell_climax",
        colors["input"], 8.5)
    arrow(7.5, 22, 5.5, 21)
    arrow(7.5, 22, 16, 21)

    # ── Layer 4: Signal Generation (4 strategies) ──
    y_sig = 16.5
    box(0.5, y_sig, 4.8, 2.2,
        "Strategy 1\nTrend Pullback\n─────────\n"
        "bull + (H1|H2)\n+ ema_slope > 1.0\n→ trend_pb_long\n"
        "(bear mirror)",
        colors["signal"], 8.5, True)

    box(5.8, y_sig, 4.8, 2.2,
        "Strategy 2\nBreakout\n─────────\n"
        "bo_up + bo_prob ≥ 0.55\n→ bo_long(p=X)\n"
        "bo_down + bo_prob ≥ 0.55\n→ bo_short(p=X)",
        colors["signal"], 8.5, True)

    box(11.1, y_sig, 4.8, 2.2,
        "Strategy 3\nFade / Mean-Rev\n─────────\n"
        "bull + (climax | overshoot\n| wedge) → fade_short\n"
        "(bear mirror\n→ fade_long)",
        colors["signal"], 8.5, True)

    box(16.4, y_sig, 4.8, 2.2,
        "Strategy 4\nMTR (Reversal)\n─────────\n"
        "mtr_bull + bear state\n+ P(bull) > 0.25\n→ mtr_long\n"
        "(mirror → mtr_short)",
        colors["signal"], 8.5, True)

    arrow(5.5, 19.5, 3, 18.7)
    arrow(5.5, 19.5, 8, 18.7)
    arrow(16, 19.5, 13.5, 18.7)
    arrow(16, 19.5, 18.8, 18.7)

    # ── Layer 5: Confidence Gate ──
    box(2, 13.8, 8, 2,
        "Confidence Tiering — Position Weight\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "fade / bo:    0.55→0.6x  0.60→1.0x  0.70→1.3x  0.80→1.5x\n"
        "mtr:          0.55→0.5x  0.65→1.0x  0.80→1.3x\n"
        "trend_pb:     0.65→0.5x  0.70→0.8x  0.80→1.0x\n"
        "Below min → REJECT (weight = 0)",
        colors["filter"], 8.5, True)

    arrow(3, 16.5, 6, 15.8)
    arrow(8, 16.5, 6, 15.8)
    arrow(13.5, 16.5, 6, 15.8)
    arrow(18.8, 16.5, 6, 15.8)

    # ── Layer 6: CVD Filter ──
    box(12, 13.8, 8.5, 2,
        "CVD Divergence Filter\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Long: recent_bull_div OR cvd_pressure > 0\n"
        "Short: recent_bear_div OR cvd_pressure < 0\n"
        "Fail → skip entry",
        colors["filter"], 9, True)
    arrow(6, 13.8, 16, 13.8, "signal passed")

    # ── Layer 7: Entry ──
    box(3, 11, 7, 1.5,
        "Entry Execution\n"
        "Long: price > prev_5m_high + 0.01\n"
        "Short: price < prev_5m_low − 0.01\n"
        "SL = PA stop ± 0.1×ATR (min 0.5×ATR)  |  TP = entry ± 2.0×ATR",
        colors["exec"], 8.5, True)

    box(12, 11, 8.5, 1.5,
        "Time / Session Filters\n"
        "No entry during 11:30–13:30 (garbage time)\n"
        "No entry after 15:30  |  Max 6 trades/day\n"
        "EOD forced exit at 15:50",
        colors["filter"], 8.5)

    arrow(6, 13.8, 6, 12.5)
    arrow(16, 13.8, 16, 12.5)

    # ── Layer 8: Exit Logic ──
    box(2, 8, 18, 2.2,
        "Exit Logic (checked every 1-min bar)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "1. Stop Loss Hit → exit at SL price\n"
        "2. Take Profit Hit → exit at TP price\n"
        "3. Trailing Stop: if MFE > 1.0×ATR → SL = max(ema20 − 0.2×ATR, entry + 0.5×MFE)\n"
        "4. Time Stop: holding > 30 bars → exit at close\n"
        "5. EOD: ≥ 15:50 or last bar of day → exit at close",
        colors["logic"], 8.5, True)
    arrow(6, 11, 11, 10.2)

    # ── Layer 9: Output ──
    box(5, 5.5, 12, 1.5,
        "Trade Record Output\n"
        "entry/exit time & price | direction | raw & weighted return\n"
        "signal type | state confidence | bo probability | exit reason",
        colors["output"], 9, True)
    arrow(11, 8, 11, 7)

    # Legend
    legend_items = [
        ("Data / Features", colors["input"]),
        ("ML Models", colors["model"]),
        ("Signal Strategy", colors["signal"]),
        ("Filters / Gates", colors["filter"]),
        ("Execution", colors["exec"]),
        ("Exit Logic", colors["logic"]),
        ("Output", colors["output"]),
    ]
    for idx, (label, color) in enumerate(legend_items):
        bx = 1 + idx * 3
        by = 4
        rect = FancyBboxPatch((bx, by), 2.5, 0.6, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor="white", linewidth=1, alpha=0.85)
        ax.add_patch(rect)
        ax.text(bx + 1.25, by + 0.3, label, ha="center", va="center",
                fontsize=7.5, color="white", weight="bold")

    path = os.path.join(OUT_DIR, "1_decision_pipeline.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  [1] Pipeline flowchart → {path}")


# ─────────────────────────────────────────────────────────────────
# 2. Feature Importance (both models)
# ─────────────────────────────────────────────────────────────────

def draw_feature_importance(state_model, bo_model, top_n=30):
    fig, axes = plt.subplots(1, 2, figsize=(24, 14))
    fig.patch.set_facecolor("#0d1117")

    for ax, model, title, color in [
        (axes[0], state_model, "State Classifier (3-Class)\nTop Feature Importance (Gain)", "#238636"),
        (axes[1], bo_model, "Breakout Predictor (Binary)\nTop Feature Importance (Gain)", "#1f6feb"),
    ]:
        ax.set_facecolor("#161b22")
        imp = model.feature_importance(importance_type="gain")
        names = model.feature_name()
        df = pd.DataFrame({"feature": names, "importance": imp})
        df = df.sort_values("importance", ascending=True).tail(top_n)

        bars = ax.barh(range(len(df)), df["importance"].values, color=color, alpha=0.85, height=0.7)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["feature"].values, fontsize=8, color="#c9d1d9")
        ax.set_title(title, fontsize=13, color="white", weight="bold", pad=12)
        ax.set_xlabel("Gain", fontsize=10, color="#8b949e")
        ax.tick_params(axis="x", colors="#8b949e")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")

        for bar, val in zip(bars, df["importance"].values):
            ax.text(bar.get_width() + max(df["importance"].values) * 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.0f}", va="center", fontsize=7, color="#8b949e")

    fig.suptitle("LightGBM Feature Importance — Both Models",
                 fontsize=16, color="white", weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(OUT_DIR, "2_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  [2] Feature importance → {path}")


# ─────────────────────────────────────────────────────────────────
# 3. Sample Decision Trees
# ─────────────────────────────────────────────────────────────────

def draw_sample_trees(state_model, bo_model):
    for model, name, tree_idx in [
        (state_model, "state_classifier", 0),
        (state_model, "state_classifier", 1),
        (bo_model, "breakout_predictor", 0),
    ]:
        fig, ax = plt.subplots(figsize=(28, 12))
        fig.patch.set_facecolor("#ffffff")
        try:
            lgb.plot_tree(model, tree_index=tree_idx, ax=ax, show_info=["split_gain", "leaf_count"])
            ax.set_title(f"{name} — Tree #{tree_idx}", fontsize=14, weight="bold")
            path = os.path.join(OUT_DIR, f"3_tree_{name}_t{tree_idx}.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            print(f"  [3] Tree {name} #{tree_idx} → {path}")
        except Exception as e:
            print(f"  [3] Tree plot skipped ({name} #{tree_idx}): {e}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# 4. Confidence Tier Heatmap
# ─────────────────────────────────────────────────────────────────

def draw_confidence_tiers():
    tiers = {
        "fade / bo":  [(0.55, 0.60, 0.6), (0.60, 0.70, 1.0), (0.70, 0.80, 1.3), (0.80, 1.00, 1.5)],
        "mtr":        [(0.55, 0.65, 0.5), (0.65, 0.80, 1.0), (0.80, 1.00, 1.3)],
        "trend_pb":   [(0.65, 0.70, 0.5), (0.70, 0.80, 0.8), (0.80, 1.00, 1.0)],
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    conf_range = np.arange(0.40, 1.001, 0.005)
    strategy_colors = {"fade / bo": "#1f6feb", "mtr": "#a371f7", "trend_pb": "#238636"}

    for strat, tier_list in tiers.items():
        weights = []
        for c in conf_range:
            w = 0.0
            for lo, hi, wt in tier_list:
                if lo <= c < hi:
                    w = wt
                    break
            weights.append(w)
        ax.plot(conf_range, weights, linewidth=2.5, label=strat,
                color=strategy_colors[strat], marker="", alpha=0.9)
        ax.fill_between(conf_range, weights, alpha=0.1, color=strategy_colors[strat])

    ax.axhline(y=0, color="#30363d", linewidth=0.8)
    ax.set_xlabel("State Confidence (max probability)", fontsize=11, color="#c9d1d9")
    ax.set_ylabel("Position Weight", fontsize=11, color="#c9d1d9")
    ax.set_title("Confidence Tiering → Position Weight by Signal Type",
                 fontsize=14, color="white", weight="bold")
    ax.legend(fontsize=11, loc="upper left", facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#c9d1d9")
    ax.set_xlim(0.40, 1.0)
    ax.set_ylim(-0.1, 1.7)
    ax.tick_params(colors="#8b949e")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")

    for strat, tier_list in tiers.items():
        for lo, hi, wt in tier_list:
            ax.annotate(f"{wt}x", xy=(lo, wt), fontsize=8, color=strategy_colors[strat],
                        xytext=(5, 8), textcoords="offset points", weight="bold")

    ax.axvline(x=0.55, color="#da3633", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(0.555, 1.55, "fade/bo/mtr\nmin threshold", fontsize=7.5, color="#da3633", alpha=0.7)
    ax.axvline(x=0.65, color="#d29922", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(0.655, 1.55, "trend_pb\nmin threshold", fontsize=7.5, color="#d29922", alpha=0.7)

    path = os.path.join(OUT_DIR, "4_confidence_tiers.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  [4] Confidence tiers → {path}")


# ─────────────────────────────────────────────────────────────────
# 5. Signal Strategy Decision Tree (text-based logic)
# ─────────────────────────────────────────────────────────────────

def draw_signal_decision_tree():
    fig, ax = plt.subplots(figsize=(20, 16))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9, alpha=0.9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25",
                              facecolor=color, edgecolor="white", linewidth=1.2, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color="white", wrap=True, linespacing=1.3)

    def arrow(x1, y1, x2, y2, label="", color="#8b949e"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.15, my + 0.15, label, fontsize=7.5, color=color)

    ax.text(10, 15.5, "Signal Generation Decision Tree (per 5-min bar)",
            ha="center", fontsize=15, color="white", weight="bold")

    # Root
    box(7, 13.5, 6, 1, "New 5m bar?\nposition == 0 & !EOD & !garbage_time\n& trades_today < 6", "#30363d", 9)

    # State check
    box(2, 11, 4.5, 1.2, "state_pred == bull?\nema_slope > 1.0?", "#238636", 9)
    box(8, 11, 4.5, 1.2, "bo_up or bo_down?\nbo_prob ≥ 0.55?", "#1f6feb", 9)
    box(14, 11, 4.5, 1.2, "state_pred == bear?\nema_slope < −1.0?", "#238636", 9)

    arrow(8, 13.5, 4.25, 12.2, "check\nstrategies")
    arrow(10, 13.5, 10.25, 12.2)
    arrow(12, 13.5, 16.25, 12.2)

    # Trend PB
    box(0.5, 8.5, 3.5, 1.5, "H1 or H2 setup?\n→ trend_pb_long\n\nL1 or L2 setup?\n→ trend_pb_short", "#a371f7", 8.5)
    arrow(3.5, 11, 2.25, 10, "yes")

    # Breakout
    box(7, 8.5, 5, 1.5, "bo_up → bo_long(p=X)\nbo_down → bo_short(p=X)\n\nRequires bo_prob ≥ 0.55", "#1f6feb", 8.5)
    arrow(10.25, 11, 9.5, 10, "yes")

    # Fade
    box(0.5, 5.5, 5.5, 2, "Bull state +\n(buy_climax & strong_body)\nor overshoot_up or wedge_up\n→ fade_short\n\nBear state + mirror\n→ fade_long", "#d29922", 8.5)
    arrow(3.5, 8.5, 3.25, 7.5, "also check")

    # MTR
    box(7, 5.5, 5, 2, "mtr_bull ready?\nstate == bear & P(bull) > 0.25\n→ mtr_long\n\nmtr_bear ready?\nstate == bull & P(bear) > 0.25\n→ mtr_short", "#da3633", 8.5)
    arrow(9.5, 8.5, 9.5, 7.5, "also check")

    # Gate
    box(14, 8, 5, 2.5, "Confidence Gate\n━━━━━━━━━━━\nweight = tier_lookup(\n  state_conf, signal_type\n)\n\nweight == 0 → SKIP\nweight > 0 → setup entry\n+ set SL / TP", "#f78166", 9)
    arrow(5.5, 9, 14, 9, "signal chosen")
    arrow(12, 6.5, 14, 8, "signal chosen")

    # CVD
    box(14, 4.5, 5, 1.2, "CVD Divergence Check\nConfirms direction\nor pressure alignment", "#d29922", 9)
    arrow(16.5, 8, 16.5, 5.7)

    # Entry
    box(14, 2.5, 5, 1.2, "ENTER TRADE\nStop order fill on\nnext bar high/low break", "#238636", 10, 0.95)
    arrow(16.5, 4.5, 16.5, 3.7, "pass")

    path = os.path.join(OUT_DIR, "5_signal_decision_tree.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  [5] Signal decision tree → {path}")


# ─────────────────────────────────────────────────────────────────
# 6. Model Architecture Summary
# ─────────────────────────────────────────────────────────────────

def draw_model_summary(state_model, bo_model):
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(8, 9.5, "Model Architecture Summary", ha="center",
            fontsize=16, color="white", weight="bold")

    state_trees = state_model.num_trees()
    state_feats = state_model.num_feature()
    bo_trees = bo_model.num_trees()
    bo_feats = bo_model.num_feature()

    # State Classifier card
    card_text = (
        f"State Classifier (3-Class)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Type: LightGBM GBDT\n"
        f"Trees: {state_trees} (early-stopped from 2000)\n"
        f"Features: {state_feats}\n"
        f"Leaves: 127 per tree\n"
        f"Learning Rate: 0.05\n"
        f"L1: 0.1  |  L2: 1.0\n"
        f"Feature Fraction: 0.8\n"
        f"Min Child Samples: 100\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Calibration: Isotonic × 3 classes\n"
        f"Output: P(bull), P(bear), P(TR)"
    )
    rect = FancyBboxPatch((0.5, 2.5), 7, 6.3, boxstyle="round,pad=0.4",
                          facecolor="#161b22", edgecolor="#238636", linewidth=2)
    ax.add_patch(rect)
    ax.text(4, 5.65, card_text, ha="center", va="center",
            fontsize=10, color="#c9d1d9", family="monospace", linespacing=1.5)

    # Breakout Predictor card
    card_text2 = (
        f"Breakout Predictor (Binary)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Type: LightGBM GBDT\n"
        f"Trees: {bo_trees} (early-stopped from 2000)\n"
        f"Features: {bo_feats}\n"
        f"Leaves: 63 per tree\n"
        f"Learning Rate: 0.05\n"
        f"L1: 0.1  |  L2: 1.0\n"
        f"Feature Fraction: 0.8\n"
        f"Min Child Samples: 50\n"
        f"scale_pos_weight: auto\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Calibration: Isotonic (binary)\n"
        f"Output: P(breakout success)"
    )
    rect2 = FancyBboxPatch((8.5, 2.5), 7, 6.3, boxstyle="round,pad=0.4",
                           facecolor="#161b22", edgecolor="#1f6feb", linewidth=2)
    ax.add_patch(rect2)
    ax.text(12, 5.65, card_text2, ha="center", va="center",
            fontsize=10, color="#c9d1d9", family="monospace", linespacing=1.5)

    # TCN note
    tcn_path = os.path.join(MODEL_DIR, "tcn_meta.pkl")
    if os.path.exists(tcn_path):
        with open(tcn_path, "rb") as f:
            meta = pickle.load(f)
        tcn_info = (
            f"TCN: input={meta.get('input_size','?')}, "
            f"channels={meta.get('num_channels','?')}, "
            f"kernel={meta.get('kernel_size','?')}, "
            f"seq_len={meta.get('seq_len','?')}"
        )
    else:
        tcn_info = "TCN: model file not found"
    ax.text(8, 1.8, f"Ensemble: {tcn_info}",
            ha="center", fontsize=9, color="#8b949e", style="italic")

    path = os.path.join(OUT_DIR, "6_model_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  [6] Model summary → {path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Decision Pipeline Visualization")
    print("=" * 60)

    state_model, state_cals, bo_model, bo_cal = load_models()

    draw_pipeline_flowchart()
    draw_feature_importance(state_model, bo_model)
    draw_sample_trees(state_model, bo_model)
    draw_confidence_tiers()
    draw_signal_decision_tree()
    draw_model_summary(state_model, bo_model)

    print(f"\n  All visualizations saved to: {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
