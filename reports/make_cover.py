"""Wide cover image for the portfolio card, built from the real results."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
NAVY, RED, GREY, BG = "#1D3557", "#C44E52", "#A8B2C1", "#FFFFFF"

df = pd.read_csv(HERE.parent / "data" / "customers.csv")
scored = pd.read_csv(HERE / "customers_scored.csv")
by_contract = df.assign(c=df["churn"].eq("Yes")).groupby("contract")["c"].mean().mul(100)
by_band = scored.assign(c=scored["churn"].eq("Yes")).groupby("risk_band")["c"].mean().mul(100)
by_band = by_band.reindex(["High", "Medium", "Low"])

fig = plt.figure(figsize=(12, 3.6), facecolor=BG)
fig.text(0.03, 0.82, "Customer Churn Prediction", fontsize=18, weight="bold", color=NAVY)
fig.text(0.03, 0.7, "Telecom · who should the retention team call first?", fontsize=11, color="#6c757d")
for i, (v, l) in enumerate([("0.72", "ROC-AUC (5-fold CV)"), ("59%", "churners detected"),
                            ("54%", "of churners in top 30% risk")]):
    fig.text(0.03 + i * 0.125, 0.36, v, fontsize=26, weight="bold", color=RED)
    fig.text(0.03 + i * 0.125, 0.24, l, fontsize=9, color=NAVY, wrap=True)

for k, (data, title) in enumerate([(by_contract, "Churn rate by contract"),
                                   (by_band, "Actual churn by predicted risk")]):
    ax = fig.add_axes([0.45 + k * 0.28, 0.16, 0.23, 0.62])
    bars = ax.bar(data.index, data.values, color=[RED] + [GREY] * (len(data) - 1))
    ax.bar_label(bars, fmt="%.0f%%", fontsize=9, color=NAVY, padding=2)
    ax.set_title(title, fontsize=10, color=NAVY, loc="left")
    ax.set_ylim(0, 65)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(colors=NAVY, labelsize=8.5)
fig.text(0.97, 0.04, "synthetic data · 500 customers", ha="right", fontsize=7.5, color="#9aa0a6")
fig.savefig(HERE / "cover.png", dpi=110, facecolor=BG)
