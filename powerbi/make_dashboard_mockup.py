"""
Power BI-style dashboard page (design mockup) built from the real model output.
Layout and visuals are the ones to rebuild in Power BI on reports/customers_scored.csv.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
NAVY, RED, AMBER, GREEN, GREY, INK, CANVAS = "#1B2A41", "#D64545", "#F2A541", "#3BA37A", "#B8C2CC", "#252423", "#F3F2F1"
plt.rcParams["font.family"] = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]

df = pd.read_csv(ROOT / "data" / "customers.csv")
df["churned"] = df["churn"].eq("Yes")
scored = pd.read_csv(ROOT / "reports" / "customers_scored.csv").merge(
    df[["customer_id", "internet_service", "payment_method"]], on="customer_id")
active_high = scored[(scored["risk_band"] == "High") & (scored["churn"] == "No")]

fig = plt.figure(figsize=(16, 9), facecolor=CANVAS)
W, H = 16, 9


def box(x, y, w, h, fc="white", ec="#E1DFDD"):
    fig.patches.append(FancyBboxPatch((x / W, y / H), w / W, h / H, boxstyle="round,pad=0,rounding_size=0.004",
                                      transform=fig.transFigure, fc=fc, ec=ec, lw=0.8, zorder=0))


def visual(x, y, w, h, title, left=0.25):
    """White visual container with a Power BI-like title; returns axes inside it."""
    box(x, y, w, h)
    fig.text((x + 0.18) / W, (y + h - 0.33) / H, title, fontsize=11, color=INK, weight="semibold")
    ax = fig.add_axes([(x + left) / W, (y + 0.3) / H, (w - left - 0.3) / W, (h - 0.95) / H])
    ax.set_zorder(5)
    ax.set_facecolor("white")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#C8C6C4")
    ax.tick_params(colors="#605E5C", labelsize=9, length=0)
    return ax


# ---- header + slicers ----
fig.patches.append(Rectangle((0, (H - 0.95) / H), 1, 0.95 / H, transform=fig.transFigure, fc=NAVY, zorder=0))
fig.text(0.3 / W, (H - 0.48) / H, "Customer Churn & Retention", fontsize=19, color="white", weight="bold", va="center")
fig.text(0.3 / W, (H - 0.8) / H, "Telecom · 500 customers · churn model scored on all customers",
         fontsize=9.5, color="#C9D3DF", va="center")
for i, (label, value) in enumerate([("Contract", "All"), ("Internet service", "All"), ("Risk band", "All")]):
    x = 9.0 + i * 2.3
    fig.text(x / W, (H - 0.3) / H, label, fontsize=8, color="#C9D3DF")
    fig.patches.append(FancyBboxPatch((x / W, (H - 0.8) / H), 2.1 / W, 0.36 / H, boxstyle="round,pad=0,rounding_size=0.003",
                                      transform=fig.transFigure, fc="white", ec="none", zorder=1))
    fig.text((x + 0.12) / W, (H - 0.62) / H, value, fontsize=9.5, color=INK, va="center", zorder=2)
    fig.text((x + 1.95) / W, (H - 0.62) / H, "▾", fontsize=9, color="#605E5C", va="center", ha="right", zorder=2)

# ---- KPI cards ----
kpis = [
    ("500", "Customers", INK),
    (f"{df['churned'].mean():.1%}", "Churn rate", RED),
    ("0.72", "Model ROC-AUC (5-fold CV)", INK),
    ("59%", "Churners detected (recall)", INK),
    (f"{len(active_high)}", "Active customers at high risk", AMBER),
    (f"{active_high['monthly_charges'].sum():,.0f}", "Monthly revenue at risk", RED),
]
cw = (W - 0.6 - 5 * 0.2) / 6
for i, (v, l, c) in enumerate(kpis):
    x = 0.3 + i * (cw + 0.2)
    box(x, 6.65, cw, 1.15)
    fig.patches.append(Rectangle((x / W, 6.65 / H), 0.06 / W, 1.15 / H, transform=fig.transFigure, fc=c, zorder=1))
    fig.text((x + 0.25) / W, 7.25 / H, v, fontsize=24, color=c, weight="bold", va="center")
    fig.text((x + 0.25) / W, 6.85 / H, l, fontsize=9, color="#605E5C", va="center")

# ---- row 1 ----
ax = visual(0.3, 3.55, 4.9, 2.9, "Churn rate by contract")
c = df.groupby("contract")["churned"].mean().mul(100).reindex(["Month-to-month", "One year", "Two year"])
b = ax.bar(c.index, c.values, color=[RED, GREY, GREY], width=0.55)
ax.bar_label(b, fmt="%.1f%%", fontsize=9.5, color=INK, padding=2)
ax.set_yticks([]); ax.set_ylim(0, 50)

ax = visual(5.4, 3.55, 4.9, 2.9, "Churn rate by tenure (months)")
t = df.groupby(pd.cut(df["tenure"], [0, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]),
               observed=True)["churned"].mean().mul(100)
ax.plot(t.index.astype(str), t.values, color=NAVY, lw=2.5, marker="o", ms=7)
ax.fill_between(range(len(t)), t.values, color=NAVY, alpha=0.08)
for i, v in enumerate(t.values):
    ax.text(i, v + 2.5, f"{v:.0f}%", ha="center", fontsize=9.5, color=INK)
ax.set_yticks([]); ax.set_ylim(0, 58)

ax = visual(10.5, 3.55, 5.2, 2.9, "Actual churn by predicted risk band", left=0.9)
r = scored.assign(c=scored["churn"].eq("Yes")).groupby("risk_band")["c"].agg(["mean", "size"]).reindex(["High", "Medium", "Low"])
b = ax.barh(r.index[::-1], r["mean"][::-1] * 100, color=[GREEN, AMBER, RED], height=0.55)
for bar, (band, row) in zip(b, r[::-1].iterrows()):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{row['mean']:.0%}  ·  {int(row['size'])} customers", va="center", fontsize=9.5, color=INK)
ax.set_xticks([]); ax.set_xlim(0, 85)
ax.spines["bottom"].set_visible(False); ax.spines["left"].set_visible(True); ax.spines["left"].set_color("#C8C6C4")

# ---- row 2 ----
ax = visual(0.3, 0.3, 4.9, 3.05, "Churn rate by payment method", left=1.45)
p = df.groupby("payment_method")["churned"].mean().mul(100).sort_values()
b = ax.barh(p.index, p.values, color=[GREY] * (len(p) - 1) + [RED], height=0.55)
ax.bar_label(b, fmt="%.0f%%", fontsize=9.5, color=INK, padding=3)
ax.set_xticks([]); ax.set_xlim(0, 55)
ax.spines["bottom"].set_visible(False)

ax = visual(5.4, 0.3, 4.9, 3.05, "What drives churn (model coefficients)", left=1.55)
drivers = pd.Series({"Two-year contract": -0.71, "One-year contract": -0.46, "No internet service": -0.42,
                     "Online security": -0.32, "Electronic check": 0.25, "Fiber optic": 0.24,
                     "Monthly charges": 0.20}).sort_values()
ax.barh(drivers.index, drivers.values, color=[GREEN if v < 0 else RED for v in drivers.values], height=0.55)
ax.axvline(0, color="#8A8886", lw=0.8)
ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
ax.text(-0.72, -1.1, "◀ keeps customers", fontsize=8, color=GREEN)
ax.text(0.02, -1.1, "pushes to leave ▶", fontsize=8, color=RED)
ax.set_ylim(-1.4, len(drivers) - 0.4)

# call list table
box(10.5, 0.3, 5.2, 3.05)
fig.text(10.68 / W, 3.02 / H, "Call list · active customers at high risk", fontsize=11, color=INK, weight="semibold")
cols = [("Customer", 10.7), ("Contract", 11.75), ("Tenure", 13.2), ("Monthly", 13.95), ("Risk", 14.85)]
y0 = 2.62
for name, x in cols:
    fig.text(x / W, y0 / H, name, fontsize=8.5, color="#605E5C", weight="semibold")
fig.patches.append(Rectangle((10.68 / W, (y0 - 0.12) / H), 4.85 / W, 0.012 / H, transform=fig.transFigure, fc="#C8C6C4"))
for i, (_, row) in enumerate(active_high.head(8).iterrows()):
    y = y0 - 0.3 - i * 0.27
    vals = [row["customer_id"], row["contract"], f"{row['tenure']} m", f"{row['monthly_charges']:.0f}"]
    for (name, x), v in zip(cols[:4], vals):
        fig.text(x / W, y / H, v, fontsize=8.8, color=INK, va="center")
    pr = row["churn_probability"]
    fig.patches.append(Rectangle((14.85 / W, (y - 0.08) / H), 0.55 * pr / W, 0.16 / H, transform=fig.transFigure,
                                 fc=RED, alpha=0.25 + 0.75 * (pr - 0.6) / 0.4))
    fig.text(15.45 / W, y / H, f"{pr:.0%}", fontsize=8.8, color=INK, va="center")

fig.text((W - 0.3) / W, 0.08 / H, "Dashboard design · built from the model output · synthetic data",
         fontsize=7.5, color="#A19F9D", ha="right")
out = ROOT / "powerbi" / "dashboard_mockup.png"
fig.savefig(out, dpi=110, facecolor=CANVAS)
print(out)
