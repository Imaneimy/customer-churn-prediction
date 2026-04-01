"""
Charts for the churn analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_churn_distribution(df, out="reports/churn_distribution.png"):
    counts = df["churn"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(counts.index, counts.values, color=["#55A868", "#C44E52"])
    ax.set_title("Churn Distribution", fontsize=14, pad=12)
    ax.set_ylabel("Customers")
    for bar, val in zip(ax.patches, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(val), ha="center")
    _save(fig, out)


def plot_churn_by_contract(segment_df, out="reports/churn_by_contract.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(segment_df["contract"], segment_df["churn_rate_pct"], color="#4C72B0")
    ax.set_title("Churn Rate by Contract Type", fontsize=14, pad=12)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 100)
    for bar, val in zip(ax.patches, segment_df["churn_rate_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val}%", ha="center")
    _save(fig, out)


def plot_tenure_vs_churn(df, out="reports/tenure_vs_churn.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    churned = df[df["churn"] == "Yes"]["tenure"]
    retained = df[df["churn"] == "No"]["tenure"]
    ax.hist(retained, bins=20, alpha=0.6, label="No churn", color="#55A868")
    ax.hist(churned, bins=20, alpha=0.6, label="Churned", color="#C44E52")
    ax.set_title("Tenure Distribution by Churn", fontsize=14, pad=12)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Customers")
    ax.legend()
    _save(fig, out)


def plot_feature_importance(importance_df, out="reports/feature_importance.png"):
    top = importance_df.head(10)
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top["coefficient"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.set_title("Top 10 Feature Coefficients (Logistic Regression)", fontsize=13, pad=12)
    ax.set_xlabel("Coefficient")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    _save(fig, out)


def plot_confusion_matrix(cm, out="reports/confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=13, pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No churn", "Churned"])
    ax.set_yticklabels(["No churn", "Churned"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    _save(fig, out)
