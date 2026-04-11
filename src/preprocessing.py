"""
Load and prepare the customer dataset for churn modeling.
"""

import pandas as pd
from pathlib import Path


def load_customers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["churn_flag"] = (df["churn"] == "Yes").astype(int)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    df = df.copy()

    for col in ["gender", "online_security", "tech_support"]:
        df[col] = df[col].map(lambda x: binary_map.get(x, 0))

    df = pd.get_dummies(df, columns=["contract", "internet_service", "payment_method"], drop_first=True)
    return df


def get_feature_matrix(df: pd.DataFrame):
    drop_cols = ["customer_id", "churn", "churn_flag"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].select_dtypes(include=["number"])
    y = df["churn_flag"]
    return X, y


def churn_rate_by_segment(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    return (
        df.groupby(segment)["churn_flag"]
        .agg(customers="count", churned="sum")
        .assign(churn_rate_pct=lambda x: (x["churned"] / x["customers"] * 100).round(1))
        .sort_values("churn_rate_pct", ascending=False)
        .reset_index()
    )

def class_balance(df) -> dict:
    counts = df['churn'].value_counts()
    return counts.to_dict()
