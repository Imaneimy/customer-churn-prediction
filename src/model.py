"""
Train a logistic regression model and evaluate it.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_model(X_train, y_train) -> tuple:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y_train)
    return clf, scaler


def evaluate_model(clf, scaler, X_test, y_test) -> dict:
    X_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_scaled)
    y_proba = clf.predict_proba(X_scaled)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def feature_importance(clf, feature_names: list) -> pd.DataFrame:
    coefficients = clf.coef_[0]
    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
        .assign(abs_coef=lambda x: x["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )
