"""
Train a logistic regression model and evaluate it.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    recall_score,
)


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_model(X_train, y_train) -> tuple:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    # ~30% of customers churn: balance the classes so the model doesn't just predict "stays"
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    clf.fit(X_scaled, y_train)
    return clf, scaler


def evaluate_model(clf, scaler, X_test, y_test) -> dict:
    X_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_scaled)
    y_proba = clf.predict_proba(X_scaled)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "recall_churn": round(recall_score(y_test, y_pred), 4),
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


def cross_validated_auc(X, y, folds: int = 5) -> tuple:
    """Mean and std ROC-AUC over stratified folds: steadier than one 100-row test set."""
    pipe = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return round(scores.mean(), 4), round(scores.std(), 4)


def score_customers(clf, scaler, df, X) -> pd.DataFrame:
    """Churn probability per customer, with a risk band a retention team can act on."""
    proba = clf.predict_proba(scaler.transform(X))[:, 1]
    out = df[["customer_id", "contract", "tenure", "monthly_charges", "churn"]].copy()
    out["churn_probability"] = proba.round(3)
    out["risk_band"] = pd.cut(proba, bins=[0, 0.4, 0.6, 1], labels=["Low", "Medium", "High"],
                              include_lowest=True)
    return out.sort_values("churn_probability", ascending=False).reset_index(drop=True)
