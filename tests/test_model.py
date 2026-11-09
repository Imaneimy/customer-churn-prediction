import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocessing import load_customers, encode_features, get_feature_matrix
from model import split_data, train_model, evaluate_model, feature_importance

DATA = Path(__file__).parent.parent / "data" / "customers.csv"


@pytest.fixture(scope="module")
def trained():
    df = load_customers(DATA)
    encoded = encode_features(df)
    X, y = get_feature_matrix(encoded)
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf, scaler = train_model(X_train, y_train)
    results = evaluate_model(clf, scaler, X_test, y_test)
    imp = feature_importance(clf, list(X.columns))
    return {"clf": clf, "scaler": scaler, "X_test": X_test, "y_test": y_test,
            "results": results, "imp": imp, "X": X}


# TC-MOD-001
# Accuracy is misleading here (predicting "stays" for everyone scores ~70%),
# so the model is judged on how many churners it actually catches.
def test_model_catches_at_least_half_of_churners(trained):
    assert trained["results"]["recall_churn"] >= 0.50


# TC-MOD-002
def test_roc_auc_above_threshold(trained):
    assert trained["results"]["roc_auc"] >= 0.65


# TC-MOD-003
def test_classification_report_present(trained):
    assert "precision" in trained["results"]["classification_report"]


# TC-MOD-004
def test_confusion_matrix_shape(trained):
    cm = trained["results"]["confusion_matrix"]
    assert cm.shape == (2, 2)


# TC-MOD-005
def test_confusion_matrix_sums_to_test_size(trained):
    cm = trained["results"]["confusion_matrix"]
    assert cm.sum() == len(trained["y_test"])


# TC-MOD-006
def test_feature_importance_row_count(trained):
    assert len(trained["imp"]) == trained["X"].shape[1]


# TC-MOD-007
def test_feature_importance_has_coefficient_col(trained):
    assert "coefficient" in trained["imp"].columns


# TC-MOD-008
def test_split_stratified(trained):
    # churn rate should be roughly similar in train and test
    df = load_customers(DATA)
    encoded = encode_features(df)
    X, y = get_feature_matrix(encoded)
    _, _, y_train, y_test = split_data(X, y)
    assert abs(y_train.mean() - y_test.mean()) < 0.05


# TC-MOD-009 (regression): categorical dummies used to be silently dropped
def test_categorical_features_reach_the_model(trained):
    cols = trained["X"].columns
    for prefix in ["contract_", "internet_service_", "payment_method_"]:
        assert any(c.startswith(prefix) for c in cols), prefix


# TC-MOD-010
def test_cross_validated_auc_is_stable():
    from model import cross_validated_auc
    df = load_customers(DATA)
    X, y = get_feature_matrix(encode_features(df))
    mean, std = cross_validated_auc(X, y)
    assert mean >= 0.65 and std < 0.1


# TC-MOD-011
def test_scored_customers_are_ranked_with_valid_probabilities(trained):
    from model import score_customers
    df = load_customers(DATA)
    scored = score_customers(trained["clf"], trained["scaler"], df, trained["X"])
    assert len(scored) == len(df)
    assert scored["churn_probability"].between(0, 1).all()
    assert scored["churn_probability"].is_monotonic_decreasing
