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
def test_model_accuracy_above_threshold(trained):
    assert trained["results"]["accuracy"] >= 0.70


# TC-MOD-002
def test_roc_auc_above_threshold(trained):
    assert trained["results"]["roc_auc"] >= 0.50


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
