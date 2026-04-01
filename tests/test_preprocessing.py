import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocessing import load_customers, encode_features, get_feature_matrix, churn_rate_by_segment

DATA = Path(__file__).parent.parent / "data" / "customers.csv"


@pytest.fixture(scope="module")
def df():
    return load_customers(DATA)


# TC-CHR-001
def test_load_returns_dataframe(df):
    assert isinstance(df, pd.DataFrame)


# TC-CHR-002
def test_churn_flag_is_binary(df):
    assert set(df["churn_flag"].unique()).issubset({0, 1})


# TC-CHR-003
def test_no_null_customer_id(df):
    assert df["customer_id"].isna().sum() == 0


# TC-CHR-004
def test_tenure_positive(df):
    assert (df["tenure"] > 0).all()


# TC-CHR-005
def test_monthly_charges_positive(df):
    assert (df["monthly_charges"] > 0).all()


# TC-CHR-006
def test_encode_drops_no_rows(df):
    encoded = encode_features(df)
    assert len(encoded) == len(df)


# TC-CHR-007
def test_encode_adds_dummies(df):
    encoded = encode_features(df)
    assert any("contract_" in c for c in encoded.columns)


# TC-CHR-008
def test_feature_matrix_shape(df):
    encoded = encode_features(df)
    X, y = get_feature_matrix(encoded)
    assert len(X) == len(y)
    assert X.shape[1] > 0


# TC-CHR-009
def test_churn_rate_by_contract_has_all_types(df):
    result = churn_rate_by_segment(df, "contract")
    assert set(result["contract"]) == set(df["contract"].unique())


# TC-CHR-010
def test_churn_rate_sorted_descending(df):
    result = churn_rate_by_segment(df, "contract")
    rates = result["churn_rate_pct"].tolist()
    assert rates == sorted(rates, reverse=True)
