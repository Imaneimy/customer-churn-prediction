"""
Entry point: preprocess, train, evaluate, plot.
"""

from pathlib import Path
from preprocessing import load_customers, encode_features, get_feature_matrix, churn_rate_by_segment
from model import split_data, train_model, evaluate_model, feature_importance
from visualizations import (
    plot_churn_distribution,
    plot_churn_by_contract,
    plot_tenure_vs_churn,
    plot_feature_importance,
    plot_confusion_matrix,
)

DATA = Path(__file__).parent.parent / "data" / "customers.csv"
REPORTS = Path(__file__).parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)


def main():
    df = load_customers(DATA)

    print(f"Dataset: {len(df)} customers, {df['churn_flag'].mean():.1%} churn rate\n")

    print("--- Churn rate by contract type ---")
    by_contract = churn_rate_by_segment(df, "contract")
    print(by_contract.to_string(index=False))

    print("\n--- Churn rate by internet service ---")
    by_internet = churn_rate_by_segment(df, "internet_service")
    print(by_internet.to_string(index=False))

    encoded = encode_features(df)
    X, y = get_feature_matrix(encoded)

    X_train, X_test, y_train, y_test = split_data(X, y)
    clf, scaler = train_model(X_train, y_train)
    results = evaluate_model(clf, scaler, X_test, y_test)

    print(f"\nModel: Logistic Regression")
    print(f"Accuracy : {results['accuracy']}")
    print(f"ROC-AUC  : {results['roc_auc']}")
    print("\nClassification Report:")
    print(results["classification_report"])

    imp = feature_importance(clf, list(X.columns))
    print("Top 10 features:")
    print(imp.head(10).to_string(index=False))

    plot_churn_distribution(df, str(REPORTS / "churn_distribution.png"))
    plot_churn_by_contract(by_contract, str(REPORTS / "churn_by_contract.png"))
    plot_tenure_vs_churn(df, str(REPORTS / "tenure_vs_churn.png"))
    plot_feature_importance(imp, str(REPORTS / "feature_importance.png"))
    plot_confusion_matrix(results["confusion_matrix"], str(REPORTS / "confusion_matrix.png"))

    print(f"\nCharts saved to {REPORTS}/")


if __name__ == "__main__":
    main()
