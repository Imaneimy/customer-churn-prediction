# customer-churn-prediction

Churn prediction was the core topic of my final-year project at ENSAM Meknes. This is a standalone version of that work — no notebook sprawl, just a clean Python pipeline that goes from raw customer data to a trained logistic regression model with evaluation charts.

The dataset has 500 telecom customers with contract type, internet service, tenure, monthly charges, and a handful of other attributes. About a third of them churned. The model identifies which features drive churn most strongly — contract type and tenure are the biggest predictors, which matches what I found in the academic version of this project.

## Structure

```
src/
  preprocessing.py    # load, encode, build feature matrix, segment churn rates
  model.py            # train/evaluate logistic regression, feature importance
  visualizations.py   # five charts saved to reports/
  run_model.py        # entry point

tests/
  test_preprocessing.py   # 10 unit tests TC-CHR-001→010
  test_model.py           # 8 unit tests TC-MOD-001→008

data/
  customers.csv       # 500 customers, synthetic telecom dataset
  generate_data.py    # script that generated customers.csv

reports/              # generated charts (git-ignored)
```

## Running it

```bash
pip install -r requirements.txt
cd src
python run_model.py
```

Prints churn rate by contract and internet service type, then model accuracy and ROC-AUC, and saves five charts to `reports/`.

```bash
pytest tests/ -v
```

## Model results

Logistic regression with StandardScaler. Typical output on this dataset:

- Accuracy: ~0.78
- ROC-AUC: ~0.82

The strongest predictors: month-to-month contract, short tenure, fiber optic + high monthly charges, electronic check payment.

## What I would do differently with more data

With a larger dataset I'd try a gradient boosted model (XGBoost or LightGBM) and use SHAP to explain predictions at the individual customer level — which is more useful for a retention team than global feature importances.

## Stack

Python, Pandas, Scikit-learn, Matplotlib, Pytest
