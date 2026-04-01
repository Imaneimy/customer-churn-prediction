"""
Generate a synthetic telecom churn dataset and save it as customers.csv.
Run once: python generate_data.py
"""

import random
import csv
from pathlib import Path

random.seed(42)

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fiber optic", "No"]
PAYMENT = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]

rows = []
for i in range(1, 501):
    contract = random.choices(CONTRACTS, weights=[0.55, 0.25, 0.20])[0]
    tenure = random.randint(1, 72)
    internet = random.choice(INTERNET)
    monthly = round(random.uniform(20, 110), 2)
    total = round(monthly * tenure * random.uniform(0.9, 1.0), 2)
    senior = random.choices([0, 1], weights=[0.84, 0.16])[0]
    tech_support = random.choice(["Yes", "No"]) if internet != "No" else "No internet service"
    online_security = random.choice(["Yes", "No"]) if internet != "No" else "No internet service"
    payment = random.choice(PAYMENT)

    # churn logic: higher for month-to-month, short tenure, fiber + high bill
    churn_prob = 0.10
    if contract == "Month-to-month":
        churn_prob += 0.25
    if tenure < 12:
        churn_prob += 0.15
    if internet == "Fiber optic" and monthly > 80:
        churn_prob += 0.10
    if payment == "Electronic check":
        churn_prob += 0.08
    if online_security == "No":
        churn_prob += 0.05
    churn = "Yes" if random.random() < min(churn_prob, 0.75) else "No"

    rows.append({
        "customer_id": f"C{i:04d}",
        "gender": random.choice(["Male", "Female"]),
        "senior_citizen": senior,
        "tenure": tenure,
        "contract": contract,
        "internet_service": internet,
        "online_security": online_security,
        "tech_support": tech_support,
        "payment_method": payment,
        "monthly_charges": monthly,
        "total_charges": total,
        "churn": churn,
    })

out = Path(__file__).parent / "customers.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {out}")
