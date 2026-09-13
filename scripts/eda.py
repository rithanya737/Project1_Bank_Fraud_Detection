"""
eda.py
======
Quick exploratory data analysis on data/transactions.csv.

Produces:
  - outputs/class_imbalance.png   : bar chart of fraud vs. non-fraud counts
  - outputs/amount_distribution.png : amount distribution, fraud vs. non-fraud
  - outputs/correlation_heatmap.png : correlation of numeric features with is_fraud

Also prints summary stats to stdout (class imbalance %, feature means by class).

Run:
    python scripts/eda.py
"""

import os
import sys
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")  # headless backend, no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/transactions.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df["hour"] = df["timestamp"].dt.hour

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
print(df.dtypes)
print()

# ------------------------------------------------------------------
# 1. Class imbalance
# ------------------------------------------------------------------
fraud_counts = df["is_fraud"].value_counts().sort_index()
fraud_rate = df["is_fraud"].mean() * 100
print("=" * 60)
print("CLASS IMBALANCE")
print("=" * 60)
print(f"Non-fraud: {fraud_counts[0]:,}   Fraud: {fraud_counts[1]:,}")
print(f"Fraud rate: {fraud_rate:.2f}%")
print()

plt.figure(figsize=(5, 4))
ax = sns.barplot(x=["Non-Fraud", "Fraud"], y=fraud_counts.values, palette=["#4C72B0", "#C44E52"])
ax.set_title(f"Class Imbalance (Fraud rate = {fraud_rate:.2f}%)")
ax.set_ylabel("Number of Transactions")
for i, v in enumerate(fraud_counts.values):
    ax.text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_imbalance.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 2. Amount distribution: fraud vs non-fraud
# ------------------------------------------------------------------
plt.figure(figsize=(7, 4.5))
sns.kdeplot(df.loc[df.is_fraud == 0, "amount"].clip(upper=2000), label="Non-Fraud", fill=True, alpha=0.4)
sns.kdeplot(df.loc[df.is_fraud == 1, "amount"].clip(upper=2000), label="Fraud", fill=True, alpha=0.4, color="#C44E52")
plt.title("Transaction Amount Distribution by Class (clipped at $2,000)")
plt.xlabel("Amount ($)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/amount_distribution.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 3. Correlation of numeric / encoded features with is_fraud
# ------------------------------------------------------------------
corr_df = df.copy()
for col in ["transaction_type", "merchant_category", "country"]:
    corr_df[col] = corr_df[col].astype("category").cat.codes

numeric_cols = ["amount", "device_risk_score", "ip_risk_score", "hour",
                 "transaction_type", "merchant_category", "country", "is_fraud"]
corr = corr_df[numeric_cols].corr()

plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

print("=" * 60)
print("CORRELATION WITH is_fraud")
print("=" * 60)
print(corr["is_fraud"].sort_values(ascending=False))
print()

# ------------------------------------------------------------------
# 4. Feature means by class (quick sanity check on signal strength)
# ------------------------------------------------------------------
print("=" * 60)
print("MEAN FEATURE VALUES BY CLASS")
print("=" * 60)
print(df.groupby("is_fraud")[["amount", "device_risk_score", "ip_risk_score"]].mean())
print()

print("=" * 60)
print("FRAUD RATE BY CATEGORICAL FEATURE")
print("=" * 60)
for col in ["transaction_type", "merchant_category", "country"]:
    print(f"\n-- {col} --")
    print(df.groupby(col)["is_fraud"].mean().sort_values(ascending=False) * 100)

print(f"\nSaved plots to {OUTPUT_DIR}/: class_imbalance.png, amount_distribution.png, correlation_heatmap.png")
