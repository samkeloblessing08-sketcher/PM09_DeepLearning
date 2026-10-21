# PM-09-PS01: Identify a Problem
from pathlib import Path
import pandas as pd

print("=" * 60)
print("PM-09-PS01: IDENTIFY A PROBLEM")
print("=" * 60)

# Anchor to the script's location so it works regardless of CWD
# parents[0] = PS01, parents[1] = sections, parents[2] = PM09_DeepLearning (project root)
DATA_PATH = Path(__file__).resolve().parents[2] / "hr_data.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Run Setup_Dataset.py first from the project root."
    )

df = pd.read_csv(DATA_PATH)

print("\n--- PROBLEM STATEMENT ---")
print("HR assigns salary bands manually, leading to inconsistency")
print("and inequity. We need an AI system that predicts an")
print("employee's salary band from objective features.")

print("\n--- SUCCESS CRITERIA ---")
print("Primary:  Test set accuracy >= 80%")
print("Fairness: Per-class recall >= 70% for all 3 bands")
print("Speed:    Full pipeline runs in < 5 minutes")

print("\n--- PERFORMANCE STANDARD ---")
print("Model must generalise to unseen employees across all")
print("5 departments, without using EmployeeID or Salary directly.")

print("\n --- DATA OVERVIEW ---")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

print("\n--- TARGET DISTRIBUTION (SalaryBand) ---")
band_counts = df['SalaryBand'].value_counts()
print(band_counts)

print(f"\nClass balance: {band_counts.min() / band_counts.max():.2f} "
      "(1.0 = perfectly balanced)")

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- FEATURE / TARGET RELATIONSHIP (hypothesis test) ---")
print("Average salary by band (sanity check that bands are ordered):")
print(df.groupby('SalaryBand')['Salary'].mean().round(2))

print("\nHypothesis: Department, YearsOfService, IsManager,")
print("PerformanceScore and Age carry enough signal to predict")
print("SalaryBand with >= 80% accuracy.")

print("\n" + "=" * 60)
print("Problem identified. Ready for data preparation (PS02).")
print("=" * 60)