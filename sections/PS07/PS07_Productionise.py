# PM-09-PS07: Turn Prototype into an Economically Viable Product
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # silence TF C++ logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # no oneDNN line
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"        # no absl line

import sys, functools
print = functools.partial(print, flush=True)  # unbuffered prints

from pathlib import Path
BASE   = Path(__file__).resolve().parent      # .../sections/PS07
ROOT   = BASE.parent.parent                   # .../PM09_DeepLearning
MODELS = ROOT / "models"
DATA   = ROOT / "hr_data.csv"
EVID   = ROOT / "evidence"

import numpy as np
import pandas as pd
import joblib
import json
import time
from datetime import datetime
import tensorflow as tf
from scipy import stats

print("=" * 60)
print("PM-09-PS07: TURN PROTOTYPE INTO A VIABLE PRODUCT")
print("=" * 60)

# --- PRE-FLIGHT -------------------------------------------------
required = [
    MODELS / "hr_model.keras",
    MODELS / "scaler.pkl",
    MODELS / "label_encoder.pkl",
    MODELS / "feature_cols.pkl",
    DATA,
]
for p in required:
    assert p.exists(), f"Missing required file: {p}"
EVID.mkdir(parents=True, exist_ok=True)

# 1. LOAD THE AI SOLUTION
model         = tf.keras.models.load_model(MODELS / "hr_model.keras")
scaler        = joblib.load(MODELS / "scaler.pkl")
label_encoder = joblib.load(MODELS / "label_encoder.pkl")
feature_cols  = joblib.load(MODELS / "feature_cols.pkl")

print(f"Model loaded. Input features: {len(feature_cols)}")
print(f"Classes: {list(label_encoder.classes_)}")


# 2. PRODUCTION-READY SYSTEM
class SalaryBandPredictor:
    """Wraps the trained model with encoding, scaling, logging
    and graceful error handling."""

    def __init__(self, model, scaler, label_encoder, feature_cols):
        self.model         = model
        self.scaler        = scaler
        self.label_encoder = label_encoder
        self.feature_cols  = feature_cols
        self.prediction_log = []

    def _prepare(self, employee_dict):
        row = pd.DataFrame([employee_dict])
        row = pd.get_dummies(row, columns=['Department'],
                             prefix='Dept', dtype=float)
        for col in self.feature_cols:
            if col not in row.columns:
                row[col] = 0.0
        row = row[self.feature_cols]
        return self.scaler.transform(row.values.astype(float))

    def predict(self, employee_dict):
        try:
            X = self._prepare(employee_dict)
            probs = self.model.predict(X, verbose=0)[0]
            idx   = int(np.argmax(probs))
            label = self.label_encoder.classes_[idx]
            confidence = float(probs[idx])

            self.prediction_log.append({
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'input':     employee_dict,
                'prediction': label,
                'confidence': round(confidence, 4),
            })

            return {
                'prediction':          label,
                'confidence':          round(confidence, 4),
                'needs_human_review':  confidence < 0.6,
                'status':              'success',
            }
        except Exception as e:
            return {
                'prediction':          None,
                'confidence':          None,
                'needs_human_review':  True,
                'status':              'error',
                'error':               str(e),
            }

    def get_logs(self):
        if not self.prediction_log:
            # guarantee the expected columns exist even when no rows were logged
            return pd.DataFrame(columns=['timestamp', 'input',
                                         'prediction', 'confidence'])
        return pd.DataFrame(self.prediction_log)


# 3. RUN THE PRODUCTION SYSTEM
print("\n--- PRODUCTION SYSTEM TEST ---")
predictor = SalaryBandPredictor(model, scaler, label_encoder, feature_cols)

employees = [
    {"Department": "Engineering", "YearsOfService": 12,
     "IsManager": 1, "PerformanceScore": 4.5, "Age": 45},
    {"Department": "Support", "YearsOfService": 2,
     "IsManager": 0, "PerformanceScore": 3.0, "Age": 25},
    {"Department": "Management", "YearsOfService": 20,
     "IsManager": 1, "PerformanceScore": 4.8, "Age": 55},
]

for emp in employees:
    result = predictor.predict(emp)
    print(f"\nInput: {emp['Department']}, {emp['YearsOfService']}yrs")
    print(json.dumps(result, indent=2))

# 4. MONITOR AND MAINTAIN
print("\n--- MONITORING LOG ---")
logs = predictor.get_logs()
print(logs[['timestamp', 'prediction', 'confidence']].to_string(index=False))

# 5. ITERATE FOR NEXT VERSION
print("\n--- ITERATION PLAN (v2.0) ---")
print("""
Based on monitoring, the next version should:
  1. Add more training data (currently 350 rows; target 5000+)
  2. Add features: job title, certifications, location
  3. Try class weights to improve minority-class recall
  4. Add SHAP values for explainability (HR requirement)
  5. Set up automated retraining every quarter
  6. Add drift detection (monitor input distribution)
""")

# 6. USER ACCEPTANCE TESTING PLAN
print("\n--- USER ACCEPTANCE TESTING PLAN ---")
print("""
Phase 1 (Week 1-2): HR analysts review 50 predictions
  - Do the labels match their judgement?
  - Are confidence scores intuitive?

Phase 2 (Week 3-4): Shadow mode
  - System runs alongside manual process
  - Compare outputs; log disagreements

Phase 3 (Week 5-6): Assisted mode
  - HR sees AI suggestion + confidence
  - HR makes final decision
  - Collect feedback for retraining
""")

# 7. A/B TEST SIMULATION
print("\n--- A/B TEST SIMULATION ---")
print("Comparing AI-assisted vs manual assignment over 100 employees...")

np.random.seed(42)

df_test = pd.read_csv(DATA).sample(100, random_state=42)

DROP_COLS = [c for c in ('EmployeeID', 'Salary', 'SalaryBand')
             if c in df_test.columns]
df_input = df_test.drop(columns=DROP_COLS)

# Arm A: manual assignment (simulated as 75% accurate)
manual_correct = np.random.random(len(df_test)) < 0.75

# Arm B: AI-assisted (real model)
X = df_input.copy()
X = pd.get_dummies(X, columns=['Department'], prefix='Dept', dtype=float)
for col in feature_cols:
    if col not in X.columns:
        X[col] = 0.0
X = X[feature_cols]
X_scaled = scaler.transform(X.values.astype(float))
preds = np.argmax(model.predict(X_scaled, verbose=0), axis=1)
predicted_labels = label_encoder.classes_[preds]

if 'SalaryBand' in df_test.columns:
    ai_correct = (predicted_labels == df_test['SalaryBand'].values)

    print(f"\nArm A (Manual):      accuracy = {manual_correct.mean():.3f}")
    print(f"Arm B (AI-assisted): accuracy = {ai_correct.mean():.3f}")
    print(f"Improvement:         "
          f"{(ai_correct.mean() - manual_correct.mean())*100:+.1f} percentage points")

    table = [[int(ai_correct.sum()),     int((~ai_correct).sum())],
             [int(manual_correct.sum()), int((~manual_correct).sum())]]
    # chi2_contingency requires all expected frequencies > 0; guard against a
    # degenerate 2x2 where one arm has zero successes or zero failures.
    if any(v == 0 for row in table for v in row):
        print("Chi-square skipped: contingency table has a zero cell.")
    else:
        chi2, p_value, _, _ = stats.chi2_contingency(table)
        print(f"Chi-square p-value: {p_value:.4f}")
        print(f"Statistically significant: "
              f"{'YES' if p_value < 0.05 else 'NO'}")
else:
    print("WARNING: 'SalaryBand' column missing — cannot compute AI accuracy.")

# 8. AFFECTED SYSTEMS
print("\n--- SYSTEMS AND PROCESSES AFFECTED ---")
print("""
1. HR annual salary review process
2. Payroll system (budget forecasting)
3. Performance management (promotion eligibility)
4. Compliance and audit (logged, explainable predictions)
5. Employee communication (explaining bands)
6. Recruitment (offer ranges for new hires)

Each of these needs a change-management plan before go-live.
""")

# 9. BUSINESS CASE
print("--- ECONOMIC VIABILITY ---")
print("""
COSTS:
  - Development: 1 developer x 2 weeks = R40,000
  - Infrastructure: cloud CPU inference = R500/month
  - Annual retraining: 2 days = R5,000/year
  - Total first year: ~R52,000

BENEFITS (annual):
  - HR time saved: 200 hours x R300/hr = R60,000
  - Reduced misclassification / retention: R150,000
  - Faster annual review cycle: R30,000
  - Total annual benefit: ~R240,000

ROI: (240,000 - 52,000) / 52,000 = 362% in year 1
Payback period: ~3 months
""")

# 10. SAVE PRODUCTION ARTEFACTS
out_csv = EVID / "ps07_prediction_logs.csv"
logs.to_csv(out_csv, index=False)
print(f"\nPrediction logs saved to {out_csv}")

print("\n" + "=" * 60)
print("Prototype turned into an economically viable product.")
print("PM-09 complete.")
print("=" * 60)