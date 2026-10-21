# PM-09-PS06: Run AI Implementation
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # silence TF C++ logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # no oneDNN line
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"        # no absl line

import sys, functools
print = functools.partial(print, flush=True)  # unbuffered prints

from pathlib import Path
BASE   = Path(__file__).resolve().parent           # .../sections/PS06
ROOT   = BASE.parent.parent                        # .../PM09_DeepLearning
MODELS = ROOT / "models"
DATA   = ROOT / "hr_data.csv"
EVID   = ROOT / "evidence"

import numpy as np
import pandas as pd
import joblib
import time
import tensorflow as tf
keras = tf.keras

print("=" * 60)
print("PM-09-PS06: RUN AI IMPLEMENTATION")
print("=" * 60)

# --- PRE-FLIGHT: verify required artefacts exist -----------------
for p in [MODELS / "hr_model.keras",
          MODELS / "scaler.pkl",
          MODELS / "label_encoder.pkl",
          MODELS / "feature_cols.pkl",
          DATA]:
    assert p.exists(), f"Missing required file: {p}"
EVID.mkdir(parents=True, exist_ok=True)            # create evidence/ if absent

# 1. LOAD THE AI SOLUTION
model         = keras.models.load_model(MODELS / "hr_model.keras")
scaler        = joblib.load(MODELS / "scaler.pkl")
label_encoder = joblib.load(MODELS / "label_encoder.pkl")
feature_cols  = joblib.load(MODELS / "feature_cols.pkl")

print(f"Model loaded. Input features: {len(feature_cols)}")
print(f"Classes: {list(label_encoder.classes_)}")

# 2. BATCH PREDICTION FUNCTION
def batch_predict(df_input):
    df = df_input.copy()
    df = pd.get_dummies(df, columns=['Department'],
                        prefix='Dept', dtype=float)   # dtype=float avoids bool→object issues
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_cols]

    X = scaler.transform(df.values.astype(float))
    probs = model.predict(X, verbose=0)
    preds = np.argmax(probs, axis=1)

    labels      = label_encoder.classes_[preds]
    confidences = probs[np.arange(len(preds)), preds]
    return labels, confidences, probs

# 3. RUN ON THE FULL DATASET
print("\n--- RUNNING AI SOLUTION ON FULL DATASET ---")

df_full = pd.read_csv(DATA)
DROP_COLS = [c for c in ('EmployeeID', 'Salary', 'SalaryBand') if c in df_full.columns]
df_input  = df_full.drop(columns=DROP_COLS)

start = time.time()
labels, confidences, probs = batch_predict(df_input)
elapsed = time.time() - start
elapsed_safe = max(elapsed, 1e-6)              # avoid div-by-zero on very fast machines

print(f"Predicted {len(labels)} employees in {elapsed:.3f} seconds")
print(f"Throughput: {len(labels)/elapsed_safe:.0f} predictions/second")

# 4. BUILD THE OUTPUT REPORT
meta_cols = [c for c in ('EmployeeID', 'Department', 'YearsOfService',
                         'IsManager', 'PerformanceScore', 'Age')
             if c in df_full.columns]
results = df_full[meta_cols].copy()
results['PredictedBand'] = labels
if 'SalaryBand' in df_full.columns:
    results['ActualBand'] = df_full['SalaryBand'].values
    results['Correct']    = (results['PredictedBand'] == results['ActualBand'])
results['Confidence'] = confidences.round(3)

print("\n--- SAMPLE PREDICTIONS ---")
print(results.head(10).to_string(index=False))

# 5. MONITORING DASHBOARD
print("\n--- MONITORING DASHBOARD ---")

if 'Correct' in results.columns:
    overall_acc = results['Correct'].mean()
    print(f"Overall accuracy on full dataset: {overall_acc:.4f}")

    print("\nAccuracy by department:")
    print(results.groupby('Department')['Correct'].mean().round(3).to_string())

    print("\nAccuracy by actual band:")
    print(results.groupby('ActualBand')['Correct'].mean().round(3).to_string())
else:
    print("WARNING: no ground-truth 'SalaryBand' column — skipping accuracy metrics.")

print("\nConfidence distribution:")
print(f"  Mean confidence:   {results['Confidence'].mean():.3f}")
print(f"  Min confidence:    {results['Confidence'].min():.3f}")
print(f"  Low-confidence (< 0.6) predictions: "
      f"{(results['Confidence'] < 0.6).sum()}")

# 6. FLAG LOW-CONFIDENCE CASES
print("\n--- LOW-CONFIDENCE CASES (need human review) ---")
low_conf = results[results['Confidence'] < 0.6].sort_values('Confidence')
if len(low_conf) > 0:
    print(low_conf.head(10).to_string(index=False))
else:
    print("No low-confidence cases found.")

# 7. SAVE THE REPORT
out_csv = EVID / "ps06_predictions.csv"
results.to_csv(out_csv, index=False)
print(f"\nFull predictions saved to {out_csv}")

print("\n" + "=" * 60)
print("AI implementation ran successfully.")
print("Ready to productionise the solution (PS07).")
print("=" * 60)