# PM-09-PS05: Select an ML System
import numpy as np
import pandas as pd
import joblib
import os
import functools
print = functools.partial(print, flush=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # hide INFO/WARN/ERROR from TF C++
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"       # remove oneDNN warning
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONUNBUFFERED"] = "1"
import tensorflow as tf
keras = tf.keras
from sklearn.metrics import recall_score

print("=" * 60) 
print("PM-09-PS05: SELECT AN ML SYSTEM") 
print("=" * 60) 
print("\n--- ML SYSTEM IDENTIFICATION ---") 
print(""" 
System type: Supervised multi-class classifier 
Framework:   TensorFlow / Keras 
Input:       13 numeric features (department one-hot, years, 
             manager status, performance, age) 
Output:      1 of 3 salary bands (Junior / Mid / Senior) 
 
Why this is an ML system and not a rule-based programme: 
  - No human wrote "if years > 10 and manager then Senior" 
  - The model LEARNED the decision boundaries from data 
  - It can generalise to employees it has never seen 
""") 

 # LOAD THE TRAINED SYSTEM 
model = keras.models.load_model("../../models/hr_model.keras") 
scaler = joblib.load("../../models/scaler.pkl")
label_encoder = joblib.load("../../models/label_encoder.pkl") 
feature_cols = joblib.load("../../models/feature_cols.pkl") 

print(f"Model loaded. Input features: {len(feature_cols)}") 
print(f"Feature names: {feature_cols}")

# REUSABLE PREDICTION FUNCTION 
def predict_salary_band(employee_dict): 
    row = pd.DataFrame([employee_dict]) 
    row = pd.get_dummies(row, columns=['Department'], prefix='Dept') 
 
    for col in feature_cols: 
        if col not in row.columns: 
            row[col] = 0 
 
    row = row[feature_cols] 
    row_scaled = scaler.transform(row.values) 
    probs = model.predict(row_scaled, verbose=0)[0] 
 
    class_idx = int(np.argmax(probs)) 
    label = label_encoder.classes_[class_idx] 
    confidence = float(probs[class_idx]) 
 
    return label, confidence, dict(zip(label_encoder.classes_, probs.round(4)))

# TEST WITH 5 EXAMPLE EMPLOYEES 
print("\n--- SYSTEM TEST: 5 EXAMPLE EMPLOYEES ---") 
 
test_employees = [ 
    {"Department": "Engineering", "YearsOfService": 12, 
     "IsManager": 1, "PerformanceScore": 4.5, "Age": 45}, 
    {"Department": "Support", "YearsOfService": 2, 
     "IsManager": 0, "PerformanceScore": 3.0, "Age": 25}, 
    {"Department": "Management", "YearsOfService": 20, 
     "IsManager": 1, "PerformanceScore": 4.8, "Age": 55}, 
    {"Department": "Sales", "YearsOfService": 6, 
     "IsManager": 0, "PerformanceScore": 3.5, "Age": 32}, 
    {"Department": "Finance", "YearsOfService": 15, 
     "IsManager": 0, "PerformanceScore": 4.0, "Age": 48}, 
] 
 
for i, emp in enumerate(test_employees, 1): 
    label, conf, probs = predict_salary_band(emp) 
    print(f"\nEmployee {i}: {emp['Department']}, " 
          f"{emp['YearsOfService']}yrs, " 
          f"Manager={emp['IsManager']}") 
    print(f"  Predicted: {label} (confidence: {conf:.2%})") 
    print(f"  Probabilities: {probs}") 
 
# DEBUG: EDGE CASES 
print("\n--- DEBUGGING: EDGE CASES ---") 
 
edge_cases = [ 
    {"Department": "Unknown", "YearsOfService": 0, 
     "IsManager": 0, "PerformanceScore": 1.0, "Age": 22}, 
    {"Department": "Engineering", "YearsOfService": 50, 
     "IsManager": 1, "PerformanceScore": 5.0, "Age": 70}, 
] 
 
for i, emp in enumerate(edge_cases, 1): 
    try: 
        label, conf, _ = predict_salary_band(emp) 
        print(f"Edge case {i}: predicted {label} " 
              f"(confidence: {conf:.2%})") 
    except Exception as e: 
        print(f"Edge case {i}: ERROR - {e}") 

 # VALIDATE AGAINST THE PERFORMANCE METRIC 
print("\n--- VALIDATION AGAINST PERFORMANCE METRIC ---") 
 
X_test = np.load("../../models/X_test.npy") 
y_test = np.load("../../models/y_test.npy") 
 
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1) 
test_acc = (y_pred == y_test).mean() 
 
print(f"Test accuracy: {test_acc:.4f}") 
print(f"Required:      >= 0.80") 
print(f"Status:        {'PASS' if test_acc >= 0.80 else 'FAIL'}") 
 
recalls = recall_score(y_test, y_pred, average=None) 
print("\nPer-class recall:") 
for i, r in enumerate(recalls): 
    status = "OK" if r >= 0.70 else "BELOW TARGET" 
    print(f"  {label_encoder.classes_[i]}: {r:.4f} ({status})") 
 
print("\n" + "=" * 60) 
print("ML system selected, tested, debugged, and validated.") 
print("Ready to run AI implementation (PS06).") 
print("=" * 60) 

 
