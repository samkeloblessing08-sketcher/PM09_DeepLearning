# PM-09-PS02: Prepare Data
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
import joblib 
import os

from pathlib import Path
import pandas as pd

# Anchor to project root regardless of CWD
DATA_PATH = Path(__file__).resolve().parents[2] / "hr_data.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Run Setup_Dataset.py first from the project root."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 60) 
print("PM-09-PS02: PREPARE DATA") 
print("=" * 60) 
 
# 1. SOURCE DATA 
df = pd.read_csv(Path(__file__).resolve().parents[2] / "hr_data.csv")
print(f"\nSourced {len(df)} rows from hr_data.csv") 
 
# 2. ANALYSE DATA 
print("\n--- DESCRIPTIVE STATISTICS ---") 
print(df.describe().round(2)) 
 
print("\n--- CORRELATION WITH SALARY ---") 
numeric_cols = ['YearsOfService', 'IsManager', 'PerformanceScore', 'Age', 'Salary'] 
print(df[numeric_cols].corr()['Salary'].round(3)) 
 
# 3. FIND PATTERNS 
print("\n--- PATTERN: SALARY BY DEPARTMENT ---") 
print(df.groupby('Department')['Salary'].agg(['mean', 'count']).round(2)) 
 
print("\n--- PATTERN: SALARY BAND BY MANAGER STATUS ---") 
print(pd.crosstab(df['IsManager'], df['SalaryBand'], normalize='index').round(3))
# 4. CONSISTENCY CHECKS 
print("\n--- CONSISTENCY CHECKS ---") 
missing = df.isnull().sum().sum() 
print(f"Total missing values: {missing}") 
 
duplicates = df.duplicated(subset=['EmployeeID']).sum() 
print(f"Duplicate EmployeeIDs: {duplicates}") 
 
print(f"PerformanceScore range: {df['PerformanceScore'].min()} " 
      f"to {df['PerformanceScore'].max()} (expected 1-5)") 
print(f"Age range: {df['Age'].min()} to {df['Age'].max()} " 
      "(expected 22-65)") 
 
# Drop the identifier and the raw salary (it would leak the answer) 
df = df.drop(columns=['EmployeeID', 'Salary']) 

 # 5. ENCODE CATEGORICAL FEATURES 
print("\n--- ENCODING CATEGORICAL FEATURES ---") 
df = pd.get_dummies(df, columns=['Department'], prefix='Dept') 
print(f"After one-hot encoding: {df.shape[1]} columns") 
 
label_encoder = LabelEncoder() 
df['SalaryBandEncoded'] = label_encoder.fit_transform(df['SalaryBand']) 
print(f"Target classes: {list(label_encoder.classes_)}") 
print(f"Encoded as: {list(range(len(label_encoder.classes_)))}") 
 
os.makedirs("../../models", exist_ok=True) 
joblib.dump(label_encoder, "../../models/label_encoder.pkl") 
 
# 6. SPLIT: 70% TRAIN / 15% VALIDATION / 15% TEST 
print("\n--- SPLITTING DATA: 70% TRAIN / 15% VALIDATION / 15% TEST ---") 
 
feature_cols = [c for c in df.columns 
                if c not in ['SalaryBand', 'SalaryBandEncoded']] 
X = df[feature_cols].values 
y = df['SalaryBandEncoded'].values 
 
X_train, X_temp, y_train, y_temp = train_test_split( 
    X, y, test_size=0.30, random_state=42, stratify=y 
) 
X_val, X_test, y_val, y_test = train_test_split( 
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp 
) 
 
print(f"Training set:   {X_train.shape[0]} rows") 
print(f"Validation set: {X_val.shape[0]} rows") 
print(f"Test set:       {X_test.shape[0]} rows")

# 7. SCALE FEATURES (fit on training data only) 
print("\n--- SCALING FEATURES (StandardScaler) ---") 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_val = scaler.transform(X_val) 
X_test = scaler.transform(X_test) 
 
print(f"Feature means (train): {X_train.mean(axis=0).round(3)}") 
print(f"Feature stds  (train): {X_train.std(axis=0).round(3)}") 
 
joblib.dump(scaler, "../../models/scaler.pkl") 
joblib.dump(feature_cols, "../../models/feature_cols.pkl") 
 
np.save("../../models/X_train.npy", X_train) 
np.save("../../models/X_val.npy", X_val) 
np.save("../../models/X_test.npy", X_test) 
np.save("../../models/y_train.npy", y_train) 
np.save("../../models/y_val.npy", y_val) 
np.save("../../models/y_test.npy", y_test)

# 8. EXPLORE CLASS DISTRIBUTION IN EACH SPLIT 
print("\n--- CLASS DISTRIBUTION IN EACH SPLIT ---") 
for name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]: 
    counts = np.bincount(y_split) 
    print(f"{name}: {counts} (classes: {label_encoder.classes_})") 
 
print("\n" + "=" * 60) 
print("Data prepared and saved to models/ folder.") 
print("Ready for learning category selection (PS03).") 
print("=" * 60)
