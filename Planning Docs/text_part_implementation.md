# 🧠 Clinical / Text Branch: XGBoost Pipeline

This section covers the **tabular clinical data** used by the **XGBoost classifier** in the architecture diagram.  
In this project, that means the ADNI clinical fields that describe the subject at baseline, not MRI pixels.

---

## 🛠️ Where to Get the Data

Use the **ADNI Data Archive / LONI** and download the clinical files that contain:

- **Subject ID** (`PTID`)
- **Visit code** (`VISCODE2`)
- **Diagnosis label** (`CN` vs `AD`)
- **MMSE**
- **Age**
- **Sex**
- **APOE-ε4 / APOE4 status**

For this branch, the easiest setup is to create **one merged clinical CSV** that already contains all of those columns.

Typical ADNI files used here are:

- `DXSUM_15Apr2026.csv` for baseline diagnosis and subject IDs
- a master clinical/demographics table for `MMSE`, `AGE`, and `SEX`
- an APOE/genetics table if `APOE4` is not already in the master file

If your ADNI download is split across multiple files, merge them first by `PTID` and `VISCODE2`, then upload the merged file to Kaggle as `clinical_master.csv`.

> Suggested Kaggle input folder: `/kaggle/input/your-clinical-dataset-name/`

---

## 🛠️ Step 1: Install the Required Libraries

```python
!pip install xgboost scikit-learn pandas numpy joblib
```

---

## 🛠️ Step 2: Load the Clinical Dataset

This cell loads the merged clinical file and checks the available columns.

```python
import os
import pandas as pd
import numpy as np

clinical_root = "/kaggle/input/your-clinical-dataset-name"
clinical_csv = os.path.join(clinical_root, "clinical_master.csv")  # replace if needed

df = pd.read_csv(clinical_csv, low_memory=False)

print("Shape:", df.shape)
print("Columns:")
print(df.columns.tolist())
```

---

## 🛠️ Step 3: Keep Only Baseline CN and AD Subjects

This cell keeps the baseline visit only and converts the diagnosis to a binary label:

- `0` = CN
- `1` = AD

```python
df.columns = [c.strip() for c in df.columns]

df["PTID"] = df["PTID"].astype(str).str.strip()
df["VISCODE2"] = df["VISCODE2"].astype(str).str.strip().str.lower()

baseline_df = df[df["VISCODE2"] == "bl"].copy()

diag_col = None
for candidate in ["DIAGNOSIS", "DX", "DX_bl"]:
    if candidate in baseline_df.columns:
        diag_col = candidate
        break

if diag_col is None:
    raise KeyError("No diagnosis column found. Expected one of: DIAGNOSIS, DX, DX_bl")

baseline_df = baseline_df[baseline_df[diag_col].isin([1, 3, "CN", "AD"])]

def map_label(value):
    if value in [1, "CN"]:
        return 0
    if value in [3, "AD"]:
        return 1
    return np.nan

baseline_df["label"] = baseline_df[diag_col].apply(map_label)
baseline_df = baseline_df.dropna(subset=["label"]).copy()
baseline_df["label"] = baseline_df["label"].astype(int)

# Optional: keep only the same subjects used in the MRI branch
# target_subjects = [...]
# baseline_df = baseline_df[baseline_df["PTID"].isin(target_subjects)].copy()

print("Baseline samples:", baseline_df.shape[0])
print(baseline_df[["PTID", diag_col, "label"]].head())
```

---

## 🛠️ Step 4: Build the Feature Table

This cell selects the clinical features used by the XGBoost branch.  
It follows the architecture diagram and roadmap:

- One-hot encode: `APOE4`, `SEX`
- Z-score normalize: `MMSE`, `AGE`

```python
def first_existing(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of these columns were found: {candidates}")

mmse_col = first_existing(baseline_df, ["MMSE", "MMSCORE", "mmse"])
age_col = first_existing(baseline_df, ["AGE", "AGE_bl", "PTAGE", "age"])
sex_col = first_existing(baseline_df, ["PTGENDER", "SEX", "sex"])
apoe_col = first_existing(baseline_df, ["APOE4", "APOE-ε4", "APOE_E4", "APOE"])

feature_df = baseline_df[["PTID", "label", mmse_col, age_col, sex_col, apoe_col]].copy()
feature_df = feature_df.rename(
    columns={
        mmse_col: "MMSE",
        age_col: "AGE",
        sex_col: "SEX",
        apoe_col: "APOE4",
    }
)

print(feature_df.head())
print(feature_df.isna().sum())
```

---

## 🛠️ Step 5: Encode and Normalize the Clinical Features

This cell creates the preprocessing pipeline and splits the data into train/test sets.

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = feature_df[["MMSE", "AGE", "SEX", "APOE4"]]
y = feature_df["label"]
subject_ids = feature_df["PTID"]

numeric_features = ["MMSE", "AGE"]
categorical_features = ["SEX", "APOE4"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, subject_ids,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
```

---

## 🛠️ Step 6: Train the XGBoost Classifier

This cell trains the clinical classifier and reports the evaluation results.

```python
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos if pos > 0 else 1.0

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", xgb_model),
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
```

---

## 🛠️ Step 7: Save the Clinical Logic Clues

This cell exports the XGBoost probability scores for later multimodal fusion.

```python
import joblib

all_probs = clf.predict_proba(X)[:, 1]
logic_clues = pd.DataFrame({
    "PTID": subject_ids,
    "AD_probability": all_probs,
    "predicted_label": (all_probs >= 0.5).astype(int),
})

output_dir = "/kaggle/working/clinical_xgb_outputs"
os.makedirs(output_dir, exist_ok=True)

logic_path = os.path.join(output_dir, "xgb_logic_clues.csv")
model_path = os.path.join(output_dir, "xgb_clinical_model.joblib")

logic_clues.to_csv(logic_path, index=False)
joblib.dump(clf, model_path)

print("Saved:", logic_path)
print("Saved:", model_path)
print(logic_clues.head())
```

---

## 🚀 What’s Next?

Once this branch is ready, the `AD_probability` output can be used as the **clinical logic clue** for the later multimodal fusion stage.
