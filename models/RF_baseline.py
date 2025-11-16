# model_baseline_rf.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os, sys

# -------------------------------
# CONFIG
# -------------------------------
TRAIN_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/splits/epl_2024_train.csv"
VAL_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/splits/epl_2024_val.csv"
TEST_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/splits/epl_2024_test.csv"
ENCODED_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/encoded/epl_2024_matches_encoded.csv"  # source of 'result' if missing
MODEL_OUT = "rf_baseline_model.pkl"

# -------------------------------
# LOAD DATA (with sanity checks)
# -------------------------------
for path in (TRAIN_CSV, VAL_CSV, TEST_CSV):
    if not os.path.exists(path):
        print(f"ERROR: Required file not found: {path}")
        sys.exit(1)

train = pd.read_csv(TRAIN_CSV)
val = pd.read_csv(VAL_CSV)
test = pd.read_csv(TEST_CSV)
print(f"✅ Loaded data: train={len(train)}, val={len(val)}, test={len(test)}")

# -------------------------------
# Ensure target column 'result' exists (merge from encoded if necessary)
# -------------------------------
if "result" not in train.columns or "result" not in val.columns or "result" not in test.columns:
    if not os.path.exists(ENCODED_CSV):
        print("ERROR: 'result' missing in splits and encoded CSV not found.")
        sys.exit(1)
    print("⚠️ 'result' not found in one or more split files — merging from encoded dataset...")
    encoded = pd.read_csv(ENCODED_CSV, usecols=["match_id", "result"])
    # coerce match_id types for safe merging
    for df_name, df in (("train", train), ("val", val), ("test", test)):
        if "match_id" not in df.columns:
            print(f"ERROR: match_id missing from {df_name} split — cannot merge result.")
            sys.exit(1)
    train = train.merge(encoded, on="match_id", how="left")
    val = val.merge(encoded, on="match_id", how="left")
    test = test.merge(encoded, on="match_id", how="left")

# -------------------------------
# FEATURE & TARGET SELECTION
# -------------------------------
features = [
    # rolling/shrunk averages (adjust if your column names differ)
    "home_xg_shr5", "away_xg_shr5",
    "home_chance_prob_shr5", "away_chance_prob_shr5",
    "home_win_rate_shr5", "away_win_rate_shr5",
    "form_diff_shr5", "xg_diff_shr5",
    "chance_prob_diff_shr5",
    # team identifiers (must be numeric)
    "home_team_id", "away_team_id",
]

target = "result"

# -------------------------------
# Basic column checks
# -------------------------------
def check_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in {name}: {missing}")
        return False
    return True

if not (check_columns(train, features + [target], "train") and
        check_columns(val, features + [target], "val") and
        check_columns(test, features + [target], "test")):
    print("Fix missing columns and rerun.")
    sys.exit(1)

# -------------------------------
# Coerce types for team ids and features
# -------------------------------
for df in (train, val, test):
    # team ids to numeric (if they are strings)
    for tid in ("home_team_id", "away_team_id"):
        if tid in df.columns:
            df[tid] = pd.to_numeric(df[tid], errors="coerce")

# -------------------------------
# Drop rows without target (if any) - report counts
# -------------------------------
before_train = len(train)
train = train.dropna(subset=[target])
dropped_train = before_train - len(train)

before_val = len(val)
val = val.dropna(subset=[target])
dropped_val = before_val - len(val)

before_test = len(test)
test = test.dropna(subset=[target])
dropped_test = before_test - len(test)

print(f"Dropped rows with missing target -> train: {dropped_train}, val: {dropped_val}, test: {dropped_test}")

# -------------------------------
# Impute missing feature values using TRAIN medians (honest imputation)
# -------------------------------
train_medians = {}
for feat in features:
    med = pd.to_numeric(train[feat], errors="coerce").median(skipna=True)
    if pd.isna(med):
        # fallback to 0 if median cannot be computed
        med = 0.0
    train_medians[feat] = med
    # fill train, val, test
    train[feat] = pd.to_numeric(train[feat], errors="coerce").fillna(med)
    val[feat] = pd.to_numeric(val[feat], errors="coerce").fillna(med)
    test[feat] = pd.to_numeric(test[feat], errors="coerce").fillna(med)

print("✅ Imputed missing feature values using train medians.")

# -------------------------------
# Split Features & Labels
# -------------------------------
X_train, y_train = train[features].copy(), train[target].astype(int)
X_val, y_val = val[features].copy(), val[target].astype(int)
X_test, y_test = test[features].copy(), test[target].astype(int)

# final shapes
print(f"Shapes -> X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# -------------------------------
# MODEL: Random Forest (baseline)
# -------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"  # helpful since draws are rare
)
rf.fit(X_train, y_train)
print("🌲 Random Forest trained successfully!")

# -------------------------------
# VALIDATION EVALUATION
# -------------------------------
y_val_pred = rf.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print("\n📊 Validation Performance:")
print(f"Accuracy: {val_acc:.3f}")
print(classification_report(y_val, y_val_pred, digits=3))

# -------------------------------
# TEST EVALUATION
# -------------------------------
y_test_pred = rf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print("\n🧪 Test Performance:")
print(f"Accuracy: {test_acc:.3f}")
print(classification_report(y_test, y_test_pred, digits=3))

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, y_test_pred))

# -------------------------------
# SAVE MODEL
# -------------------------------
# joblib.dump(rf, MODEL_OUT)
# print(f"\n💾 Model saved as {MODEL_OUT}")
