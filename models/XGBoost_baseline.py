# model_xgb.py
"""
XGBoost multiclass baseline pipeline (skeleton)
- Loads train/val/test CSVs (created by prepare_train_val_test.py)
- Builds numeric preprocessing (median impute + optional scaling)
- Trains XGBClassifier with early stopping on validation set
- Evaluates and saves model + reports metrics
"""

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ----------------------------
# CONFIG - change as needed
# ----------------------------
# TRAIN_CSV = "epl_2024_train.csv"
# VAL_CSV = "epl_2024_val.csv"
# TEST_CSV = "epl_2024_test.csv"
MODEL_OUT = "/Users/rishimodi/Desktop/Match_Predictor/models/xgb_baseline_model.pkl"

TRAIN_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/splits/epl_all_seasons_train.csv"
VAL_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/splits/epl_all_seasons_val.csv"
TEST_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/splits/epl_all_seasons_test.csv"

# FEATURES: list the exact column names you want to feed to XGBoost
# Example based on your pipeline — adjust if your columns differ
FEATURES = [
    "home_xg_shr5", "away_xg_shr5",
    "home_chance_prob_shr5", "away_chance_prob_shr5",
    "home_win_rate_shr5", "away_win_rate_shr5",
    "form_diff_shr5", "xg_diff_shr5",
    "chance_prob_diff_shr5",
    # team ids or target-encodings (choose one strategy)
    "home_team_id", "away_team_id",
    # if you have target-encoded columns, you can include them instead:
    # "home_enc_homewin", "away_enc_awaywin", "home_enc_draw", "away_enc_draw"
]

TARGET = "result"  # 0=away,1=draw,2=home

# XGBoost hyperparams (sensible starting point)
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": 3,
    "random_state": 42,
    "use_label_encoder": False,  # avoid warning for older versions
    "verbosity": 0
}

EARLY_STOPPING_ROUNDS = 30  # stop if no val improvement
VERBOSE_EVAL = False        # make True to print XGBoost training logs

# ----------------------------
# LOAD DATA
# ----------------------------
for p in (TRAIN_CSV, VAL_CSV, TEST_CSV):
    if not os.path.exists(p):
        raise SystemExit(f"Missing file: {p}")

train = pd.read_csv(TRAIN_CSV)
val = pd.read_csv(VAL_CSV)
test = pd.read_csv(TEST_CSV)
print(f"Loaded: train={len(train)}, val={len(val)}, test={len(test)}")

# ----------------------------
# BASIC SANITY: ensure features & target present
# ----------------------------
missing_feats = [c for c in FEATURES if c not in train.columns]
if missing_feats:
    raise SystemExit(f"Missing feature columns in train: {missing_feats}")
if TARGET not in train.columns:
    raise SystemExit("Target column not found in train/val/test")

# Drop rows without target if any (should be none)
train = train.dropna(subset=[TARGET]).reset_index(drop=True)
val = val.dropna(subset=[TARGET]).reset_index(drop=True)
test = test.dropna(subset=[TARGET]).reset_index(drop=True)

# ----------------------------
# PREPROCESSING PIPELINE
# - numeric_imputer: median (fit on train only)
# - scaler: optional but helps linear/regularized models; XGBoost doesn't require it
# ----------------------------
numeric_features = FEATURES[:]  # in this skeleton we treat all FEATURES as numeric
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())   # optional: keeps feature scales similar
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    # if you had categorical features, you'd add an encoder here
])

# ----------------------------
# BUILD FULL PIPELINE (preproc -> model)
# ----------------------------
xgb = XGBClassifier(**XGB_PARAMS)
pipeline = Pipeline(steps=[
    ("preproc", preprocessor),
    ("xgb", xgb)
])

# ----------------------------
# FITTING: Use early stopping with validation set
# Note: we pass preprocessed validation data via eval_set inside XGBoost by calling
#       pipeline.named_steps['xgb'].fit(...) after transforming the data.
# This avoids leaking val through pipeline.fit if your preprocess has state.
# ----------------------------
X_train = train[FEATURES]
y_train = train[TARGET].astype(int)

X_val = val[FEATURES]
y_val = val[TARGET].astype(int)

# Fit preprocessing on train, transform train/val for xgboost early stopping
preproc = pipeline.named_steps["preproc"]
preproc.fit(X_train)                # fit only on train
X_train_pre = preproc.transform(X_train)
X_val_pre = preproc.transform(X_val)
X_test_pre = preproc.transform(test[FEATURES])

# Fit XGB with early stopping (use eval_set)
xgb = pipeline.named_steps["xgb"]
xgb.fit(
    X_train_pre, y_train,
    eval_set=[(X_val_pre, y_val)],
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose=VERBOSE_EVAL
)

# attach the trained xgb back into the pipeline object for convenience
pipeline.named_steps["xgb"] = xgb
pipeline.steps[-1] = ("xgb", xgb)

# ----------------------------
# EVALUATION
# ----------------------------
def evaluate_model(model, X_pre, y_true, split_name="test"):
    y_pred = model.predict(X_pre)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n--- {split_name.upper()} ---")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_true, y_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    return y_pred

# evaluate val and test
evaluate_model(xgb, X_val_pre, y_val, split_name="val")
evaluate_model(xgb, X_test_pre, test[TARGET].astype(int), split_name="test")

# ----------------------------
# SAVE the whole pipeline (preproc + xgb)
# We'll wrap them to keep single object: store a dict
# ----------------------------
to_save = {
    "preproc": preproc,
    "xgb": xgb,
    "features": FEATURES
}
joblib.dump(to_save, MODEL_OUT)
print(f"\nSaved model artifacts to {MODEL_OUT}")