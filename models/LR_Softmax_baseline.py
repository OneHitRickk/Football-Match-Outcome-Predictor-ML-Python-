# model_logistic_pipeline.py
"""
Train a multinomial Logistic Regression pipeline:
- imputes numeric features (median on TRAIN)
- scales numeric features (StandardScaler)
- one-hot encodes team ids (home/away)
- fits LogisticRegression (multinomial / softmax)
- evaluates on val and test
- saves pipeline artifact
"""

import os, sys, joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------- CONFIG -----------------
# Data for 2024 season
# TRAIN_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/splits/epl_2024_train.csv"
# VAL_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/splits/epl_2024_val.csv"
# TEST_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/splits/epl_2024_test.csv"
# ENCODED_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/encoded/epl_2024_matches_encoded.csv"  # fallback for a result if missing

# all seasons
TRAIN_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/splits/epl_all_seasons_train.csv"
VAL_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/splits/epl_all_seasons_val.csv"
TEST_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/splits/epl_all_seasons_test.csv"
ENCODED_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/encoded/epl_all_seasons_matches_encoded.csv"  # source of 'result' if missing

PIPE_OUT = "lr_pipeline.pkl"

# smoothing param for encoding (tune later)
SMOOTH_K = 10.0

TARGET_COL = "result"
ROLL_WINDOW = 5  # just for naming clarity

# numeric features we want to keep (existing rolling/shrunk stats)
BASE_NUMERIC_FEATURES = [
    "home_xg_shr5", "away_xg_shr5",
    "home_chance_prob_shr5", "away_chance_prob_shr5",
    "home_win_rate_shr5", "away_win_rate_shr5",
    "form_diff_shr5", "xg_diff_shr5",
    "chance_prob_diff_shr5"
]

# --------- LOAD ---------
for p in (TRAIN_CSV, VAL_CSV, TEST_CSV):
    if not os.path.exists(p):
        print(f"ERROR: missing file: {p}")
        sys.exit(1)

train = pd.read_csv(TRAIN_CSV)
val = pd.read_csv(VAL_CSV)
test = pd.read_csv(TEST_CSV)
print(f"Loaded: train={len(train)}, val={len(val)}, test={len(test)}")

# Merge result if missing in splits
if TARGET_COL not in train.columns or TARGET_COL not in val.columns or TARGET_COL not in test.columns:
    if not os.path.exists(ENCODED_CSV):
        print("ERROR: 'result' missing and encoded CSV not available.")
        sys.exit(1)
    enc = pd.read_csv(ENCODED_CSV, usecols=["match_id", "result"])
    train = train.merge(enc, on="match_id", how="left")
    val = val.merge(enc, on="match_id", how="left")
    test = test.merge(enc, on="match_id", how="left")
    print("Merged 'result' into splits from encoded CSV.")

# ensure team id columns exist
for c in ("home_team_id", "away_team_id"):
    if c not in train.columns:
        print(f"ERROR: required column missing in splits: {c}")
        sys.exit(1)

# Drop missing targets (report)
before = (len(train), len(val), len(test))
train = train.dropna(subset=[TARGET_COL])
val = val.dropna(subset=[TARGET_COL])
test = test.dropna(subset=[TARGET_COL])
after = (len(train), len(val), len(test))
print(f"Dropped rows w/o target -> train: {before[0]-after[0]}, val: {before[1]-after[1]}, test: {before[2]-after[2]}")

# coerce team ids numeric (safe)
train["home_team_id"] = pd.to_numeric(train["home_team_id"], errors="coerce")
train["away_team_id"] = pd.to_numeric(train["away_team_id"], errors="coerce")
val["home_team_id"] = pd.to_numeric(val["home_team_id"], errors="coerce")
val["away_team_id"] = pd.to_numeric(val["away_team_id"], errors="coerce")
test["home_team_id"] = pd.to_numeric(test["home_team_id"], errors="coerce")
test["away_team_id"] = pd.to_numeric(test["away_team_id"], errors="coerce")

# ---------- TARGET ENCODING (TRAIN-ONLY SMOOTHED) ----------
# create binary indicator columns on TRAIN
train['is_home_win'] = (train[TARGET_COL] == 2).astype(int)
train['is_draw']     = (train[TARGET_COL] == 1).astype(int)
train['is_away_win'] = (train[TARGET_COL] == 0).astype(int)

def smooth_enc_map(train_df, team_col, binary_col, k):
    """
    Return (mapping dict, global_prior)
    mapping: team_id -> smoothed probability of binary_col
    """
    g = train_df.groupby(team_col)[binary_col].agg(['sum','count']).reset_index()
    global_prior = train_df[binary_col].mean()
    # avoid division by zero for teams with 0 count (shouldn't happen in groupby)
    g['enc'] = (g['sum'] + k * global_prior) / (g['count'] + k)
    return dict(zip(g[team_col], g['enc'])), global_prior

# Compute maps (train-only)
home_win_map, home_win_prior = smooth_enc_map(train, 'home_team_id', 'is_home_win', SMOOTH_K)
home_draw_map, home_draw_prior = smooth_enc_map(train, 'home_team_id', 'is_draw', SMOOTH_K)
away_win_map, away_win_prior = smooth_enc_map(train, 'away_team_id', 'is_away_win', SMOOTH_K)
away_draw_map, away_draw_prior = smooth_enc_map(train, 'away_team_id', 'is_draw', SMOOTH_K)

# Apply maps to train/val/test with fallback to train prior
def apply_enc(df):
    df['home_enc_homewin'] = df['home_team_id'].map(home_win_map).fillna(home_win_prior)
    df['home_enc_draw']    = df['home_team_id'].map(home_draw_map).fillna(home_draw_prior)
    df['away_enc_awaywin'] = df['away_team_id'].map(away_win_map).fillna(away_win_prior)
    df['away_enc_draw']    = df['away_team_id'].map(away_draw_map).fillna(away_draw_prior)
    return df

train = apply_enc(train)
val = apply_enc(val)
test = apply_enc(test)

print("Applied target encodings (smoothed) to train/val/test.")
print("Example train encodings (head):")
print(train[["home_team_id","home_enc_homewin","home_enc_draw"]].head().to_string(index=False))

# -------- assemble final numeric features --------
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + [
    "home_enc_homewin", "home_enc_draw",
    "away_enc_awaywin", "away_enc_draw"
]

# verify features present
missing = [c for c in NUMERIC_FEATURES if c not in train.columns]
if missing:
    print("ERROR: Missing numeric features in train:", missing)
    sys.exit(1)

# coerce numeric types for features
for df in (train, val, test):
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

X_train = train[NUMERIC_FEATURES].copy()
y_train = train[TARGET_COL].astype(int)

X_val = val[NUMERIC_FEATURES].copy()
y_val = val[TARGET_COL].astype(int)

X_test = test[NUMERIC_FEATURES].copy()
y_test = test[TARGET_COL].astype(int)

print(f"Feature matrix shapes -> X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# -------- PIPELINE (numeric only) --------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # median from training will be computed by fit
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_FEATURES)
], remainder="drop", sparse_threshold=0)

clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

pipeline = Pipeline(steps=[("pre", preprocessor), ("clf", clf)])

# -------- TRAIN --------
print("Fitting logistic regression pipeline (with target encodings)...")
pipeline.fit(X_train, y_train)
print("Trained.")

# -------- EVALUATE --------
def evaluate(name, X, y):
    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y, preds, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y, preds))
    return preds

_ = evaluate("Validation", X_val, y_val)
_ = evaluate("Test", X_test, y_test)

# -------- FEATURE IMPORTANCE (coeffs) --------
coefs = pipeline.named_steps["clf"].coef_  # shape = (n_classes, n_features)
classes = pipeline.named_steps["clf"].classes_

print("\nModel classes:", classes)
print("Numeric feature coefficients (per class):")
for cls_idx, cls in enumerate(classes):
    pairs = list(zip(NUMERIC_FEATURES, coefs[cls_idx]))
    pairs_sorted = sorted(pairs, key=lambda x: -abs(x[1]))[:12]
    print(f"\nClass {cls} top numeric weights:")
    for n, v in pairs_sorted:
        print(f"  {n}: {v:.4f}")

# -------- SAVE PIPELINE --------
joblib.dump(pipeline, PIPE_OUT)
print(f"\nPipeline saved to {PIPE_OUT}")