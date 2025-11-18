# prepare_train_val_test.py
import pandas as pd
import numpy as np
import os

 # input rolling output (relative to this script)
IN_CSV = os.path.join(os.path.dirname(__file__), "data", "rolls", "/Users/rishimodi/Desktop/Match_Predictor/src/data/rolls/epl_all_seasons_matches_with_rolls.csv")
OUT_PREFIX = "epl_all_seasons"                      # will write epl_2024_train.csv, _val.csv, _test.csv
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
ROLL_WINDOW = 5
IMPUTE_STRATEGY = "median"  # median from train

if not os.path.exists(IN_CSV):
    raise SystemExit(f"Input csv not found: {IN_CSV}")

df = pd.read_csv(IN_CSV)
df = df.sort_values("match_id").reset_index(drop=True)
n = len(df)
i1 = int(n * TRAIN_FRAC)
i2 = int(n * (TRAIN_FRAC + VAL_FRAC))

train = df.iloc[:i1].copy()
val = df.iloc[i1:i2].copy()
test = df.iloc[i2:].copy()

print(f"Rows: total={n}, train={len(train)}, val={len(val)}, test={len(test)}")

# determine rolling shrunk columns we created (columns that end with _shr{ROLL_WINDOW})
suffix = f"_shr{ROLL_WINDOW}"
shr_cols = [c for c in df.columns if c.endswith(suffix)]
# also include matches_played columns
matches_play_cols = [c for c in df.columns if f"matches_played_roll{ROLL_WINDOW}" in c]
all_feature_cols = shr_cols + matches_play_cols

print("Detected shrunk cols (examples):", shr_cols[:8])

# compute train priors (medians) for each shrunk stat and for imputation
train_priors = {}
for c in shr_cols:
    train_priors[c] = train[c].median(skipna=True)
    # fallback safe value if median is NaN
    if pd.isna(train_priors[c]):
        train_priors[c] = 0.0

# Impute val/test using train medians
def impute_with_train_priors(df_part):
    for c, v in train_priors.items():
        if c in df_part.columns:
            if IMPUTE_STRATEGY == "median":
                df_part[c] = df_part[c].fillna(v)
            elif IMPUTE_STRATEGY == "zero":
                df_part[c] = df_part[c].fillna(0.0)
    # also impute matches_played with 0
    for c in matches_play_cols:
        if c in df_part.columns:
            df_part[c] = df_part[c].fillna(0)
    return df_part

train = impute_with_train_priors(train)
val = impute_with_train_priors(val)
test = impute_with_train_priors(test)


# ensure output directory exists (data/splits relative to this script)
splits_dir = os.path.join(os.path.dirname(__file__), "data", "splits")
os.makedirs(splits_dir, exist_ok=True)

train_path = os.path.join(splits_dir, f"{OUT_PREFIX}_train.csv")
val_path = os.path.join(splits_dir, f"{OUT_PREFIX}_val.csv")
test_path = os.path.join(splits_dir, f"{OUT_PREFIX}_test.csv")

train.to_csv(train_path, index=False)
val.to_csv(val_path, index=False)
test.to_csv(test_path, index=False)

print("Saved train/val/test CSVs:")
print(f"  {train_path} ({len(train)})")
print(f"  {val_path}   ({len(val)})")
print(f"  {test_path}  ({len(test)})")
