import pandas as pd

# Load CSV
matches = pd.read_csv("epl_2024_matches.csv", index_col="match_id")

# Columns that are percentages and need conversion
percent_cols = ["CHANCES_home", "CHANCES_away"]

# Convert percentages to float
for col in percent_cols:
    matches[col] = matches[col].str.rstrip('%').astype(float)

# All numeric columns now
numeric_cols = [
    "CHANCES_home", "CHANCES_away", "GOALS_home", "GOALS_away",
    "XG_home", "XG_away", "SHOTS_home", "SHOTS_away",
    "SHOTS ON TARGET_home", "SHOTS ON TARGET_away",
    "DEEP_home", "DEEP_away", "PPDA_home", "PPDA_away",
    "XPTS_home", "XPTS_away"
]

# Fill NaNs with median
for col in numeric_cols:
    median_val = matches[col].median()
    matches[col] = matches[col].fillna(median_val)
    print(f"Filled NaNs in {col} with median value {median_val}")

# Verify
print(matches.isnull().sum())

# Save cleaned CSV
matches.to_csv("epl_2024_matches_clean.csv")
print("✅ Cleaned CSV saved!")
