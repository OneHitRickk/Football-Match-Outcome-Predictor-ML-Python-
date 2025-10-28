import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# Reading the CSV
matches = pd.read_csv("epl_2024_matches_clean.csv")

# Normalizing the column names
matches.columns = matches.columns.str.strip().str.lower()

# removing '%' and convert numerics safely
for col in matches.columns:
    if matches[col].dtype == "object":
        if matches[col].str.contains('%').any():
            matches[col] = matches[col].str.replace('%', '', regex=False)
    # try to convert every column that looks numeric
    try:
        matches[col] = pd.to_numeric(matches[col])
    except:
        pass

# Encoders
le_team = LabelEncoder()
all_teams = pd.concat([matches['teams_home'], matches['teams_away']]).unique()
le_team.fit(all_teams)
matches['home_team'] = le_team.transform(matches['teams_home'])
matches['away_team'] = le_team.transform(matches['teams_away'])


def encode_result(row):
    if row['goals_home'] > row['goals_away']:
        return 2  # Home win
    elif row['goals_home'] == row['goals_away']:
        return 1  # Draw
    else:
        return 0  # Away win


matches['result'] = matches.apply(encode_result, axis=1)

# === Step 1.4: Sort chronologically ===
if 'match_id' in matches.columns:
    matches = matches.sort_values(by='match_id').reset_index(drop=True)

# Temp checks for now...//todo
# print(" Columns:", matches.columns.tolist())
# print("\n Dtypes:\n", matches.dtypes)
# print("\n Result label counts:\n", matches['result'].value_counts())
# print("\nSample rows:\n", matches.head(5))
# print("Step 1 Over")


# Step 2: Rolling stats computation (window = 5)
window_size = 5
stats_cols = ['chances', 'xg', 'shots', 'shots on target', 'deep', 'ppda', 'xpts']

# Initialize rolling columns
for s in stats_cols:
    matches[f'rolling_{s}_home'] = pd.NA
    matches[f'rolling_{s}_away'] = pd.NA

# Compute rolling stats for each team
for team in matches['teams_home'].unique():
    # --- HOME matches ---
    home_mask = matches['teams_home'] == team
    team_home = matches.loc[home_mask, ['match_id'] + [f'{s}_home' for s in stats_cols]].sort_values('match_id')

    rolling_home = (
        team_home[[f'{s}_home' for s in stats_cols]]
        .shift(1)
        .rolling(window=window_size, min_periods=1)
        .mean()
    )
    matches.loc[rolling_home.index, [f'rolling_{s}_home' for s in stats_cols]] = rolling_home.values

    # --- AWAY matches ---
    away_mask = matches['teams_away'] == team
    team_away = matches.loc[away_mask, ['match_id'] + [f'{s}_away' for s in stats_cols]].sort_values('match_id')

    rolling_away = (
        team_away[[f'{s}_away' for s in stats_cols]]
        .shift(1)
        .rolling(window=window_size, min_periods=1)
        .mean()
    )
    matches.loc[rolling_away.index, [f'rolling_{s}_away' for s in stats_cols]] = rolling_away.values

# Fill any NaNs from the first few matches with the column mean
rolling_cols = [f'rolling_{s}_home' for s in stats_cols] + [f'rolling_{s}_away' for s in stats_cols]
matches[rolling_cols] = matches[rolling_cols].fillna(matches[rolling_cols].mean())

print(" Step 2: Rolling averages computed successfully!")


# STEP 3: FEATURE SET CONSTRUCTION

print("\n Step 3: Building the training feature set...\n")

rolling_cols = [col for col in matches.columns if col.startswith('rolling_')]
base_features = ['home_team', 'away_team']  # encoded team IDs
target_col = 'result'

X = matches[base_features + rolling_cols].copy()
y = matches[target_col].copy()

# print(f"Features selected: {len(X.columns)} columns")
# print(f" Total matches: {len(X)}")
# print(f" Target distribution:\n{y.value_counts().sort_index().to_dict()}\n")

# Check for any missing values
missing = X.isna().sum().sum()
if missing > 0:
    print(f"Warning: {missing} missing values found in X — filling with column means.")
    X = X.fillna(X.mean())
else:
    print(" No missing values in feature set.")

print("\n Feature preview (first 5 rows):")
print(X.head(5))

print("\n Step 3 complete — dataset ready for ML training!\n")


# todo
matches_final = pd.concat([X, y], axis=1)
matches_final.to_csv("epl_2024_prepared.csv", index=False)



features = X.copy()
features["target"] = y


#  STEP 4: MODEL TRAINING


print("\n Step 4: Training Random Forest model...")

# 1. Prepare data

X = features.drop(columns=["target"])
y = features["target"]

# Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Train model //todo
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

# 3. Predictions & Evaluation
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy: {accuracy:.3f}")

print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

print("\n Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# For better visualization//todo
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Home Win", "Draw", "Away Win"], yticklabels=["Home Win", "Draw", "Away Win"])
# plt.title("Confusion Matrix - Random Forest")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# 4. Feature Importance
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n Top 10 Important Features:")
print(importances.head(10))
