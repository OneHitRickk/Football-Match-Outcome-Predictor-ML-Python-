import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# ==========================
# STEP 1: DATA CLEANING
# ==========================
matches = pd.read_csv("epl_2024_matches_clean.csv")

# Normalize column names
matches.columns = matches.columns.str.strip().str.lower()

# Remove '%' and safely convert numerics
for col in matches.columns:
    if matches[col].dtype == "object":
        if matches[col].str.contains('%').any():
            matches[col] = matches[col].str.replace('%', '', regex=False)
    try:
        matches[col] = pd.to_numeric(matches[col])
    except:
        pass

# Label encode team names
le_team = LabelEncoder()
all_teams = pd.concat([matches['teams_home'], matches['teams_away']]).unique()
le_team.fit(all_teams)
matches['home_team'] = le_team.transform(matches['teams_home'])
matches['away_team'] = le_team.transform(matches['teams_away'])


# Encode match result: 2 = Home Win, 1 = Draw, 0 = Away Win
def encode_result(row):
    if row['goals_home'] > row['goals_away']:
        return 2
    elif row['goals_home'] == row['goals_away']:
        return 1
    else:
        return 0


matches['result'] = matches.apply(encode_result, axis=1)

# Sort chronologically if match_id exists
if 'match_id' in matches.columns:
    matches = matches.sort_values(by='match_id').reset_index(drop=True)

print("✅ Step 1: Data cleaned and normalized successfully!\n")

# ==========================
# STEP 2: ROLLING STATS
# ==========================
window_size = 5
stats_cols = ['chances', 'xg', 'shots', 'shots on target', 'deep', 'ppda', 'xpts']

# Initialize rolling columns
for s in stats_cols:
    matches[f'rolling_{s}_home'] = pd.NA
    matches[f'rolling_{s}_away'] = pd.NA

# Compute rolling averages
for team in matches['teams_home'].unique():
    # Home matches
    home_mask = matches['teams_home'] == team
    team_home = matches.loc[home_mask, ['match_id'] + [f'{s}_home' for s in stats_cols]].sort_values('match_id')
    rolling_home = (
        team_home[[f'{s}_home' for s in stats_cols]]
        .shift(1)
        .rolling(window=window_size, min_periods=1)
        .mean()
    )
    matches.loc[rolling_home.index, [f'rolling_{s}_home' for s in stats_cols]] = rolling_home.values

    # Away matches
    away_mask = matches['teams_away'] == team
    team_away = matches.loc[away_mask, ['match_id'] + [f'{s}_away' for s in stats_cols]].sort_values('match_id')
    rolling_away = (
        team_away[[f'{s}_away' for s in stats_cols]]
        .shift(1)
        .rolling(window=window_size, min_periods=1)
        .mean()
    )
    matches.loc[rolling_away.index, [f'rolling_{s}_away' for s in stats_cols]] = rolling_away.values

# Fill NaNs from first matches with column mean
rolling_cols = [f'rolling_{s}_home' for s in stats_cols] + [f'rolling_{s}_away' for s in stats_cols]
matches[rolling_cols] = matches[rolling_cols].fillna(matches[rolling_cols].mean())

print("✅ Step 2: Rolling averages computed successfully!\n")

# ==========================
# STEP 2.5: CONTEXT FEATURES
# ==========================
matches['home_win'] = (matches['goals_home'] > matches['goals_away']).astype(int)
matches['away_win'] = (matches['goals_away'] > matches['goals_home']).astype(int)
matches['draw'] = (matches['goals_home'] == matches['goals_away']).astype(int)

window_size = 5

# Form calculation
for team in matches['teams_home'].unique():
    home_mask = matches['teams_home'] == team
    away_mask = matches['teams_away'] == team

    team_home = matches.loc[home_mask].sort_values('match_id')
    matches.loc[home_mask, 'home_form'] = team_home['home_win'].shift(1).rolling(window=window_size,
                                                                                 min_periods=1).mean()

    team_away = matches.loc[away_mask].sort_values('match_id')
    matches.loc[away_mask, 'away_form'] = team_away['away_win'].shift(1).rolling(window=window_size,
                                                                                 min_periods=1).mean()

matches[['home_form', 'away_form']] = matches[['home_form', 'away_form']].fillna(0.5)

# Rolling goal difference
matches['goal_diff_home'] = matches['goals_home'] - matches['goals_away']
matches['goal_diff_away'] = -matches['goal_diff_home']

for team in matches['teams_home'].unique():
    home_mask = matches['teams_home'] == team
    away_mask = matches['teams_away'] == team

    matches.loc[home_mask, 'rolling_gd_home'] = matches.loc[home_mask, 'goal_diff_home'].shift(1).rolling(window=5,
                                                                                                          min_periods=1).mean()
    matches.loc[away_mask, 'rolling_gd_away'] = matches.loc[away_mask, 'goal_diff_away'].shift(1).rolling(window=5,
                                                                                                          min_periods=1).mean()

# Team strength (historical win ratio)
team_strength = {}
for team in matches['teams_home'].unique():
    team_matches = matches[(matches['teams_home'] == team) | (matches['teams_away'] == team)]
    wins = ((team_matches['teams_home'] == team) & (team_matches['goals_home'] > team_matches['goals_away'])) | \
           ((team_matches['teams_away'] == team) & (team_matches['goals_away'] > team_matches['goals_home']))
    team_strength[team] = wins.mean()

matches['home_strength'] = matches['teams_home'].map(team_strength)
matches['away_strength'] = matches['teams_away'].map(team_strength)

# Differential contextual features
matches['form_diff'] = matches['home_form'] - matches['away_form']
matches['xg_diff'] = matches['rolling_xg_home'] - matches['rolling_xg_away']
matches['strength_diff'] = matches['home_strength'] - matches['away_strength']
matches['gd_diff'] = matches['rolling_gd_home'] - matches['rolling_gd_away']

print("✅ Step 2.5: Advanced contextual features added successfully!\n")

# ==========================
# STEP 3: FEATURE SET
# ==========================
target_col = 'result'
rolling_cols = [col for col in matches.columns if col.startswith('rolling_')]
context_cols = ['home_form', 'away_form', 'home_strength', 'away_strength', 'form_diff', 'xg_diff', 'strength_diff',
                'gd_diff']
base_features = ['home_team', 'away_team']

X = matches[base_features + rolling_cols + context_cols].copy()
y = matches[target_col].copy()

X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean())

print("✅ Step 3: Feature set constructed successfully!")
print(f"→ Total features: {len(X.columns)}\n")

# Save for backup
matches_final = pd.concat([X, y], axis=1)
matches_final.to_csv("epl_2024_prepared.csv", index=False)

# ==========================
# STEP 4: MODEL TRAINING
# ==========================
print("\n🚀 Step 4: Training Random Forest model...\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    # class_weight="balanced"
    class_weight={0: 1, 1: 2, 2: 1}  # increase weight for draws
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.3f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Away Win", "Draw", "Home Win"]))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Away Win", "Draw", "Home Win"],
            yticklabels=["Away Win", "Draw", "Home Win"])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n🏆 Top 10 Important Features:")
print(importances.head(10))