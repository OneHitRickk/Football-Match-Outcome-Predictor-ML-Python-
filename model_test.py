import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

matches = pd.read_csv('epl_2024_matches.csv')
matches.columns = matches.columns.str.strip().str.lower()

le_team = LabelEncoder()
all_teams = pd.concat([matches['teams_home'], matches['teams_away']]).unique()
le_team.fit(all_teams)
matches['home_team'] = le_team.transform(matches['teams_home'])
matches['away_team'] = le_team.transform(matches['teams_away'])

matches['target'] = (matches['goals_home'].fillna(0).astype(int) >
                     matches['goals_away'].fillna(0).astype(int)).astype(int)

predictors = [
    'home_team', 'away_team',
    'xg_home', 'xg_away',
    'deep_home', 'deep_away',
    'ppda_home', 'ppda_away',
    'xpts_home', 'xpts_away'
]
predictors = [p for p in predictors if p in matches.columns]

X = matches[predictors]
y = matches['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions) * 100
print(f"Accuracy: {accuracy:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, predictions))
