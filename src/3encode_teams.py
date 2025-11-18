# step1_encode_teams_fixed.py
import pandas as pd
import json
import os
from sklearn.preprocessing import LabelEncoder

IN_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/clean/epl_all_seasons_clean.csv"         # your cleaned CSV
OUT_CSV = "epl_all_season_matches_encoded.csv"      # result with numeric team ids
ENC_JSON = "team_label_encoder_map.json"      # saved mapping

# ---- Load + normalize column names ----
if not os.path.exists(IN_CSV):
    raise SystemExit(f"Input CSV not found: {IN_CSV}")

# read as strings to avoid surprises, we'll coerce numeric later
df = pd.read_csv(IN_CSV, dtype=str)
# keep original column names mapping for robust matching
orig_cols = list(df.columns)
# normalize
df.columns = df.columns.str.strip().str.lower()

# ---- Ensure team columns exist ----
# Common names we expect: teams_home, teams_away (from your scraper)
if "teams_home" not in df.columns or "teams_away" not in df.columns:
    # try a couple of fallbacks
    possible_home = [c for c in df.columns if "team" in c and "home" in c]
    possible_away = [c for c in df.columns if "team" in c and "away" in c]
    if possible_home and "teams_home" not in df.columns:
        df["teams_home"] = df[possible_home[0]]
    if possible_away and "teams_away" not in df.columns:
        df["teams_away"] = df[possible_away[0]]

if "teams_home" not in df.columns or "teams_away" not in df.columns:
    raise SystemExit("Could not find teams_home / teams_away columns. Check your CSV headers: \n" + ", ".join(orig_cols))

# Fill any missing names defensively
df["teams_home"] = df["teams_home"].fillna("unknown_home")
df["teams_away"] = df["teams_away"].fillna("unknown_away")

# ---- Fit LabelEncoder on union of team names ----
le = LabelEncoder()
all_teams = pd.concat([df["teams_home"], df["teams_away"]]).astype(str).unique()
le.fit(all_teams)

# map to numeric ids (0..n-1)
df["home_team_id"] = le.transform(df["teams_home"].astype(str)).astype(int)
df["away_team_id"] = le.transform(df["teams_away"].astype(str)).astype(int)

# ---- Save mapping for reproducibility ----
mapping = {
    "classes": list(le.classes_),
    "id_to_team": {int(i): name for i, name in enumerate(le.classes_)},
    "team_to_id": {name: int(i) for i, name in enumerate(le.classes_)}
}
with open(ENC_JSON, "w") as f:
    json.dump(mapping, f, indent=2)

# ---- Ensure match_id numeric and sort (chronological proxy) ----
if "match_id" in df.columns:
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce")
    df = df.sort_values("match_id").reset_index(drop=True)

# ---- Compute result (0 away, 1 draw, 2 home) if not present ----
# Try several common goal column name patterns and fuzzy matches
if "result" not in df.columns:
    def compute_result_from_df(dframe):
        cols = [c.lower() for c in dframe.columns]
        # exact common patterns
        candidates = [
            ("goals_home","goals_away"),
            ("home_goals","away_goals"),
            ("score_home","score_away"),
            ("home_score","away_score"),
            ("fthg","ftag")  # some datasets use these
        ]
        for a,b in candidates:
            if a in cols and b in cols:
                a_col = [c for c in dframe.columns if c.lower()==a][0]
                b_col = [c for c in dframe.columns if c.lower()==b][0]
                gh = pd.to_numeric(dframe[a_col], errors="coerce")
                ga = pd.to_numeric(dframe[b_col], errors="coerce")
                res = pd.Series(index=dframe.index, dtype=int)
                res[(gh > ga)] = 2
                res[(gh == ga)] = 1
                res[(gh < ga)] = 0
                return res
        # fuzzy: find any pair where one column contains 'home' & 'goal' and other 'away' & 'goal'
        home_candidates = [c for c in dframe.columns if ("home" in c.lower() or c.lower().endswith("_h")) and "goal" in c.lower()]
        away_candidates = [c for c in dframe.columns if ("away" in c.lower() or c.lower().endswith("_a")) and "goal" in c.lower()]
        if home_candidates and away_candidates:
            a_col = home_candidates[0]
            b_col = away_candidates[0]
            gh = pd.to_numeric(dframe[a_col], errors="coerce")
            ga = pd.to_numeric(dframe[b_col], errors="coerce")
            res = pd.Series(index=dframe.index, dtype=int)
            res[(gh > ga)] = 2
            res[(gh == ga)] = 1
            res[(gh < ga)] = 0
            return res
        return None

    result_series = compute_result_from_df(df)
    if result_series is None:
        print("WARNING: Could not automatically compute 'result' column from goals. 'result' will be missing in the output CSV.")
    else:
        df["result"] = result_series
else:
    # already present - try to coerce to numeric and ensure mapping 0/1/2
    try:
        df["result"] = pd.to_numeric(df["result"], errors="coerce").astype(pd.Int64Dtype())
    except Exception:
        pass

# ---- Save encoded CSV ----
# --- Ensure encoded directory exists ---
encoded_dir = os.path.join(os.path.dirname(__file__), "data", "encoded")
os.makedirs(encoded_dir, exist_ok=True)

# --- Output file paths ---
out_encoded_csv = os.path.join(encoded_dir, "epl_all_seasons_encoded.csv")
out_encoder_json = os.path.join(encoded_dir, "team_label_encoder_map.json")

# Save CSV
df.to_csv(out_encoded_csv, index=False)

# Save JSON
with open(out_encoder_json, "w") as f:
    json.dump(mapping, f, indent=2)

print("✅ Team encoding complete.")
print(f"💾 Encoded CSV saved to: {out_encoded_csv}")
print(f"💾 Encoder map saved to: {out_encoder_json}")