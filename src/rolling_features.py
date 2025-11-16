# step2_rolls_and_context.py (UPDATED: infer draw prob fallback)
import pandas as pd
import numpy as np
import os

# ---------- CONFIG ----------
IN_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/encoded/epl_2024_matches_encoded.csv"   # change if your file is named differently
OUT_CSV = "epl_2024_matches_with_rolls.csv"
ROLL_WINDOW = 5
MIN_PERIODS = 1   # we'll compute counts and use shrinkage, so min_periods=1 is fine
SHRINKAGE_K = 5.0  # pseudo-count for Bayesian shrinkage; tune if needed
IMPUTE_STRATEGY = "median"  # used to fill any final NaNs (note: in final training compute median ON TRAIN ONLY)

# ---------- HELPERS ----------
def find_col(df, base_names, side):
    """
    Try to find a column for a stat. base_names is list of possible bases like ['shots on target','shots_on_target','shotsontarget'].
    side is 'home' or 'away' and we look for f"{base}_{side}" or f"{base}{sep}{side}" variants.
    """
    for base in base_names:
        candidates = [
            f"{base}_{side}",
            f"{base}{' '}{side}",
            f"{base}{side}",
            f"{base}{'-'}{side}"
        ]
        for c in candidates:
            if c in df.columns:
                return c
    return None

# ---------- LOAD ----------
if not os.path.exists(IN_CSV):
    raise SystemExit(f"Input CSV not found: {IN_CSV}")

df = pd.read_csv(IN_CSV, dtype=str)  # read as strings to avoid surprises
# normalize column names
df.columns = df.columns.str.strip().str.lower()

# attempt to coerce match_id numeric and team ids numeric
if "match_id" in df.columns:
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce")
else:
    raise SystemExit("match_id column required")

# prefer these team id columns if present; fallback to home_team/away_team strings (but model needs numeric ids)
if "home_team_id" not in df.columns or "away_team_id" not in df.columns:
    raise SystemExit("home_team_id and away_team_id columns required (run encoding step first)")

df["home_team_id"] = pd.to_numeric(df["home_team_id"], errors="coerce").astype("Int64")
df["away_team_id"] = pd.to_numeric(df["away_team_id"], errors="coerce").astype("Int64")

# coercion to numeric for stat columns is done later; keep original df saved copy
orig_df = df.copy()

# ---------- DEFINE STAT BASES & detect columns ----------
bases = {
    "xg": ["xg"],
    "goals": ["goals"],
    "shots": ["shots"],
    "shots_on_target": ["shots_on_target", "shots on target", "shotsontarget"],
    "deep": ["deep"],
    "ppda": ["ppda"],
    "xpts": ["xpts"],
    "chance_prob": ["chance_home_prob", "chance_home", "chances_home", "chance_home_pct", "chances_home_pct"],
    "chance_draw_prob": ["chance_draw_prob", "chances_draw", "chance_draw", "chances_draw_pct", "chance_draw_pct"]
}

# Build mapping for home/away columns
stat_cols = {}
for stat, aliases in bases.items():
    home_col = None
    away_col = None
    # Try obvious patterns
    for a in aliases:
        if a.endswith("_home") and a in df.columns:
            home_col = a
            away_col = a.replace("_home", "_away")
            break
    if not home_col:
        home_col = find_col(df, aliases, "home")
        away_col = find_col(df, aliases, "away")
    stat_cols[stat] = (home_col, away_col)

# For convenience, ensure goals exist
if stat_cols["goals"][0] is None or stat_cols["goals"][1] is None:
    if "goals_home" in df.columns and "goals_away" in df.columns:
        stat_cols["goals"] = ("goals_home", "goals_away")
    else:
        raise SystemExit("goals_home / goals_away columns required")

print("Detected stat columns (home, away):")
for k,v in stat_cols.items():
    print(f"  {k}: {v}")

# ---------- Convert found columns to numeric floats ----------
for stat,(home,away) in stat_cols.items():
    if home and home in df.columns:
        df[home] = pd.to_numeric(df[home].str.replace("%","", regex=False) if df[home].dtype==object else df[home], errors='coerce')
    if away and away in df.columns:
        df[away] = pd.to_numeric(df[away].str.replace("%","", regex=False) if df[away].dtype==object else df[away], errors='coerce')

# ---------- If draw chance wasn't found, try to infer it ----------
# If chance_draw_prob detection failed (None, None) but chance_home/away exist, infer draw = 1 - home - away
if (stat_cols["chance_draw_prob"][0] is None or stat_cols["chance_draw_prob"][1] is None):
    ch_home_col, ch_away_col = stat_cols["chance_prob"]
    if ch_home_col and ch_away_col and ch_home_col in df.columns and ch_away_col in df.columns:
        # ensure they are probabilities (0..1). If values appear >1, assume percent and convert
        try:
            max_home = pd.to_numeric(df[ch_home_col], errors='coerce').dropna().astype(float).max()
        except:
            max_home = None
        if max_home is not None and max_home > 1.5:
            df[ch_home_col] = df[ch_home_col].astype(float) / 100.0
        try:
            max_away = pd.to_numeric(df[ch_away_col], errors='coerce').dropna().astype(float).max()
        except:
            max_away = None
        if max_away is not None and max_away > 1.5:
            df[ch_away_col] = df[ch_away_col].astype(float) / 100.0

        inferred_col = "chance_draw_prob_inferred"
        # compute inference (clip to 0..1)
        df[inferred_col] = 1.0 - df[ch_home_col].astype(float) - df[ch_away_col].astype(float)
        df[inferred_col] = df[inferred_col].clip(lower=0.0, upper=1.0)
        # update stat_cols to use inferred draw column for both sides (match-level)
        stat_cols["chance_draw_prob"] = (inferred_col, inferred_col)
        print(f"[INFO] Inferred draw column created: {inferred_col}")
    else:
        print("[WARN] Could not find chance_draw columns nor infer from home/away — draw-rolls will be NaN.")

# ---------- Build long-format team-match rows ----------
rows = []
for _, r in df.iterrows():
    mid = r["match_id"]
    home_id = int(r["home_team_id"]) if pd.notna(r["home_team_id"]) else None
    away_id = int(r["away_team_id"]) if pd.notna(r["away_team_id"]) else None

    home_stats = {}
    away_stats = {}
    for stat,(home_col,away_col) in stat_cols.items():
        home_stats[stat] = float(r[home_col]) if (home_col and pd.notna(r.get(home_col))) else np.nan
        away_stats[stat] = float(r[away_col]) if (away_col and pd.notna(r.get(away_col))) else np.nan

    gh = home_stats.get("goals", np.nan)
    ga = away_stats.get("goals", np.nan)
    home_win = 1 if (pd.notna(gh) and pd.notna(ga) and gh > ga) else 0
    away_win = 1 if (pd.notna(gh) and pd.notna(ga) and ga > gh) else 0
    draw_flag = 1 if (pd.notna(gh) and pd.notna(ga) and gh == ga) else 0

    rows.append({
        "match_id": mid,
        "team_id": home_id,
        "is_home": 1,
        "opp_id": away_id,
        **home_stats,
        "win_flag": home_win,
        "draw_flag": draw_flag
    })
    rows.append({
        "match_id": mid,
        "team_id": away_id,
        "is_home": 0,
        "opp_id": home_id,
        **away_stats,
        "win_flag": away_win,
        "draw_flag": draw_flag
    })

team_df = pd.DataFrame(rows)
team_df_cols = [c for c in team_df.columns if c not in ("match_id","team_id","is_home","opp_id")]
team_df[team_df_cols] = team_df[team_df_cols].apply(pd.to_numeric, errors="coerce")

# ---------- sort by team/time ----------
team_df = team_df.sort_values(["team_id","match_id"]).reset_index(drop=True)

# ---------- compute rolling means, counts, and win/draw rates (shifted) ----------
roll_stats = list(stat_cols.keys())

for stat in roll_stats:
    mean_col = f"{stat}_roll{ROLL_WINDOW}"
    count_col = f"{stat}_n{ROLL_WINDOW}"
    series_mean = team_df.groupby("team_id")[stat].apply(lambda s: s.shift(1).rolling(window=ROLL_WINDOW, min_periods=MIN_PERIODS).mean()).reset_index(level=0, drop=True)
    series_count = team_df.groupby("team_id")[stat].apply(lambda s: s.shift(1).rolling(window=ROLL_WINDOW, min_periods=MIN_PERIODS).count()).reset_index(level=0, drop=True)

    team_df[mean_col] = series_mean
    team_df[count_col] = series_count

team_df[f"win_rate_roll{ROLL_WINDOW}"] = team_df.groupby("team_id")["win_flag"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean()).reset_index(level=0, drop=True)
team_df[f"draw_rate_roll{ROLL_WINDOW}"] = team_df.groupby("team_id")["draw_flag"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean()).reset_index(level=0, drop=True)
team_df[f"matches_played_roll{ROLL_WINDOW}"] = team_df.groupby("team_id")["match_id"].apply(lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).count()).reset_index(level=0, drop=True)

# ---------- compute league priors (mu) for shrinkage ----------
league_mu = {}
for stat in roll_stats + ["win_rate"]:
    col = f"{stat}_roll{ROLL_WINDOW}" if stat in roll_stats else f"win_rate_roll{ROLL_WINDOW}"
    if col in team_df.columns:
        m = team_df[col].median(skipna=True)
        # defensive fallback if median is NaN
        league_mu[stat] = 0.0 if pd.isna(m) else m
    else:
        league_mu[stat] = 0.0

print("\nLeague priors (medians used as prior mu) for shrinkage:")
for k,v in league_mu.items():
    print(f"  {k}: {v}")

# ---------- apply Bayesian shrinkage to each rolling stat ----------
def apply_shrinkage(mean_col, count_col, mu, k):
    adj_col = mean_col.replace("_roll"+str(ROLL_WINDOW), f"_shr{ROLL_WINDOW}")
    team_df[adj_col] = team_df[mean_col]  # default copy
    mask = team_df[count_col].notna()
    n = team_df.loc[mask, count_col].astype(float)
    m = team_df.loc[mask, mean_col]
    team_df.loc[mask, adj_col] = (n * m + k * mu) / (n + k)
    team_df.loc[~mask, adj_col] = mu
    return adj_col

shrunk_cols = []
for stat in roll_stats:
    mean_col = f"{stat}_roll{ROLL_WINDOW}"
    count_col = f"{stat}_n{ROLL_WINDOW}"
    mu = league_mu.get(stat, 0.0)
    adj = apply_shrinkage(mean_col, count_col, mu, SHRINKAGE_K)
    shrunk_cols.append(adj)

# also shrink win_rate
mean_col = f"win_rate_roll{ROLL_WINDOW}"
mu_win = league_mu.get("win_rate", 0.0)
team_df[f"win_rate_shr{ROLL_WINDOW}"] = team_df[mean_col]
mask = team_df[f"matches_played_roll{ROLL_WINDOW}"].notna()
n = team_df.loc[mask, f"matches_played_roll{ROLL_WINDOW}"].astype(float)
m = team_df.loc[mask, mean_col]
team_df.loc[mask, f"win_rate_shr{ROLL_WINDOW}"] = (n * m + SHRINKAGE_K * mu_win) / (n + SHRINKAGE_K)
team_df.loc[~mask, f"win_rate_shr{ROLL_WINDOW}"] = mu_win

# ---------- map back to match-level (home/away) ----------
home_df = team_df[team_df["is_home"] == 1].set_index("match_id")
away_df = team_df[team_df["is_home"] == 0].set_index("match_id")

out = orig_df.copy()

# map shrunk features for each stat
for stat in roll_stats:
    shr_col = f"{stat}_shr{ROLL_WINDOW}"
    home_col_name = f"home_{stat}_shr{ROLL_WINDOW}"
    away_col_name = f"away_{stat}_shr{ROLL_WINDOW}"
    out[home_col_name] = out["match_id"].map(home_df[shr_col])
    out[away_col_name] = out["match_id"].map(away_df[shr_col])

# win rate
out[f"home_win_rate_shr{ROLL_WINDOW}"] = out["match_id"].map(home_df[f"win_rate_shr{ROLL_WINDOW}"])
out[f"away_win_rate_shr{ROLL_WINDOW}"] = out["match_id"].map(away_df[f"win_rate_shr{ROLL_WINDOW}"])

# matches played
out[f"home_matches_played_roll{ROLL_WINDOW}"] = out["match_id"].map(home_df[f"matches_played_roll{ROLL_WINDOW}"])
out[f"away_matches_played_roll{ROLL_WINDOW}"] = out["match_id"].map(away_df[f"matches_played_roll{ROLL_WINDOW}"])

# ---------- derived match-level diffs ----------
if f"home_xg_shr{ROLL_WINDOW}" in out.columns and f"away_xg_shr{ROLL_WINDOW}" in out.columns:
    out[f"xg_diff_shr{ROLL_WINDOW}"] = out[f"home_xg_shr{ROLL_WINDOW}"] - out[f"away_xg_shr{ROLL_WINDOW}"]
if f"home_chance_prob_shr{ROLL_WINDOW}" in out.columns and f"away_chance_prob_shr{ROLL_WINDOW}" in out.columns:
    out[f"chance_prob_diff_shr{ROLL_WINDOW}"] = out[f"home_chance_prob_shr{ROLL_WINDOW}"] - out[f"away_chance_prob_shr{ROLL_WINDOW}"]

out[f"form_diff_shr{ROLL_WINDOW}"] = out[f"home_win_rate_shr{ROLL_WINDOW}"] - out[f"away_win_rate_shr{ROLL_WINDOW}"]

# ---------- final imputation for any remaining NaNs ----------
roll_out_cols = [c for c in out.columns if c.endswith(f"_shr{ROLL_WINDOW}") or f"matches_played_roll{ROLL_WINDOW}" in c or c.endswith(f"win_rate_shr{ROLL_WINDOW}")]

if IMPUTE_STRATEGY == "median":
    for c in roll_out_cols:
        med = out[c].median(skipna=True)
        out[c] = out[c].fillna(med)
elif IMPUTE_STRATEGY == "zero":
    out[roll_out_cols] = out[roll_out_cols].fillna(0.0)
else:
    pass

import os

# ---------- save (write to data/rolls/) ----------
# Ensure output directory exists relative to this script
rolls_dir = os.path.join(os.path.dirname(__file__), "data", "rolls")
os.makedirs(rolls_dir, exist_ok=True)
out_path = os.path.join(rolls_dir, OUT_CSV)

# write
out.to_csv(out_path, index=False)

# friendly logging
print(f"\n✅ Saved rolling + context features to {out_path}")
print("Sample new columns (first 6 rows):")
new_cols_preview = [col for col in out.columns if col.endswith(f"_shr{ROLL_WINDOW}")]
print(out[new_cols_preview[:8]].head(6).to_string(index=False))
