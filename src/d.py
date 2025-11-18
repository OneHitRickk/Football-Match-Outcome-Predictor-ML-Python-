# src/merge_all_seasons.py
"""
Merge multiple season clean CSVs (epl_YYYY_clean.csv) into one deduped file.

Saves:
 - src/data/clean/epl_all_clean.csv        # merged & deduped
 - src/data/clean/merge_dedupe_report.txt  # short summary
 - src/data/clean/duplicates_removed_verbose.csv  # verbose list of removed duplicate rows for audit
"""
import pandas as pd
import os
import re
from pathlib import Path

CLEAN_DIR = Path("/Users/rishimodi/Desktop/Match_Predictor/src/data/clean")
OUT_MERGED = CLEAN_DIR / "/Users/rishimodi/Desktop/Match_Predictor/src/data/clean/epl_all_seasons_clean.csv"
OUT_REPORT = CLEAN_DIR / "merge_dedupe_report.txt"
OUT_REMOVED = CLEAN_DIR / "duplicates_removed_verbose.csv"

# 1) gather season files
pattern = re.compile(r"epl[_\-]?(\d{4})(?:[_\-]?matches)?[_\-]?clean\.csv", re.IGNORECASE)
files = []
for p in sorted(CLEAN_DIR.glob("*.csv")):
    m = pattern.search(p.name)
    if m:
        season = int(m.group(1))
        files.append((season, p))

if not files:
    raise SystemExit(f"No epl_*_clean.csv files found in {CLEAN_DIR.resolve()}")

print("Found season files:")
for season, p in files:
    print(f"  {season}: {p}")

# 2) load and add season column
dfs = []
for season, p in files:
    df = pd.read_csv(p, dtype=str)  # keep strings to avoid dtype surprises
    df["__source_file"] = p.name
    df["season"] = season
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print(f"\nLoaded total rows: {len(combined)} from {len(files)} files")

# 3) drop exact duplicate rows (identical across all cols)
before_exact = len(combined)
combined = combined.drop_duplicates(keep="first").reset_index(drop=True)
exact_dropped = before_exact - len(combined)
print(f"Dropped exact-identical duplicate rows: {exact_dropped}")

# 4) ensure match_id exists and normalize it to str (strip whitespace)
if "match_id" not in combined.columns:
    raise SystemExit("ERROR: 'match_id' column NOT found in combined files; cannot dedupe by match_id.")
combined["match_id"] = combined["match_id"].astype(str).str.strip()

# 5) identify duplicate match_id groups
dupe_mask = combined["match_id"].duplicated(keep=False)
dupe_ids = combined.loc[dupe_mask, "match_id"].unique().tolist()
print(f"Distinct match_id values that have >1 row: {len(dupe_ids)}")

# 6) choose columns to evaluate completeness (adjust if your column names differ)
candidate_stat_cols = [
    "CHANCES_home","CHANCES_away","CHANCES_draw",
    "GOALS_home","GOALS_away",
    "XG_home","XG_away",
    "SHOTS_home","SHOTS_away",
    "SHOTS ON TARGET_home","SHOTS ON TARGET_away",
    "DEEP_home","DEEP_away",
    "PPDA_home","PPDA_away",
    "XPTS_home","XPTS_away",
    # fallback names (lowercase variants)
]
# include any variant in combined.columns (case-sensitive)
stat_cols = [c for c in candidate_stat_cols if c in combined.columns]
# also accept lowercase variants
lower_map = {c.lower(): c for c in combined.columns}
for cand in candidate_stat_cols:
    lc = cand.lower()
    if lc in lower_map and cand not in stat_cols:
        stat_cols.append(lower_map[lc])

print("Completeness checked over stat columns (found):", stat_cols)

# 7) compute completeness score and for each match_id keep the best row
if dupe_ids:
    # compute completeness = count of non-null among stat_cols
    def completeness_score(row):
        if not stat_cols:
            return 0
        return row[stat_cols].notna().sum()

    combined["_completeness"] = combined.apply(lambda r: completeness_score(r), axis=1)

    # sort so highest completeness per match_id comes first
    combined = combined.sort_values(["match_id", "_completeness"], ascending=[True, False]).reset_index(drop=True)

    # For audit: collect removed rows
    removed_rows = []

    # iterate grouped and keep first row (best completeness), mark others as removed
    keep_rows = []
    grouped = combined.groupby("match_id")
    keep_indices = []
    remove_indices = []
    for mid, group in grouped:
        if len(group) == 1:
            keep_indices.append(group.index[0])
            continue
        # group is already sorted so first is best completeness
        gidx = group.index.tolist()
        keep_indices.append(gidx[0])
        remove_indices.extend(gidx[1:])

    # build kept and removed df
    kept = combined.loc[keep_indices].sort_index().reset_index(drop=True)
    removed = combined.loc[remove_indices].reset_index(drop=True)

    # clean up temp column from kept
    if "_completeness" in kept.columns:
        kept = kept.drop(columns=["_completeness"], errors="ignore")
    # removed keep completeness for audit
    if "_completeness" in removed.columns:
        pass  # keep for debugging

    # replace combined with kept
    combined = kept

    print(f"Resolved duplicate match_id groups. Kept {len(keep_indices)} rows, removed {len(remove_indices)} rows.")
else:
    removed = pd.DataFrame(columns=combined.columns)
    print("No duplicate match_id groups found after exact dedupe.")

# 8) final sanity checks
final_total = len(combined)
final_unique_ids = combined["match_id"].nunique()
print(f"Final merged rows: {final_total}, unique match_id: {final_unique_ids}")

# 9) write merged file and audit files
combined.to_csv(OUT_MERGED, index=False)
print("Saved merged file:", OUT_MERGED)

# write verbose removed rows (if any) for your inspection
if not removed.empty:
    removed.to_csv(OUT_REMOVED, index=False)
    print("Saved removed-duplicates verbose CSV:", OUT_REMOVED)
else:
    # ensure previous file removed if exists
    if OUT_REMOVED.exists():
        OUT_REMOVED.unlink()
    print("No removed duplicates to write.")

# 10) write short report
lines = []
lines.append(f"Files merged ({len(files)}): " + ", ".join(p.name for _, p in files))
lines.append(f"Rows loaded (sum of files): {sum(len(pd.read_csv(p, dtype=str)) for _, p in files)}")
lines.append(f"Rows after concat & exact-dedupe: {len(pd.concat(dfs, ignore_index=True).drop_duplicates())}")
lines.append(f"Exact duplicate rows removed: {exact_dropped}")
lines.append(f"Distinct match_id groups found duplicated: {len(dupe_ids)}")
lines.append(f"Rows removed due to match_id duplicate resolving: {len(removed)}")
lines.append(f"Final rows written: {final_total}")
lines.append(f"Final unique match_id: {final_unique_ids}")
lines.append("")
if stat_cols:
    lines.append("Stat columns used for completeness scoring:")
    lines += stat_cols
else:
    lines.append("No stat columns detected to score completeness (verify your column names).")

with open(OUT_REPORT, "w") as f:
    f.write("\n".join(lines))

print("Saved report:", OUT_REPORT)
print("\nDone.")