#!/usr/bin/env python3
"""
clean_chances_fixed.py

Robust CHANCES filler:
- Recognizes '<null>', 'NaN', 'null', empty strings as missing
- Fills 1-missing, 2-missing, all-missing cases per your rules
- Scales down if present values sum > 100
- Logs before/after changed rows
- Writes cleaned CSV
"""

import pandas as pd
import numpy as np
import re
import os

IN_CSV = "/Users/rishimodi/Desktop/Match_Predictor/src/data/raw/epl_2024_matches.csv"
OUT_CSV = "epl_2024_matches_clean.csv"


def parse_percent_cell(x):
    """Return float 0-100 or np.nan for invalid/missing."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "<null>", "-"):
        return np.nan
    # strip percent sign
    if s.endswith("%"):
        s = s[:-1].strip()
    # remove commas and other non-numeric chars except dot/minus
    s = re.sub(r"[^\d\.\-]+", "", s)
    if s == "":
        return np.nan
    try:
        v = float(s)
        if not np.isfinite(v):
            return np.nan
        return v
    except Exception:
        return np.nan


def split_evenly(remaining, count):
    base = remaining / count
    parts = [base for _ in range(count)]
    # fix rounding drift
    diff = remaining - sum(parts)
    parts[-1] += diff
    return parts


def scale_to_100(vals):
    total = sum(vals)
    if total == 0:
        return [0.0 for _ in vals]
    factor = 100.0 / total
    return [v * factor for v in vals]


# --- load ---
if not os.path.exists(IN_CSV):
    raise SystemExit(f"Input CSV not found: {IN_CSV}")

raw = pd.read_csv(IN_CSV, dtype=str)
print(f"Loaded {len(raw)} rows from {IN_CSV}")

# detect CHANCES columns
chance_home_col = next((c for c in raw.columns if c.upper().startswith("CHANCES") and c.upper().endswith("_HOME")),
                       None)
chance_away_col = next((c for c in raw.columns if c.upper().startswith("CHANCES") and c.upper().endswith("_AWAY")),
                       None)
chance_draw_col = next((c for c in raw.columns if c.upper().startswith("CHANCES") and c.upper().endswith("_DRAW")),
                       None)

print("Detected CHANCES columns:", chance_home_col, chance_away_col, chance_draw_col)
if not chance_home_col or not chance_away_col:
    raise SystemExit("Missing CHANCES_home or CHANCES_away in CSV headers.")

# create parsed numeric columns (floats or np.nan)
df = raw.copy()
df["_CH_home"] = df[chance_home_col].apply(parse_percent_cell)
df["_CH_away"] = df[chance_away_col].apply(parse_percent_cell)
# ensure draw column exists in df
if chance_draw_col not in df.columns:
    df[chance_draw_col] = np.nan
df["_CH_draw"] = df[chance_draw_col].apply(parse_percent_cell)

# quick counts before
missing_home = df["_CH_home"].isna().sum()
missing_away = df["_CH_away"].isna().sum()
missing_draw = df["_CH_draw"].isna().sum()
print(f"Missing counts before fix -> home: {missing_home}, away: {missing_away}, draw: {missing_draw}")

changed_rows = []

# process rows deterministically
for i, row in df.iterrows():
    h = row["_CH_home"]
    a = row["_CH_away"]
    d = row["_CH_draw"]
    before = (np.nan if pd.isna(h) else float(h),
              np.nan if pd.isna(a) else float(a),
              np.nan if pd.isna(d) else float(d))

    # use boolean mask via pd.isna
    missing = [pd.isna(h), pd.isna(a), pd.isna(d)]
    mcount = sum(missing)

    if mcount == 0:
        total = (float(h) + float(a) + float(d))
        # if total is approximately 100 do nothing
        if abs(total - 100.0) > 1e-6:
            # scale proportionally
            scaled = scale_to_100([float(h), float(a), float(d)])
            df.at[i, "_CH_home"], df.at[i, "_CH_away"], df.at[i, "_CH_draw"] = scaled
            changed_rows.append((i, before, tuple(scaled), "scaled"))
        # else leave as-is
    elif mcount == 1:
        existing_sum = sum(v for v in (h, a, d) if not pd.isna(v))
        remaining = 100.0 - existing_sum
        if remaining < 0:
            # present values exceed 100: scale present down to sum=100, missing -> 0
            present_idx = [idx for idx, v in enumerate((h, a, d)) if not pd.isna(v)]
            present_vals = [float((h, a, d)[idx]) for idx in present_idx]
            scaled = scale_to_100(present_vals)
            # write scaled back
            p = 0
            for idx_present, val in zip(present_idx, scaled):
                if idx_present == 0:
                    df.at[i, "_CH_home"] = val
                elif idx_present == 1:
                    df.at[i, "_CH_away"] = val
                else:
                    df.at[i, "_CH_draw"] = val
                p += 1
            # set missing to 0
            if missing[0]:
                df.at[i, "_CH_home"] = 0.0
            elif missing[1]:
                df.at[i, "_CH_away"] = 0.0
            else:
                df.at[i, "_CH_draw"] = 0.0
            changed_rows.append(
                (i, before, (df.at[i, "_CH_home"], df.at[i, "_CH_away"], df.at[i, "_CH_draw"]), "scale_overflow"))
        else:
            # fill missing with remaining
            if missing[0]:
                df.at[i, "_CH_home"] = remaining
            elif missing[1]:
                df.at[i, "_CH_away"] = remaining
            else:
                df.at[i, "_CH_draw"] = remaining
            changed_rows.append(
                (i, before, (df.at[i, "_CH_home"], df.at[i, "_CH_away"], df.at[i, "_CH_draw"]), "one_missing_filled"))
    elif mcount == 2:
        present_val = next((float(v) for v in (h, a, d) if not pd.isna(v)), 0.0)
        remaining = 100.0 - present_val
        if remaining < 0:
            # present >100 -> set present to 100, others 0
            if not pd.isna(h):
                df.at[i, "_CH_home"], df.at[i, "_CH_away"], df.at[i, "_CH_draw"] = 100.0, 0.0, 0.0
            elif not pd.isna(a):
                df.at[i, "_CH_home"], df.at[i, "_CH_away"], df.at[i, "_CH_draw"] = 0.0, 100.0, 0.0
            else:
                df.at[i, "_CH_home"], df.at[i, "_CH_away"], df.at[i, "_CH_draw"] = 0.0, 0.0, 100.0
            changed_rows.append(
                (i, before, (df.at[i, "_CH_home"], df.at[i, "_CH_away"], df.at[i, "_CH_draw"]), "two_missing_overflow"))
        else:
            parts = split_evenly(remaining, 2)
            miss_positions = [idx for idx, m in enumerate(missing) if m]
            for pos, val in zip(miss_positions, parts):
                if pos == 0:
                    df.at[i, "_CH_home"] = val
                elif pos == 1:
                    df.at[i, "_CH_away"] = val
                else:
                    df.at[i, "_CH_draw"] = val
            changed_rows.append(
                (i, before, (df.at[i, "_CH_home"], df.at[i, "_CH_away"], df.at[i, "_CH_draw"]), "two_missing_split"))
    else:  # mcount == 3
        df.at[i, "_CH_home"], df.at[i, "_CH_away"], df.at[i, "_CH_draw"] = 33.33, 33.33, 33.34
        changed_rows.append((i, before, (33.33, 33.33, 33.34), "all_missing"))

# Summary
print(f"Total rows changed: {len(changed_rows)}")
for idx, before, after, reason in changed_rows[:30]:
    print(f"Row {idx} | {reason}")
    print(f"  before: home={before[0]}, away={before[1]}, draw={before[2]}")
    print(f"  after : home={after[0]}, away={after[1]}, draw={after[2]}")

# write back numeric columns (rounded)
df[chance_home_col] = df["_CH_home"].round(2)
df[chance_away_col] = df["_CH_away"].round(2)
df[chance_draw_col] = df["_CH_draw"].round(2)

# ensure directory exists
clean_dir = os.path.join(os.path.dirname(__file__), "data", "clean")
os.makedirs(clean_dir, exist_ok=True)

out_path = os.path.join(clean_dir, "epl_2024_matches_clean.csv")

df.drop(columns=["_CH_home", "_CH_away", "_CH_draw"], inplace=True, errors="ignore")
df.to_csv(out_path, index=False)

print(f"💾 Cleaned file saved to {out_path}")
