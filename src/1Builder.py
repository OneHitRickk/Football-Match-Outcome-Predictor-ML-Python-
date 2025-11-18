import requests, json, re, pandas as pd
import os
from bs4 import BeautifulSoup


def get_teams(url, headers):
    """Return dict of {team_name: team_url} for a given season page."""
    # Scraping the main page from where I will scrap teams' URLs
    print("🔍 Fetching team list...")  # todo: remove
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, "html.parser")  # Converting Raw HTML to BeautifulSoup object

    # Fetching the main EPL 2024 page and using headers to avoid bot-blocking
    pattern = re.compile(r"var\s+teamsData\s+=\s+JSON\.parse\('(.*)'\)")

    """Looping over every <script> tag in the page, then we check if the script content matches our regex pattern.
    """
    data = None
    for script in soup.find_all("script"):
        match = pattern.search(script.string or "")
        if match:
            json_data = match.group(1).encode().decode("unicode_escape")
            data = json.loads(json_data)
            break
    if not data:
        raise RuntimeError("teamsData not found.")

    # Extract team names and building URLs, data over here contains all teams and their info.
    teams = {}
    for team_id, team_info in data.items():
        name = team_info.get("title", "").strip()  # Team Name
        slug = name.replace(" ", "_")  # Kept here purposely to avoid breaking of the link.
        teams[name] = f"https://understat.com/team/{slug}/2020"
    print(f"✅ Found {len(teams)} teams.")  # todo: remove
    return teams


def get_team_matches(team_url, headers):
    """Get all matches (JSON) for a single team."""
    print(f"\n⚽ Fetching matches for: {team_url}")  # todo: remove
    team_response = requests.get(team_url, headers=headers)
    team_soup = BeautifulSoup(team_response.text, "html.parser")

    pattern = re.compile(r"var\s+datesData\s*=\s*JSON\.parse\('(.*)'\)", re.DOTALL)
    matches = None
    for script in team_soup.find_all("script"):
        match = pattern.search(script.string or "")
        if match:
            json_data = match.group(1).encode().decode('unicode_escape')
            matches = json.loads(json_data)
            break

    if not matches:
        print(f"⚠️ No matches found for team: {team_url}")  # todo: remove
        return []

    print(f"✅ Found {len(matches)} matches.")  # todo: remove
    return matches


def get_match_stats(match_id, headers):
    """Scrape detailed match stats (xG, shots, PPDA, etc.) from a single match page."""
    match_url = f"https://understat.com/match/{match_id}"
    print(f"\n📊 Accessing match: {match_url}")  # todo: remove

    resp = requests.get(match_url, headers=headers)
    match_soup = BeautifulSoup(resp.text, "html.parser")

    stats_block = match_soup.find("div", {"data-scheme": "stats"})
    if not stats_block:
        print(f"⚠️ Stats block not found for match {match_id}")  # todo: remove
        return None

    match_stats = {"match_id": match_id}

    for bar in stats_block.find_all("div", class_="progress-bar"):
        title_div = bar.find("div", class_="progress-title")
        if not title_div:
            continue
        title = title_div.text.strip().upper()

        home = bar.find("div", class_="progress-home")
        draw = bar.find("div", class_="progress-draw")
        away = bar.find("div", class_="progress-away")

        match_stats[title] = {
            "home": home.text.strip() if home else None,
            "draw": draw.text.strip() if draw else None,
            "away": away.text.strip() if away else None
        }

    print(f"✅ Collected stats for match {match_id}")  # todo: remove
    for key, val in match_stats.items():
        if key != "match_id":
            print(f"   {key}: {val}")  # todo: remove
    return match_stats


if __name__ == "__main__":
    # --- CONFIG ---
    url = "https://understat.com/league/EPL/2024"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/141.0.0.0 Safari/537.36"
    }

    # --- STEP 1: Get teams ---
    teams = get_teams(url, headers)

    # --- STEP 2: Loop over all teams ---
    all_matches_data = []
    seen_matches = set()  # To avoid duplicates

    for team_name, team_url in teams.items():
        print(f"🎯 Scraping matches for: {team_name}")
        matches = get_team_matches(team_url, headers)

        for match in matches:
            match_id = match["id"]
            if match_id in seen_matches:
                continue  # skip duplicate match
            seen_matches.add(match_id)

            stats = get_match_stats(match_id, headers)  # no per-match print inside the function
            if stats:
                all_matches_data.append(stats)

    # --- STEP 3: Convert all match stats to pandas DataFrame (defensive; draw only for CHANCES) ---
    flat_data = []
    for match in all_matches_data:
        flat = {"match_id": match.get("match_id")}
        for metric, vals in match.items():
            if metric == "match_id":
                continue

            # Case A: already flattened keys like 'CHANCES_home' / 'XG_away' etc.
            if isinstance(metric, str) and (
                    metric.endswith("_home") or metric.endswith("_away") or metric.endswith("_draw")):
                flat[metric] = vals
                continue

            # Case B: grouped dict e.g., "CHANCES": {"home":..., "draw":..., "away":...}
            if isinstance(vals, dict):
                home = vals.get("home") if vals.get("home") is not None else None
                away = vals.get("away") if vals.get("away") is not None else None
                draw = vals.get("draw") if vals.get("draw") is not None else None

                flat[f"{metric}_home"] = home
                flat[f"{metric}_away"] = away
                # Only keep draw for CHANCES
                if str(metric).upper() == "CHANCES":
                    flat[f"{metric}_draw"] = draw
                continue

            # Case C: scalar or unexpected format -> copy as-is
            flat[metric] = vals

        flat_data.append(flat)

    # Build DataFrame and sort for neatness
    df = pd.DataFrame(flat_data)
    if "match_id" in df.columns:
        try:
            df = df.sort_values("match_id").reset_index(drop=True)
        except Exception:
            pass

    # OLD PRINTER

    # # --- STEP 4: Save to CSV ---
    # df.to_csv("epl_2023_matches.csv", index=False)
    # print(f"\n✅ Finished scraping {len(all_matches_data)} unique matches across all teams.")
    # print("💾 Saved DataFrame to epl_2023_matches.csv")



    # --- STEP 4: Save to CSV (robust path handling) ---
    raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")  # folder relative to this script
    os.makedirs(raw_dir, exist_ok=True)  # create if missing (no error if exists)

    out_path = os.path.join(raw_dir, "epl_2020_matches.csv")
    df.to_csv(out_path, index=False)

    print(f"\n✅ Finished scraping {len(all_matches_data)} unique matches across all teams.")
    print(f"💾 Saved DataFrame to {out_path}")
