import requests, json, re
from bs4 import BeautifulSoup
import pandas as pd

url = "https://understat.com/league/EPL/2024"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# --- Extract team data ---
pattern = re.compile(r"var\s+teamsData\s+=\s+JSON\.parse\('(.*)'\)")
data = None
for script in soup.find_all("script"):
    match = pattern.search(script.string or "")
    if match:
        json_data = match.group(1).encode().decode('unicode_escape')
        data = json.loads(json_data)
        break

if not data:
    raise RuntimeError("teamsData JSON not found on the page")

teams = {}
for team_id, team_info in data.items():
    name = team_info.get("title", "").strip()
    slug = name.replace(" ", "_")
    team_url = f"https://understat.com/team/{slug}/2024"
    teams[name] = team_url

# --- Pick one team to start ---
test_team_url = list(teams.values())[0]
print(f"\n🔍 Fetching matches for team: {test_team_url}")

team_response = requests.get(test_team_url, headers=headers)
team_soup = BeautifulSoup(team_response.text, "html.parser")

# --- Extract matches JSON ---
pattern = re.compile(r"var\s+datesData\s*=\s*JSON\.parse\('(.*)'\)", re.DOTALL)
matches = None
for script in team_soup.find_all("script"):
    match = pattern.search(script.string or "")
    if match:
        json_data = match.group(1).encode().decode('unicode_escape')
        matches = json.loads(json_data)
        break

if not matches:
    raise RuntimeError("No matches found for this team.")
print(f"✅ Found {len(matches)} matches.")

# --- Loop over all matches ---
all_match_data = []

for match in matches:
    match_id = match["id"]
    match_url = f"https://understat.com/match/{match_id}"
    print(f"\n⚽ Accessing match: {match_url}")

    resp = requests.get(match_url, headers=headers)
    match_soup = BeautifulSoup(resp.text, "html.parser")

    stats_block = match_soup.find("div", {"data-scheme": "stats"})
    if not stats_block:
        print(f"⚠️ Stats block not found for {match_url}")
        continue

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

    all_match_data.append(match_stats)

print(f"\n✅ Collected data for {len(all_match_data)} matches.")

# --- Convert to DataFrame (optional) ---
flat_data = []
for match in all_match_data:
    flat = {"match_id": match["match_id"]}
    for metric, vals in match.items():
        if metric == "match_id":
            continue
        flat[f"{metric}_home"] = vals["home"]
        flat[f"{metric}_away"] = vals["away"]
    flat_data.append(flat)

df = pd.DataFrame(flat_data)
print("\n📊 Sample DataFrame:")
print(df.head())


# ------------------------------------------------------------
# 🧩 Step: Convert all collected match stats into a DataFrame
# ------------------------------------------------------------
import pandas as pd

# Collect all match data into a list
all_match_data = []

for match in matches:
    match_id = match["id"]
    match_url = f"https://understat.com/match/{match_id}"
    print(f"\n⚽ Accessing match: {match_url}")

    resp = requests.get(match_url, headers=headers)
    match_soup = BeautifulSoup(resp.text, "html.parser")

    stats_block = match_soup.find("div", {"data-scheme": "stats"})
    if not stats_block:
        print(f"⚠️ Stats block not found for {match_url}")
        continue

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

    all_match_data.append(match_stats)

# Flatten nested structure for DataFrame
flat_data = []
for match in all_match_data:
    flat = {"match_id": match["match_id"]}
    for metric, vals in match.items():
        if metric == "match_id":
            continue
        flat[f"{metric}_home"] = vals["home"]
        flat[f"{metric}_away"] = vals["away"]
    flat_data.append(flat)

df = pd.DataFrame(flat_data)

# Show sample
print("\n📊 Sample DataFrame:")
print(df.head())

# Optionally, save to CSV
df.to_csv("team_match_stats.csv", index=False)
print("\n💾 Saved to team_match_stats.csv")
