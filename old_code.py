import requests, json, re, pandas as pd
from bs4 import BeautifulSoup


# Scraping the main page from where I will scrap teams' URLs
response = requests.get(url, headers=headers)
print("Status Code:", response.status_code)  # Just kept to check if the request was successful

# Converting Raw HTML to BeautifulSoup object
soup = BeautifulSoup(response.text, "html.parser")

# Extracting All <script> Tags
scripted = soup.find_all("script")

# Fetching the main EPL 2024 page and using headers to avoid bot-blocking
pattern = re.compile(r"var\s+teamsData\s+=\s+JSON\.parse\('(.*)'\)")

"""
Looping over every <script> tag in the page, then we check if the script content matches our regex pattern.
"""
data = None
for script in scripted:
    match = pattern.search(script.string or "")
    if match:
        json_data = match.group(1)
        json_data = json_data.encode().decode('unicode_escape')
        data = json.loads(json_data)
        break

if not data:
    raise RuntimeError("teamsData JSON not found on the page")

# Extract team names and building URLs, data over here contains all teams and their info.
teams = {}
for team_id, team_info in data.items():
    name = team_info.get("title", "").strip()  # Team Name
    slug = name.replace(" ", "_")  # Kept here purposely to avoid breaking of the link.
    team_url = f"https://understat.com/team/{slug}/2024"
    teams[name] = team_url
# print(teams)
# for team_name, team_url in teams.items():
#     print(f"{team_name} --> {team_url}")

# Now till here we have a dictionary of teams and their URLs ready to be used.


################################################################################################


# Pick one team and first match
test_team_url = list(teams.values())[0]  # e.g., Aston Villa
team_response = requests.get(test_team_url, headers=headers)
team_soup = BeautifulSoup(team_response.text, "html.parser")

# Extract matches JSON
scripts = team_soup.find_all("script")
pattern = re.compile(r"var\s+datesData\s*=\s*JSON\.parse\('(.*)'\)", re.DOTALL)

matches = None
for script in scripts:
    match = pattern.search(script.string or "")
    if match:
        json_data = match.group(1).encode().decode('unicode_escape')
        matches = json.loads(json_data)
        break

if not matches:
    raise RuntimeError("No matches found for this team.")

# Pick the first match
first_match = matches[0]
match_id = first_match["id"]
match_url = f"https://understat.com/match/{match_id}"
print(f"\n🔍 Accessing match page: {match_url}")

# Fetch the match page
match_response = requests.get(match_url, headers=headers)
match_soup = BeautifulSoup(match_response.text, "html.parser")

# Find the stats block
stats_block = match_soup.find("div", {"data-scheme": "stats"})
if not stats_block:
    raise RuntimeError("Stats block not found on match page.")

# Extract all progress-bar metrics
match_stats = {}
for bar in stats_block.find_all("div", class_="progress-bar"):
    title_div = bar.find("div", class_="progress-title")
    if not title_div:
        continue
    title = title_div.text.strip()

    home = bar.find("div", class_="progress-home")
    draw = bar.find("div", class_="progress-draw")
    away = bar.find("div", class_="progress-away")

    match_stats[title] = {
        "home": home.text.strip() if home else None,
        "draw": draw.text.strip() if draw else None,
        "away": away.text.strip() if away else None
    }

# Show extracted stats
for metric, values in match_stats.items():
    print(f"{metric}: {values}")

# #####################################################################################