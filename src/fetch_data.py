import os
from dotenv import load_dotenv
from espn_api.football import League
import pandas as pd


load_dotenv()

league_id = os.getenv("LEAGUE_ID")
year = os.getenv("YEAR")
espn_s2 = os.getenv("ESPN_S2")
swid = os.getenv("ESPN_SWID")


league_id = int(league_id)
year = int(year)


league = League(
    league_id=league_id,
    year=year,
    espn_s2=espn_s2,
    swid=swid
)


rows = []

for week in range(1, league.settings.reg_season_count + 1):

    box_scores = league.box_scores(week)

    for matchup in box_scores:

        # Skip bye weeks
        if matchup.home_team is None or matchup.away_team is None:
            continue

        home = matchup.home_team
        away = matchup.away_team

        home_score = matchup.home_score
        away_score = matchup.away_score

        # Home team row
        rows.append({
            "team": home.team_name,
            "week": week,
            "points_for": home_score,
            "points_against": away_score,
            "win": int(home_score > away_score),
            "opponent": away.team_name
        })

        # Away team row
        rows.append({
            "team": away.team_name,
            "week": week,
            "points_for": away_score,
            "points_against": home_score,
            "win": int(away_score > home_score),
            "opponent": home.team_name
        })



df = pd.DataFrame(rows)

df = df.sort_values(["team", "week"]).reset_index(drop=True)


os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/weekly_stats.csv", index=False)

print("CSV saved: data/raw/weekly_stats.csv")
print(df.head())
print(f"\nTotal rows: {len(df)}")
