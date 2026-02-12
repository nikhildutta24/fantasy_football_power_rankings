import os
from dotenv import load_dotenv
from espn_api.football import League
import pandas as pd


load_dotenv()

league_id = int(os.getenv("LEAGUE_ID"))
year = int(os.getenv("YEAR"))
espn_s2 = os.getenv("ESPN_S2")
swid = os.getenv("ESPN_SWID")


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

roster = []

for team in league.teams:
    team_name = team.team_name
    for player in team.roster:
        roster.append({
            "team_name": team_name,
            "player_name": player.name,
            "position": player.position,
            "slot_position": player.lineupSlot,
            "pro_team": player.proTeam,
            "injury_status": player.injuryStatus,
        })
        
fantasy_roster = pd.DataFrame(roster)
os.makedirs("data/raw", exist_ok=True)
fantasy_roster.to_csv("data/raw/fantasy_roster.csv", index=False)

print("Fantasy roster CSV saved: data/raw/fantasy_roster.csv")
print(fantasy_roster.head())