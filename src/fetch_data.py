import os
from dotenv import load_dotenv
from espn_api.football import League
import pandas as pd

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

league_id = int(os.getenv("LEAGUE_ID"))
year = int(os.getenv("YEAR"))
espn_s2 = os.getenv("ESPN_S2")
swid = os.getenv("ESPN_SWID")

# ----------------------------
# Connect to ESPN league
# ----------------------------
league_kwargs = {
    "league_id": league_id,
    "year": year,
}

if espn_s2 and swid:
    league_kwargs["espn_s2"] = espn_s2
    league_kwargs["swid"] = swid

league = League(**league_kwargs)

# ----------------------------
# Fetch weekly matchup data
# ----------------------------
rows = []
def get_lineup_score(lineup):
    return sum(
        p.points for p in lineup
        if p.slot_position not in {"BE", "IR"}
    )

for week in range(1, league.settings.reg_season_count + 1):
    box_scores = league.box_scores(week)

        # Check if this week has real data by sampling the first matchup
    first = next((m for m in box_scores if m.home_team and m.away_team), None)
    if first is None:
        break
    
    home_score = get_lineup_score(first.home_lineup)
    away_score = get_lineup_score(first.away_lineup)
    
    # If scores match the previous week exactly, we've hit stale data
    if week > 1 and round(home_score, 2) == last_home_score:
        break
    
    last_home_score = round(home_score, 2)

    for matchup in box_scores:
        if matchup.home_team is None or matchup.away_team is None:
            continue

        home_score = get_lineup_score(matchup.home_lineup)
        away_score = get_lineup_score(matchup.away_lineup)

        if home_score == 0 and away_score == 0:
            continue

        home = matchup.home_team
        away = matchup.away_team

        rows.append({
            "team": home.team_name,
            "week": week,
            "points_for": home_score,
            "points_against": away_score,
            "win": int(home_score > away_score),
            "opponent": away.team_name
        })

        rows.append({
            "team": away.team_name,
            "week": week,
            "points_for": away_score,
            "points_against": home_score,
            "win": int(away_score > home_score),
            "opponent": home.team_name
        })

# ----------------------------
# Fetch roster data
# ----------------------------
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
            "projected_total_points": player.projected_total_points,
        })

fantasy_roster = pd.DataFrame(roster)
fantasy_roster["slot_position"] = fantasy_roster["slot_position"].replace({"RB/WR/TE": "FLEX"})

# ----------------------------
# Compute weighted projections
# ----------------------------
def get_slot_weight(slot):
    if slot == "BE":
        return 0.1
    elif slot == "IR":
        return 0.0
    else:
        return 1.0

fantasy_roster["slot_weight"] = fantasy_roster["slot_position"].apply(get_slot_weight)
fantasy_roster["weighted_projection"] = (
    fantasy_roster["slot_weight"] * fantasy_roster["projected_total_points"]
).round(2)

team_projections = (
    fantasy_roster.groupby("team_name")["weighted_projection"]
    .sum()
    .reset_index()
    .rename(columns={"weighted_projection": "projected_team_points"})
)

# ----------------------------
# Save to CSV
# ----------------------------
os.makedirs("data/raw", exist_ok=True)

if rows:
    df = pd.DataFrame(rows)
    df = df.sort_values(["team", "week"]).reset_index(drop=True)
    df.to_csv("data/raw/weekly_stats.csv", index=False)
    print("CSV saved: data/raw/weekly_stats.csv")
    print(df.head())
    print(f"\nTotal rows: {len(df)}")
else:
    print("No completed matchups found — weekly_stats.csv not written.")

fantasy_roster.to_csv("data/raw/fantasy_roster.csv", index=False)

print("\nFantasy roster CSV saved: data/raw/fantasy_roster.csv")
print(fantasy_roster.head())

team_projections.to_csv("data/raw/team_projections.csv", index=False)
print("\nTeam projections CSV saved: data/raw/team_projections.csv")
print(team_projections.head())

box_scores = league.box_scores(1)
matchup = box_scores[0]
print(matchup.__dict__)