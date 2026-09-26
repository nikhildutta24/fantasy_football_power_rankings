from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
WEEKLY_STATS_PATH = BASE_DIR / "data" / "raw" / "weekly_stats.csv"
INJURY_PATH = BASE_DIR / "data" / "raw" / "weekly_injuries.csv"
ROSTER_PATH = BASE_DIR / "data" / "raw" / "fantasy_roster.csv"
PROJECTIONS_PATH = BASE_DIR / "data" / "raw" / "team_projections.csv"
RANKINGS_HISTORY_PATH = BASE_DIR / "data" / "raw" / "rankings_history.csv"

# ----------------------------
# Load data
# ----------------------------
stats_df = pd.read_csv(WEEKLY_STATS_PATH) if WEEKLY_STATS_PATH.exists() else pd.DataFrame()
injuries_df = pd.read_csv(INJURY_PATH)
roster_df = pd.read_csv(ROSTER_PATH)
projections_df = pd.read_csv(PROJECTIONS_PATH)
if RANKINGS_HISTORY_PATH.exists():
    prev_rankings = pd.read_csv(RANKINGS_HISTORY_PATH, index_col="team_name")
else:
    prev_rankings = pd.DataFrame()

roster_df["team_name"] = roster_df["team_name"].str.strip().str.replace(r"\s+", " ", regex=True)
projections_df["team_name"] = projections_df["team_name"].str.strip().str.replace(r"\s+", " ", regex=True)
injuries_df["NAME"] = injuries_df["NAME"].str.strip().str.replace(r"\s+", " ", regex=True)

if not stats_df.empty:
    stats_df["team"] = stats_df["team"].str.strip().str.replace(r"\s+", " ", regex=True)
    stats_df = stats_df.sort_values(["team", "week"])

current_week = int(stats_df["week"].max()) if not stats_df.empty else 0

# ----------------------------
# Injury impact
# ----------------------------
fantasy_positions = {"QB", "RB", "WR", "TE", "K"}
injuries_df["POS"] = injuries_df["POS"].replace({"PK": "K"})
injuries_df = injuries_df[injuries_df["POS"].isin(fantasy_positions)]

roster_injuries_df = roster_df.merge(
    injuries_df,
    how="left",
    left_on="player_name",
    right_on="NAME"
)

POSITION_WEIGHTS = {
    "QB": 3.0,
    "RB": 2.5,
    "WR": 2.0,
    "TE": 1.5,
    "K": 1.0
}

def roster_injury_impact(team_roster_df, position_weights):
    starters = team_roster_df[~team_roster_df["slot_position"].isin({"BE", "IR"})]
    bench = team_roster_df[team_roster_df["slot_position"] == "BE"]

    total_impact = 0

    for _, injured in starters[starters["STATUS"].notna()].iterrows():
        status = str(injured["STATUS"]).lower().strip()

        if "injured reserve" in status or status == "doubtful":
            replacement = bench[bench["position"] == injured["position"]]

            if not replacement.empty:
                best_backup = replacement["projected_total_points"].max()
                dropoff = max(injured["projected_total_points"] - best_backup, 0)
            else:
                dropoff = injured["projected_total_points"]

            pos_weight = position_weights.get(injured["position"], 1.0)
            total_impact += (dropoff / 100) * pos_weight

        elif status in {"questionable", "probable"}:
            total_impact += 0.25 * position_weights.get(injured["position"], 1.0)

    return total_impact

team_injury_impact = (
    roster_injuries_df.groupby("team_name")
    .apply(lambda df: roster_injury_impact(df, POSITION_WEIGHTS))
    .reset_index()
    .rename(columns={0: "total_injury_impact"})
)

# ----------------------------
# Preseason path
# ----------------------------
if current_week == 0:
    team_stats = projections_df.set_index("team_name").copy()
    team_stats = team_stats.merge(
        team_injury_impact.set_index("team_name"),
        left_index=True,
        right_index=True,
        how="left"
    )
    team_stats["total_injury_impact"] = team_stats["total_injury_impact"].fillna(0)

    for col in ["projected_team_points", "total_injury_impact"]:
        team_stats[f"z_{col}"] = (
            (team_stats[col] - team_stats[col].mean()) / team_stats[col].std()
        )

    team_stats["power_score"] = (
          0.70 * team_stats["z_projected_team_points"]
        - 0.30 * team_stats["z_total_injury_impact"]
    )

    team_stats = team_stats.sort_values("power_score", ascending=False)

    print("=" * 50)
    print("PRESEASON POWER RANKINGS")
    print("=" * 50)
    for rank, (team, row) in enumerate(team_stats.iterrows(), 1):
        print(f"{rank}. {team} ({row['power_score']:+.2f})")
    print("=" * 50)
    exit()

# ----------------------------
# Scoring metrics
# ----------------------------
stats_df["rolling_avg"] = (
    stats_df.groupby("team")["points_for"]
    .rolling(3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

stats_df["rolling_std"] = (
    stats_df.groupby("team")["points_for"]
    .rolling(3, min_periods=2)
    .std()
    .reset_index(level=0, drop=True)
)

opp_def = (
    stats_df.groupby("team")["points_against"]
    .mean()
    .rename("opp_def_avg")
)

stats_df = stats_df.merge(
    opp_def,
    left_on="opponent",
    right_index=True,
    how="left"
)

league_week_avg = stats_df.groupby("week")["points_for"].transform("mean")
stats_df["expected_win"] = (stats_df["points_for"] > league_week_avg).astype(int)

stats_df["season_avg"] = (
    stats_df.groupby("team")["points_for"]
    .transform("mean")
)

stats_df["point_diff"] = stats_df["points_for"] - stats_df["points_against"]
stats_df["avg_point_diff"] = (
    stats_df.groupby("team")["point_diff"]
    .transform("mean")
)

stats_df["last_week_score"] = (
    stats_df.groupby("team")["points_for"]
    .shift(1)
)

# ----------------------------
# Build team-level stats
# ----------------------------
team_stats = stats_df.groupby("team").agg(
    recent_scoring=("rolling_avg", "mean"),
    consistency=("rolling_std", "mean"),
    sos=("opp_def_avg", "mean"),
    season_avg=("season_avg", "mean"),
    avg_point_diff=("avg_point_diff", "mean"),
    last_week_score=("last_week_score", "mean"),
    wins=("win", "sum"),
    expected_wins=("expected_win", "sum")
)

team_stats = team_stats.dropna(subset=["recent_scoring", "season_avg"])
team_stats["luck"] = team_stats["wins"] - team_stats["expected_wins"]

team_stats = team_stats.merge(
    team_injury_impact,
    left_index=True,
    right_on="team_name",
    how="left"
)
team_stats = team_stats.set_index("team_name")
team_stats["total_injury_impact"] = team_stats["total_injury_impact"].fillna(0)

team_stats = team_stats.merge(projections_df, left_index=True, right_on="team_name", how="left")
team_stats = team_stats.set_index("team_name")

# ----------------------------
# Z-score normalization
# ----------------------------
in_season_features = [
    "recent_scoring",
    "consistency",
    "sos",
    "luck",
    "season_avg",
    "avg_point_diff",
    "last_week_score",
    "total_injury_impact"
]

all_features = in_season_features + ["projected_team_points"]

for col in all_features:
    mean = team_stats[col].mean()
    std = team_stats[col].std()
    team_stats[f"z_{col}"] = (team_stats[col] - mean) / std if std > 0 else 0.0

# ----------------------------
# Walk-forward regression
# ----------------------------
stats_df["next_week_score"] = (
    stats_df.groupby("team")["points_for"].shift(-1)
)

model_df = stats_df.dropna(subset=["next_week_score"]).copy()

feature_cols = [
    "rolling_avg",
    "rolling_std",
    "opp_def_avg",
    "season_avg",
    "avg_point_diff",
    "last_week_score"
]

results = []
weeks = sorted(model_df["week"].unique())

for week in weeks:
    train_df = model_df[model_df["week"] < week].dropna(subset=feature_cols)
    test_df = model_df[model_df["week"] == week].dropna(subset=feature_cols)

    if len(train_df) < 20 or len(test_df) == 0:
        continue

    model = LinearRegression()
    model.fit(train_df[feature_cols], train_df["next_week_score"])

    preds = model.predict(test_df[feature_cols])
    results.append(pd.DataFrame({
        "team": test_df["team"].values,
        "week": test_df["week"].values,
        "predicted_score": preds,
        "actual_score": test_df["next_week_score"].values
    }))

if results:
    results_df = pd.concat(results, ignore_index=True)
    results_df["diff"] = abs(results_df["predicted_score"] - results_df["actual_score"])
    corr = results_df["predicted_score"].corr(results_df["actual_score"])
else:
    results_df = pd.DataFrame()
    corr = None

# ----------------------------
# Static power score
# ----------------------------
in_season_score = (
      0.30 * team_stats["z_recent_scoring"]
    + 0.25 * team_stats["z_season_avg"]
    + 0.20 * team_stats["z_avg_point_diff"]
    + 0.15 * team_stats["z_sos"]
    + 0.05 * team_stats["z_last_week_score"]
    + 0.05 * team_stats["z_luck"]
    - 0.10 * team_stats["z_consistency"]
    - 0.10 * team_stats["z_total_injury_impact"]
)

preseason_score = (
      0.70 * team_stats["z_projected_team_points"]
    - 0.30 * team_stats["z_total_injury_impact"]
)

alpha = min(current_week / 4, 1.0)
team_stats["power_score"] = alpha * in_season_score + (1 - alpha) * preseason_score
team_stats = team_stats.sort_values("power_score", ascending=False)

# ----------------------------
# Dynamic power score
# ----------------------------
z_in_season_features = [f"z_{col}" for col in in_season_features]

if current_week >= 4 and results:
    walk_forward_merged = results_df.merge(
        team_stats[z_in_season_features],
        left_on="team",
        right_index=True,
        how="inner"
    ).dropna(subset=z_in_season_features)

    reg = LinearRegression()
    reg.fit(walk_forward_merged[z_in_season_features], walk_forward_merged["actual_score"])

    weights = pd.Series(reg.coef_, index=z_in_season_features)

    team_stats["dynamic_power_score"] = team_stats[z_in_season_features] @ weights
    team_stats["dynamic_power_score_z"] = (
        (team_stats["dynamic_power_score"] - team_stats["dynamic_power_score"].mean())
        / team_stats["dynamic_power_score"].std()
    )
    team_stats = team_stats.sort_values("dynamic_power_score_z", ascending=False)
    ranking_col = "dynamic_power_score_z"
    ranking_label = "POWER RANKINGS (Dynamic Model)"
else:
    ranking_col = "power_score"
    ranking_label = f"POWER RANKINGS (Week {current_week} — Blended Score)"

# Compute rank change
team_stats["rank"] = range(1, len(team_stats) + 1)

if not prev_rankings.empty and "rank" in prev_rankings.columns:
    team_stats["prev_rank"] = team_stats.index.map(prev_rankings["rank"])
    team_stats["rank_change"] = team_stats["prev_rank"] - team_stats["rank"]
else:
    team_stats["rank_change"] = None

# Print table
print()
print(f"{ranking_label}")
print()

header = f"{'Rank':<6}{'Chg':<6}{'Team':<25}{'W':<5}{'PPG':<10}{'Score':<10}"
divider = "-" * len(header)

print(header)
print(divider)

for rank, (team, row) in enumerate(team_stats.iterrows(), 1):
    score = row[ranking_col]
    wins = int(row["wins"])
    avg = row["season_avg"]

    if pd.isna(row["rank_change"]):
        chg = "NEW"
    elif row["rank_change"] > 0:
        chg = f"+{int(row['rank_change'])}"
    elif row["rank_change"] < 0:
        chg = str(int(row["rank_change"]))
    else:
        chg = "--"

    print(f"{rank:<6}{chg:<6}{team:<25}{wins:<5}{avg:<10.1f}{score:<+10.2f}")

print(divider)

# Save current rankings for next week
team_stats[["rank"]].to_csv(RANKINGS_HISTORY_PATH)

# ----------------------------
# Model performance visualization
# ----------------------------
if not results_df.empty:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(results_df["actual_score"], results_df["predicted_score"], alpha=0.6, edgecolors="white", linewidths=0.5)
    min_val = min(results_df["actual_score"].min(), results_df["predicted_score"].min())
    max_val = max(results_df["actual_score"].max(), results_df["predicted_score"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title("Predicted vs Actual Scores (Walk-Forward Model)")
    ax.legend()
    ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=12 , verticalalignment="top")
    plt.tight_layout()
    plt.savefig("predicted_vs_actual.png", dpi=150)
    plt.show()
    
    
if not results_df.empty:
    mae = results_df["diff"].mean()
    rmse = np.sqrt((results_df["diff"] ** 2).mean())
    corr = results_df["predicted_score"].corr(results_df["actual_score"])
    baseline_mae = (results_df["actual_score"] - results_df["actual_score"].mean()).abs().mean()
    skill_score = 1 - (mae / baseline_mae)

    print("\nModel Validation Metrics:")
    print(f"  MAE:            {mae:.2f} pts")
    print(f"  RMSE:           {rmse:.2f} pts")
    print(f"  Correlation:    {corr:.3f}")
    print(f"  Baseline MAE:   {baseline_mae:.2f} pts")
    print(f"  Skill Score:    {skill_score:.3f}")
    
