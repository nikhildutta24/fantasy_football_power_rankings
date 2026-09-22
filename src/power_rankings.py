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

# ----------------------------
# Load data
# ----------------------------
stats_df = pd.read_csv(WEEKLY_STATS_PATH) if WEEKLY_STATS_PATH.exists() else pd.DataFrame()
injuries_df = pd.read_csv(INJURY_PATH)
roster_df = pd.read_csv(ROSTER_PATH)
projections_df = pd.read_csv(PROJECTIONS_PATH)

stats_df["team"] = stats_df["team"].str.strip().str.replace(r"\s+", " ", regex=True)
roster_df["team_name"] = roster_df["team_name"].str.strip().str.replace(r"\s+", " ", regex=True)
projections_df["team_name"] = projections_df["team_name"].str.strip().str.replace(r"\s+", " ", regex=True)
injuries_df["NAME"] = injuries_df["NAME"].str.strip().str.replace(r"\s+", " ", regex=True)

if not stats_df.empty:
    stats_df = stats_df.sort_values(["team", "week"])
    stats_df["team"] = stats_df["team"].str.lstrip()
    stats_df["team"] = stats_df["team"].str.replace(r"\s+", " ", regex=True)

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

def injury_impact(row):
    status = row["STATUS"]
    position = row["position"]
    is_starter = row["slot_position"] not in {"BE", "IR"}

    if pd.isna(status):
        return 0

    s = str(status).lower().replace("_", " ").strip()

    if "injured reserve" in s or s == "ir":
        base = 1.0
    elif s == "doubtful":
        base = 0.5
    elif s in {"questionable", "probable"}:
        return 0.25
    else:
        return 0

    pos_weight = POSITION_WEIGHTS.get(position, 1.0)
    starter_mult = 1.0 if is_starter else 0.25

    return base * pos_weight * starter_mult

roster_injuries_df["injury_impact"] = roster_injuries_df.apply(injury_impact, axis=1)

team_injury_impact = (
    roster_injuries_df.groupby("team_name")["injury_impact"]
    .sum()
    .reset_index()
    .rename(columns={"injury_impact": "total_injury_impact"})
)

# ----------------------------
# Preseason path
# ----------------------------
if current_week == 0:
    print("No completed matchups found — running preseason rankings.")

    team_stats = projections_df.set_index("team_name").copy()
    team_stats = team_stats.merge(team_injury_impact.set_index("team_name"), left_index=True, right_index=True, how="left")
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
    print("\nPreseason Power Rankings:")
    print(team_stats[["power_score"]])
    exit()

# ----------------------------
# Scoring metrics
# ----------------------------
stats_df["rolling_avg"] = (
    stats_df.groupby("team")["points_for"]
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)

stats_df["rolling_std"] = (
    stats_df.groupby("team")["points_for"]
    .rolling(3)
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
).dropna()

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
    team_stats[f"z_{col}"] = (
        (team_stats[col] - team_stats[col].mean())
        / team_stats[col].std()
    )

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
    print(f"Overall prediction correlation: {corr:.3f}")
    print(results_df.head(20))
else:
    print("No week had enough data to train/test the predictive model.")

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

alpha = min(current_week / 6, 1.0)
team_stats["power_score"] = alpha * in_season_score + (1 - alpha) * preseason_score

team_stats = team_stats.sort_values("power_score", ascending=False)
print("\nFinal Power Rankings:")
print(team_stats[["power_score"]])

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
    )

    X_wf = walk_forward_merged[z_in_season_features]
    Y_wf = walk_forward_merged["actual_score"]

    reg = LinearRegression()
    reg.fit(X_wf, Y_wf)

    weights = pd.Series(reg.coef_, index=z_in_season_features)
    print("\nLearned dynamic weights:\n", weights.sort_values(ascending=False))

    team_stats["dynamic_power_score"] = team_stats[z_in_season_features] @ weights
    team_stats["dynamic_power_score_z"] = (
        (team_stats["dynamic_power_score"] - team_stats["dynamic_power_score"].mean())
        / team_stats["dynamic_power_score"].std()
    )

    team_stats = team_stats.sort_values("dynamic_power_score_z", ascending=False)
    print("Dynamic Power Rankings:")
    print(team_stats[["dynamic_power_score_z"]])
else:
    print(f"Week {current_week}: not enough data for dynamic model, using blended score only.")

# ----------------------------
# Model performance visualization
# ----------------------------
if results:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(results_df["actual_score"], results_df["predicted_score"], alpha=0.6, edgecolors="white", linewidths=0.5)
    min_val = min(results_df["actual_score"].min(), results_df["predicted_score"].min())
    max_val = max(results_df["actual_score"].max(), results_df["predicted_score"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title("Predicted vs Actual Scores (Walk-Forward Model)")
    ax.legend()
    ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=12, verticalalignment="top")
    plt.tight_layout()
    plt.savefig("predicted_vs_actual.png", dpi=150)
    plt.show()