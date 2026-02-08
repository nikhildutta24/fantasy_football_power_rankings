from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression



BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "weekly_stats.csv"

df = pd.read_csv(DATA_PATH)
df = df.sort_values(["team", "week"])


#Recent Scoring
df["rolling_avg"] = (
    df.groupby("team")["points_for"]
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)

df["rolling_std"] = (
    df.groupby("team")["points_for"]
    .rolling(3)
    .std()
    .reset_index(level=0, drop=True)
)

#Strength of Schedule
opp_def = (
    df.groupby("team")["points_against"]
    .mean()
    .rename("opp_def_avg")
)

df = df.merge(
    opp_def,
    left_on="opponent",
    right_index=True,
    how="left"
)

#Expected wins
league_week_avg = df.groupby("week")["points_for"].transform("mean")

df["expected_win"] = (df["points_for"] > league_week_avg).astype(int)

# Season scoring
df["season_avg"] = (
    df.groupby("team")["points_for"]
    .transform("mean")
)

# Point Differential
df["point_diff"] = df["points_for"] - df["points_against"]

df["avg_point_diff"] = (
    df.groupby("team")["point_diff"]
    .transform("mean")
)

# Last Week Score
df["last_week_score"] = (
    df.groupby("team")["points_for"]
    .shift(1)
)


# ----------------------------
# Build team-level stats
# ----------------------------
team_stats = df.groupby("team").agg(
    recent_scoring=("rolling_avg", "mean"),
    consistency=("rolling_std", "mean"),
    sos=("opp_def_avg", "mean"),
    season_avg=("season_avg", "mean"),
    avg_point_diff=("avg_point_diff", "mean"),
    last_week_score=("last_week_score", "mean"),
    wins=("win", "sum"),
    expected_wins=("expected_win", "sum")
).dropna()

# Luck
team_stats["luck"] = team_stats["wins"] - team_stats["expected_wins"]

# ----------------------------
# Z-score normalization
# ----------------------------
features = [
    "recent_scoring",
    "consistency",
    "sos",
    "luck",
    "season_avg",
    "avg_point_diff",
    "last_week_score"
]

for col in features:
    team_stats[f"z_{col}"] = (
        (team_stats[col] - team_stats[col].mean())
        / team_stats[col].std()
    )


#Model learning section
df["next_week_score"] = (
    df.groupby("team")["points_for"].shift(-1)
)

model_df = df.dropna(subset=["next_week_score"]).copy()

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
    #Training data
    train_df = model_df[model_df["week"] < week]
    
    #Test data (current week)
    test_df = model_df[model_df["week"] == week]
    
    #Remove rows w/ missing features
    train_df = train_df.dropna(subset=feature_cols)
    test_df = test_df.dropna(subset=feature_cols)
    
    
    #Skip early weeks, not enough training data
    if len(train_df) < 20 or len(test_df) == 0:
        continue
    
    #Model inputs:
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    # y = next weeks scores
    y_train = train_df["next_week_score"]
    y_test = test_df["next_week_score"]
    
    
    #Regression training
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    week_results = pd.DataFrame({
        "team": test_df["team"].values,
        "week": test_df["week"].values,
        "predicted_score": preds,
        "actual_score": y_test.values
    })
    
    results.append(week_results)
    
#Combine predictions
results_df = pd.concat(results, ignore_index=True)
#Calculate correlation
corr = results_df["predicted_score"].corr(results_df["actual_score"])
print(f"Overall prediction correlation: {corr:.3f}")

# ----------------------------
# Final power score
# ----------------------------
team_stats["power_score"] = (
      0.30 * team_stats["z_recent_scoring"]
    + 0.25 * team_stats["z_season_avg"]
    + 0.20 * team_stats["z_avg_point_diff"]
    + 0.15 * team_stats["z_sos"]
    + 0.05 * team_stats["z_last_week_score"]
    + 0.05 * team_stats["z_luck"]
    - 0.10 * team_stats["z_consistency"]
)
