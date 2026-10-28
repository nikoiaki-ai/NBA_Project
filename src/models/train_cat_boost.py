
"""Train a chronological CatBoost player-points baseline."""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "features" / "player_features.parquet"

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "catboost_points_baseline.cbm"

FEATURES = [
    "pts_avg_5",
    "pts_avg_10",
    "min_avg_5",
    "min_avg_10",
    "fga_avg_5",
    "fga_avg_10",
    "fg3a_avg_5",
    "fg3a_avg_10",
    "fta_avg_5",
    "fta_avg_10",
    "ast_avg_5",
    "reb_avg_5",
    "tov_avg_5",
    "pts_std_10",
    "pts_season_avg",
    "min_season_avg",
    "fga_season_avg",
    "fta_season_avg",
    "fg_pct_10",
    "fg3_pct_10",
    "ft_pct_10",
    "career_games_prior",
    "season_games_prior",
    "player_rest_days",
    "rest_days",
    "opp_rest_days",
    "is_home",
    "opp_pts_allowed_10",
    "opp_def_rating_10",
    "opp_pace_10",
    "opp_win_pct_10",
]


def main():
    df = pd.read_parquet(DATA_PATH)

    train = df[df["SEASON"].isin([
        "2022-23", "2023-24", "2024-25"
    ])].copy()

    val = df[df["SEASON"] == "2025-26"].copy()

    # Avoid scoring players with no prior history.
    train = train[train["career_games_prior"] >= 3]
    val = val[val["career_games_prior"] >= 3]

    X_train = train[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    y_train = train["PTS"]

    X_val = val[FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    y_val = val["PTS"]

    model = CatBoostRegressor(
        iterations=1500,
        depth=6,
        learning_rate=0.03,
        loss_function="RMSE",
        l2_leaf_reg=5,
        random_seed=42,
        verbose=200,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        use_best_model=True,
    )

    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )

    # Simple historical baseline for comparison.
    baseline = val["pts_avg_10"]
    valid_baseline = baseline.notna()

    baseline_mae = mean_absolute_error(
        y_val[valid_baseline],
        baseline[valid_baseline],
    )

    print(f"\nTrain rows: {len(train):,}")
    print(f"Validation rows: {len(val):,}")
    print(f"CatBoost MAE: {mae:.4f}")
    print(f"CatBoost RMSE: {rmse:.4f}")
    print(f"Historical baseline MAE: {baseline_mae:.4f}")

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False)

    print("\nTop 15 features:")
    print(importance.head(15).to_string())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))

    print(f"\nSaved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
