
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.train_cat_boost import FEATURES as BASE_FEATURES


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/features/player_features.parquet"
SPATIAL_PATH = (
    ROOT / "data/features/spatial_epaa_v2_all_seasons.parquet"
)

MODEL_DIR = ROOT / "models"

KEYS = ["GAME_ID", "PLAYER_ID", "TEAM_ID"]

SPATIAL_FEATURES = [
    "spatial_epaa_v2_100",
    "v2_prior_attempts_10",
]


def prepare_data():
    df = pd.read_parquet(DATA_PATH)
    spatial = pd.read_parquet(SPATIAL_PATH)

    for data in (df, spatial):
        data["GAME_ID"] = (
            data["GAME_ID"].astype(str).str.zfill(10)
        )

        for key in ["PLAYER_ID", "TEAM_ID"]:
            data[key] = pd.to_numeric(
                data[key], errors="raise"
            ).astype("int64")

    if df.duplicated(KEYS).any():
        raise ValueError("Duplicate baseline player-game keys")

    if spatial.duplicated(KEYS).any():
        raise ValueError("Duplicate spatial player-game keys")

    spatial = spatial[KEYS + SPATIAL_FEATURES]

    original_rows = len(df)

    df = df.merge(
        spatial,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if len(df) != original_rows:
        raise ValueError("Merge changed the number of rows")

    print("Spatial merge:")
    print(df["_merge"].value_counts())

    df = df.drop(columns="_merge")

    train = df[
        df["SEASON"].isin([
            "2022-23",
            "2023-24",
            "2024-25",
        ])
    ].copy()

    val = df[
        df["SEASON"] == "2025-26"
    ].copy()

    # Exactly the same eligibility rule as the baseline.
    train = train[
        train["career_games_prior"] >= 3
    ].copy()

    val = val[
        val["career_games_prior"] >= 3
    ].copy()

    if train.empty or val.empty:
        raise ValueError("Empty train or validation split")

    print(f"\nTraining rows: {len(train):,}")
    print(f"Validation rows: {len(val):,}")

    for name, split in [("Train", train), ("Validation", val)]:
        print(f"\n{name} spatial missing values:")
        print(
            split[SPATIAL_FEATURES]
            .isna()
            .mean()
            .mul(100)
            .round(3)
        )

    return train, val


def run_experiment(name, features, train, val, filename):
    X_train = train[features].replace(
        [np.inf, -np.inf], np.nan
    )
    X_val = val[features].replace(
        [np.inf, -np.inf], np.nan
    )

    y_train = train["PTS"]
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

    print(f"\nTraining: {name}")
    print(f"Feature count: {len(features)}")

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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / filename))

    importance = pd.Series(
        model.feature_importances_,
        index=features,
    ).sort_values(ascending=False)

    print(f"\n{name} results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Best iteration: {model.best_iteration_}")

    print("\nTop 15 features:")
    print(importance.head(15).to_string())

    return {
        "model": name,
        "mae": mae,
        "rmse": rmse,
        "best_iteration": model.best_iteration_,
    }


def main():
    train, val = prepare_data()

    baseline = run_experiment(
        name="Baseline",
        features=BASE_FEATURES,
        train=train,
        val=val,
        filename="catboost_points_baseline_ablation.cbm",
    )

    epaa = run_experiment(
        name="Baseline + Spatial EPAA v2",
        features=BASE_FEATURES + SPATIAL_FEATURES,
        train=train,
        val=val,
        filename="catboost_points_epaa_v2.cbm",
    )

    results = pd.DataFrame([baseline, epaa])

    print("\nABLATION RESULTS")
    print(results.to_string(index=False))

    print("\nChange from baseline:")
    print(f"MAE:  {epaa['mae'] - baseline['mae']:+.4f}")
    print(f"RMSE: {epaa['rmse'] - baseline['rmse']:+.4f}")

    if epaa["mae"] < baseline["mae"]:
        print("\nSpatial EPAA v2 improved validation MAE.")
    else:
        print("\nSpatial EPAA v2 did not improve validation MAE.")


if __name__ == "__main__":
    main()
