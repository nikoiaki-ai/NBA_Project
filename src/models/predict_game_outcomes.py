
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd

from src.features.game_outcome_f import (
    load_logs,
    add_team_features,
    ROLLING_STATS,
    WINDOWS,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models/game_outcomes"
STACK_DIR = MODEL_DIR / "stacking"

BASE_NAMES = [
    "xgboost",
    "adaboost",
    "naive_bayes",
    "knn",
    "logistic_regression",
    "decision_tree",
]


def build_team_snapshot(featured, team, game_date):
    """Construct one team's features immediately before tipoff."""

    history = featured[
        (featured["TEAM_ABBREVIATION"] == team)
        & (featured["GAME_DATE"] < game_date)
    ].sort_values(["GAME_DATE", "GAME_ID"])

    if history.empty:
        raise ValueError(f"No completed history for {team}")

    latest = history.iloc[-1]
    snapshot = {}

    # Use actual completed-game stats, not the lagged
    # features stored on the latest historical row.
    for stat in ROLLING_STATS:
        for window in WINDOWS:
            col = f"{stat.lower()}_avg_{window}"

            snapshot[col] = (
                history[stat].tail(window).mean()
            )

    snapshot["games_prior"] = len(history)

    days_since_last_game = (
        game_date - latest["GAME_DATE"]
    ).days

    snapshot["rest_days"] = max(
        days_since_last_game - 1, 0
    )

    return snapshot, latest["GAME_DATE"]


def build_matchup_features(
    featured, home_team, away_team, game_date, features
):
    home, home_last = build_team_snapshot(
        featured, home_team, game_date
    )
    away, away_last = build_team_snapshot(
        featured, away_team, game_date
    )

    row = {}

    for key in home:
        row[f"home_{key}"] = home[key]
        row[f"away_{key}"] = away[key]
        row[f"diff_{key}"] = home[key] - away[key]

    X = pd.DataFrame([row])

    missing = [col for col in features if col not in X.columns]
    if missing:
        raise ValueError(
            f"Missing expected model features: {missing}"
        )

    X = X[features].replace(
        [np.inf, -np.inf], np.nan
    )

    print(f"\n{away_team} at {home_team}")
    print(f"Home last game: {home_last.date()}")
    print(f"Away last game: {away_last.date()}")
    print(
        f"Home rest: {home['rest_days']} days | "
        f"Away rest: {away['rest_days']} days"
    )

    return X


def predict(home_team, away_team, game_date):
    # Exclude games on or after the prediction date.
    logs = load_logs()

    logs = logs[
        logs["GAME_DATE"] < game_date
    ].copy()

    if logs.empty:
        raise ValueError("No historical games available")

    featured = add_team_features(logs)

    saved_models = {}
    for name in BASE_NAMES:
        saved_models[name] = joblib.load(
            MODEL_DIR / f"{name}.joblib"
        )

    features = saved_models["xgboost"]["features"]

    for name, saved in saved_models.items():
        if saved["features"] != features:
            raise ValueError(
                f"Feature mismatch for {name}"
            )

    X = build_matchup_features(
        featured,
        home_team,
        away_team,
        game_date,
        features,
    )

    probabilities = {}

    for name, saved in saved_models.items():
        probabilities[name] = float(
            saved["pipeline"].predict_proba(X)[0, 1]
        )

    base_frame = pd.DataFrame(
        [{name: probabilities[name] for name in BASE_NAMES}]
    )

    logistic = joblib.load(
        STACK_DIR / "logistic_stack.joblib"
    )
    mlp = joblib.load(
        STACK_DIR / "mlp_stack.joblib"
    )

    logistic_input = base_frame[
        logistic["base_names"]
    ]
    mlp_input = base_frame[
        mlp["base_names"]
    ]

    probabilities["simple_average"] = float(
        base_frame.mean(axis=1).iloc[0]
    )

    probabilities["logistic_stack"] = float(
        logistic["model"].predict_proba(
            logistic_input
        )[0, 1]
    )

    probabilities["mlp_stack"] = float(
        mlp["model"].predict_proba(
            mlp_input
        )[0, 1]
    )

    print("\nHOME WIN PROBABILITIES")
    print("-" * 38)

    for name, probability in probabilities.items():
        print(f"{name:22s} {probability:6.2%}")

    final_probability = probabilities["logistic_stack"]

    predicted_winner = (
        home_team if final_probability >= 0.5
        else away_team
    )

    print("\nPRIMARY PREDICTION")
    print(f"Winner: {predicted_winner}")
    print(
        f"Confidence: "
        f"{max(final_probability, 1 - final_probability):.2%}"
    )

    return probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--date", required=True)

    args = parser.parse_args()

    predict(
        args.home.upper(),
        args.away.upper(),
        pd.Timestamp(args.date),
    )
