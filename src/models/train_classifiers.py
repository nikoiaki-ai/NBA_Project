from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from src.models.temporal_cv import GameDateSplit


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/game_outcomes.parquet"
MODEL_DIR = ROOT / "models/game_outcomes"
REPORT_DIR = ROOT / "reports/game_outcomes"

# Preserve the existing season split for this comparison.
TRAIN_SEASONS = [
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
]

VALID_SEASON = "2024-25"
HOLDOUT_SEASON = "2025-26"

N_FOLDS = 5


def load_data():
    df = pd.read_parquet(DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df = df.sort_values(
        ["GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    # Only use matchup differences and explicit pregame context.
    feature_cols = [
        col
        for col in df.columns
        if col.startswith("diff_")
        or col in [
            "home_games_prior",
            "away_games_prior",
            "home_rest_days",
            "away_rest_days",
        ]
    ]

    if not feature_cols:
        raise ValueError("No model features found")

    X = df[feature_cols].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if not all(
        pd.api.types.is_numeric_dtype(X[col])
        for col in X.columns
    ):
        raise ValueError("Non-numeric model features detected")

    train = df["SEASON"].isin(TRAIN_SEASONS)
    valid = df["SEASON"].eq(VALID_SEASON)
    holdout = df["SEASON"].eq(HOLDOUT_SEASON)

    if (
        (train & valid).any()
        or (train & holdout).any()
        or (valid & holdout).any()
    ):
        raise ValueError("Overlapping dataset splits")

    if not train.any():
        raise ValueError("No training games found")

    if not valid.any():
        raise ValueError("No validation games found")

    if df.loc[train, "GAME_DATE"].max() >= (
        df.loc[valid, "GAME_DATE"].min()
    ):
        raise ValueError(
            "Training dates must precede validation dates"
        )

    if holdout.any():
        if df.loc[valid, "GAME_DATE"].max() >= (
            df.loc[holdout, "GAME_DATE"].min()
        ):
            raise ValueError(
                "Validation dates must precede holdout dates"
            )

    print(f"Features: {len(feature_cols)}")
    print(f"Training games: {train.sum():,}")
    print(f"Validation games: {valid.sum():,}")
    print(f"Untouched holdout games: {holdout.sum():,}")

    return df, X, feature_cols, train, valid


def chronological_folds(df, train_mask):
    dates = np.array(
        sorted(df.loc[train_mask, "GAME_DATE"].unique())
    )

    if len(dates) < N_FOLDS + 1:
        raise ValueError(
            "Not enough training dates for chronological folds"
        )

    # Six chronological blocks create five expanding folds.
    # All games on a given date stay in the same block.
    blocks = np.array_split(dates, N_FOLDS + 1)

    folds = []

    for i in range(1, len(blocks)):
        train_dates = np.concatenate(blocks[:i])
        test_dates = blocks[i]

        train_idx = np.flatnonzero(
            train_mask.to_numpy()
            & df["GAME_DATE"].isin(train_dates).to_numpy()
        )

        test_idx = np.flatnonzero(
            train_mask.to_numpy()
            & df["GAME_DATE"].isin(test_dates).to_numpy()
        )

        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError("Empty chronological fold")

        if df.iloc[train_idx]["GAME_DATE"].max() >= (
            df.iloc[test_idx]["GAME_DATE"].min()
        ):
            raise ValueError(
                "Chronological fold contains overlapping dates"
            )

        folds.append((train_idx, test_idx))

    return folds


def make_pipeline(model):
    if isinstance(model, CalibratedClassifierCV):
        # Calibrated estimators already contain preprocessing.
        # Keep the outer "model" name for parameter grids.
        return Pipeline([
            ("model", model),
        ])

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def candidate_models(game_dates=None):
    calibration_cv = GameDateSplit(
        game_dates=game_dates,
        n_splits=3,
    )

    return {
        "xgboost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=4,
                tree_method="hist",
            ),
            {
                "model__max_depth": [2, 4],
                "model__n_estimators": [150, 300],
                "model__learning_rate": [0.03, 0.1],
            },
        ),

        "adaboost": (
            AdaBoostClassifier(random_state=42),
            {
                "model__n_estimators": [50, 150],
                "model__learning_rate": [0.05, 0.5],
            },
        ),

        "naive_bayes": (
            GaussianNB(),
            {
                "model__var_smoothing": [
                    1e-9,
                    1e-7,
                    1e-5,
                ],
            },
        ),

        "naive_bayes_calibrated": (
            CalibratedClassifierCV(
                estimator=make_pipeline(GaussianNB()),
                method="sigmoid",
                cv=calibration_cv,
                ensemble=True,
            ),
            {
                "model__estimator__model__var_smoothing": [
                    1e-9,
                    1e-7,
                    1e-5,
                ],
            },
        ),

        "knn": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [7, 15, 31],
                "model__weights": [
                    "uniform",
                    "distance",
                ],
            },
        ),

        "knn_calibrated": (
            CalibratedClassifierCV(
                estimator=make_pipeline(
                    KNeighborsClassifier()
                ),
                method="sigmoid",
                cv=calibration_cv,
                ensemble=True,
            ),
            {
                "model__estimator__model__n_neighbors": [
                    7,
                    15,
                    31,
                ],
                "model__estimator__model__weights": [
                    "uniform",
                    "distance",
                ],
            },
        ),

        "logistic_regression": (
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
            {
                "model__C": [
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                ],
            },
        ),

        "decision_tree": (
            DecisionTreeClassifier(
                random_state=42,
            ),
            {
                "model__max_depth": [3, 5, 8],
                "model__min_samples_leaf": [
                    10,
                    30,
                ],
            },
        ),

        "mlp": (
            MLPClassifier(
                activation="relu",
                solver="lbfgs",
                max_iter=1500,
                max_fun=30000,
                early_stopping=False,
                random_state=42,
            ),
            {
                "model__hidden_layer_sizes": [
                    (16,),
                    (32,),
                    (32, 16),
                ],
                "model__alpha": [
                    1.0,
                    10.0,
                ],
            },
        ),
    }


def evaluate(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
        "brier": brier_score_loss(
            y_true,
            probabilities,
        ),
    }


def main():
    df, X, features, train_mask, valid_mask = load_data()

    y = df["home_win"].astype(int)

    if not y.isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1")

    folds = chronological_folds(df, train_mask)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    candidates = candidate_models(
        game_dates=df["GAME_DATE"],
    )

    for name, (estimator, param_grid) in candidates.items():
        print(f"\nTraining {name}...")

        best_loss = float("inf")
        best_accuracy = None
        best_params = None

        for params in ParameterGrid(param_grid):
            fold_losses = []
            fold_accuracies = []

            for train_idx, test_idx in folds:
                model = make_pipeline(
                    clone(estimator)
                )
                model.set_params(**params)

                model.fit(
                    X.iloc[train_idx],
                    y.iloc[train_idx],
                )

                probabilities = model.predict_proba(
                    X.iloc[test_idx]
                )[:, 1]

                fold_losses.append(
                    log_loss(
                        y.iloc[test_idx],
                        probabilities,
                        labels=[0, 1],
                    )
                )

                fold_accuracies.append(
                    accuracy_score(
                        y.iloc[test_idx],
                        probabilities >= 0.5,
                    )
                )

            mean_loss = float(
                np.mean(fold_losses)
            )
            mean_accuracy = float(
                np.mean(fold_accuracies)
            )

            # Select hyperparameters using probability quality.
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_accuracy = mean_accuracy
                best_params = params

        if best_params is None:
            raise RuntimeError(
                f"No valid parameter configuration for {name}"
            )

        print(f"Selected CV log loss: {best_loss:.4f}")
        print(f"Selected CV accuracy: {best_accuracy:.4f}")
        print(f"Best parameters: {best_params}")

        final_model = make_pipeline(
            clone(estimator)
        )
        final_model.set_params(**best_params)

        # Fit on training seasons only.
        final_model.fit(
            X.loc[train_mask],
            y.loc[train_mask],
        )

        probabilities = final_model.predict_proba(
            X.loc[valid_mask]
        )[:, 1]

        validation_metrics = evaluate(
            y.loc[valid_mask],
            probabilities,
        )

        results.append({
            "model": name,
            "cv_log_loss": best_loss,
            "cv_accuracy": best_accuracy,
            **validation_metrics,
        })

        joblib.dump(
            {
                "pipeline": final_model,
                "features": features,
                "train_seasons": TRAIN_SEASONS,
                "validation_season": VALID_SEASON,
                "selection_metric": "cv_log_loss",
                "best_params": best_params,
                "cv_log_loss": best_loss,
                "cv_accuracy": best_accuracy,
            },
            MODEL_DIR / f"{name}.joblib",
        )

    results_df = pd.DataFrame(results).sort_values(
        "log_loss"
    )

    results_df.to_csv(
        REPORT_DIR / "base_model_results.csv",
        index=False,
    )

    print("\n=== Validation results ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()