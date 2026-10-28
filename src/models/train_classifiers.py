from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/processed/game_outcomes.parquet"
MODEL_DIR = ROOT / "models/game_outcomes"
REPORT_DIR = ROOT / "reports/game_outcomes"

TRAIN_SEASONS = ["2022-23", "2023-24", "2024-25"]
VALID_SEASON = "2025-26"
HOLDOUT_SEASON = "2026-27"

N_FOLDS = 5


def load_data():
    df = pd.read_parquet(DATA_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Use matchup differences and home/away rest/history.
    feature_cols = [
        col for col in df.columns
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
        [np.inf, -np.inf], np.nan
    )

    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X):
        raise ValueError("Non-numeric model features detected")

    df = df.sort_values(
        ["GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    X = X.loc[df.index] if False else df[feature_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    train = df["SEASON"].isin(TRAIN_SEASONS)
    valid = df["SEASON"].eq(VALID_SEASON)
    holdout = df["SEASON"].eq(HOLDOUT_SEASON)

    if (train & valid).any() or (train & holdout).any():
        raise ValueError("Overlapping dataset splits")

    print(f"Features: {len(feature_cols)}")
    print(f"Training games: {train.sum():,}")
    print(f"Validation games: {valid.sum():,}")
    print(f"Untouched holdout games: {holdout.sum():,}")

    return df, X, feature_cols, train, valid


def chronological_folds(df, train_mask):
    dates = np.array(
        sorted(df.loc[train_mask, "GAME_DATE"].unique())
    )

    # Six chronological blocks create five expanding folds.
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

        folds.append((train_idx, test_idx))

    return folds


def make_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def candidate_models():
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
                "model__var_smoothing": [1e-9, 1e-7, 1e-5],
            },
        ),
        "knn": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [7, 15, 31],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "logistic_regression": (
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
            },
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "model__max_depth": [3, 5, 8],
                "model__min_samples_leaf": [10, 30],
            },
        ),
    }


def evaluate(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "brier": brier_score_loss(y_true, probabilities),
    }


def main():
    df, X, features, train_mask, valid_mask = load_data()

    y = df["home_win"].astype(int)
    folds = chronological_folds(df, train_mask)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for name, (estimator, param_grid) in candidate_models().items():
        print(f"\nTraining {name}...")

        best_score = -np.inf
        best_params = None

        for params in ParameterGrid(param_grid):
            fold_scores = []

            for train_idx, test_idx in folds:
                model = make_pipeline(
                    estimator.__class__(**estimator.get_params())
                )
                model.set_params(**params)

                model.fit(X.iloc[train_idx], y.iloc[train_idx])

                probabilities = model.predict_proba(
                    X.iloc[test_idx]
                )[:, 1]

                score = accuracy_score(
                    y.iloc[test_idx],
                    probabilities >= 0.5,
                )
                fold_scores.append(score)

            mean_accuracy = np.mean(fold_scores)

            if mean_accuracy > best_score:
                best_score = mean_accuracy
                best_params = params

        print(f"Best CV accuracy: {best_score:.4f}")
        print(f"Best parameters: {best_params}")

        final_model = make_pipeline(
            estimator.__class__(**estimator.get_params())
        )
        final_model.set_params(**best_params)

        final_model.fit(
            X.loc[train_mask],
            y.loc[train_mask],
        )

        probabilities = final_model.predict_proba(
            X.loc[valid_mask]
        )[:, 1]

        metrics = evaluate(
            y.loc[valid_mask],
            probabilities,
        )

        results.append({
            "model": name,
            "cv_accuracy": best_score,
            **metrics,
        })

        joblib.dump(
            {
                "pipeline": final_model,
                "features": features,
                "train_seasons": TRAIN_SEASONS,
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
