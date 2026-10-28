
from pathlib import Path
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    log_loss, brier_score_loss,
)

from src.models.train_classifiers import (
    load_data,
    chronological_folds,
    candidate_models,
    make_pipeline,
    MODEL_DIR,
    REPORT_DIR,
)


BASE_NAMES = list(candidate_models().keys())
STACK_DIR = MODEL_DIR / "stacking"


def metrics(y, probabilities):
    probabilities = np.clip(
        np.asarray(probabilities), 1e-6, 1 - 1e-6
    )

    return {
        "accuracy": accuracy_score(
            y, probabilities >= 0.5
        ),
        "roc_auc": roc_auc_score(y, probabilities),
        "log_loss": log_loss(y, probabilities),
        "brier": brier_score_loss(y, probabilities),
    }


def make_mlp(hidden_layers=(8,)):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="lbfgs",
            alpha=1.0,
            max_iter=2000,
            random_state=42,
        )),
    ])


def main():
    df, X, features, train_mask, valid_mask = load_data()
    y = df["home_win"].astype(int)

    folds = chronological_folds(df, train_mask)

    train_indices = np.flatnonzero(train_mask.to_numpy())
    valid_indices = np.flatnonzero(valid_mask.to_numpy())

    oof = pd.DataFrame(
        np.nan,
        index=train_indices,
        columns=BASE_NAMES,
    )

    valid_predictions = pd.DataFrame(
        index=valid_indices,
        columns=BASE_NAMES,
        dtype=float,
    )

    for name in BASE_NAMES:
        saved = joblib.load(MODEL_DIR / f"{name}.joblib")
        final_pipeline = saved["pipeline"]

        print(f"\nGenerating OOF predictions: {name}")

        for fold_num, (fit_idx, predict_idx) in enumerate(
            folds, start=1
        ):
            fold_model = deepcopy(final_pipeline)

            fold_model.fit(
                X.iloc[fit_idx],
                y.iloc[fit_idx],
            )

            oof.loc[predict_idx, name] = (
                fold_model.predict_proba(
                    X.iloc[predict_idx]
                )[:, 1]
            )

            print(
                f"  Fold {fold_num}: "
                f"{len(fit_idx)} train, "
                f"{len(predict_idx)} predicted"
            )

        valid_predictions[name] = (
            final_pipeline.predict_proba(
                X.iloc[valid_indices]
            )[:, 1]
        )

    # The earliest chronological block has no OOF predictions.
    complete_oof = oof.dropna()

    if complete_oof.empty:
        raise ValueError("No complete OOF rows generated")

    stack_y = y.loc[complete_oof.index]

    print(
        f"\nStacking training games: {len(complete_oof):,}"
    )

    # Compare small MLP architectures using an internal
    # chronological split of the OOF prediction matrix.
    ordered_indices = complete_oof.index.to_numpy()
    cutoff = int(len(ordered_indices) * 0.8)

    fit_ids = ordered_indices[:cutoff]
    check_ids = ordered_indices[cutoff:]

    candidate_layers = [(4,), (8,), (8, 4)]
    best_layers = None
    best_loss = float("inf")

    for layers in candidate_layers:
        trial = make_mlp(layers)

        trial.fit(
            complete_oof.loc[fit_ids],
            y.loc[fit_ids],
        )

        probabilities = trial.predict_proba(
            complete_oof.loc[check_ids]
        )[:, 1]

        loss = log_loss(
            y.loc[check_ids],
            np.clip(probabilities, 1e-6, 1 - 1e-6),
        )

        print(f"MLP {layers}: internal log loss {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            best_layers = layers

    mlp = make_mlp(best_layers)
    mlp.fit(complete_oof, stack_y)

    logistic_stack = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=0.1,
            max_iter=2000,
            random_state=42,
        )),
    ])
    logistic_stack.fit(complete_oof, stack_y)

    ensemble_predictions = {
        "simple_average": valid_predictions.mean(axis=1).to_numpy(),
        "logistic_stack": logistic_stack.predict_proba(
            valid_predictions
        )[:, 1],
        "mlp_stack": mlp.predict_proba(
            valid_predictions
        )[:, 1],
    }

    rows = []

    for name in BASE_NAMES:
        rows.append({
            "model": name,
            **metrics(
                y.iloc[valid_indices],
                valid_predictions[name].to_numpy(),
            ),
        })

    for name, probabilities in ensemble_predictions.items():
        rows.append({
            "model": name,
            **metrics(y.iloc[valid_indices], probabilities),
        })

    results = pd.DataFrame(rows).sort_values("log_loss")

    STACK_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": mlp,
            "base_names": BASE_NAMES,
            "features": features,
            "hidden_layers": best_layers,
        },
        STACK_DIR / "mlp_stack.joblib",
    )

    joblib.dump(
        {
            "model": logistic_stack,
            "base_names": BASE_NAMES,
            "features": features,
        },
        STACK_DIR / "logistic_stack.joblib",
    )

    complete_oof.to_csv(
        REPORT_DIR / "stacking_oof_predictions.csv",
        index_label="dataset_index",
    )

    results.to_csv(
        REPORT_DIR / "stacking_validation_results.csv",
        index=False,
    )

    print("\n=== STACKING VALIDATION RESULTS ===")
    print(results.to_string(index=False))
    print(f"\nSelected MLP architecture: {best_layers}")


if __name__ == "__main__":
    main()
