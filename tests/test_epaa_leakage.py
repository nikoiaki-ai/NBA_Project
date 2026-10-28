
import numpy as np
import pandas as pd

from src.features.spatial_features import (
    load_all_seasons,
    build_epaa_features,
)


def test_current_game_and_future_leakage():
    shots, logs = load_all_seasons()

    # Limit the test to a chronological sample for speed.
    dates = sorted(pd.to_datetime(logs["GAME_DATE"]).unique())
    cutoff = dates[30]

    logs = logs.loc[
        pd.to_datetime(logs["GAME_DATE"]) <= cutoff
    ].copy()

    game_ids = set(logs["GAME_ID"].astype(str).str.zfill(10))

    shots = shots.loc[
        shots["GAME_ID"].astype(str).str.zfill(10).isin(game_ids)
    ].copy()

    # Run the original feature pipeline.
    baseline = build_epaa_features(
        shots.copy(),
        logs.copy()
    )

    # Choose a game near the end of the sample.
    target = logs.sort_values(
        ["GAME_DATE", "GAME_ID"]
    ).iloc[-1]

    target_game = target["GAME_ID"]
    target_player = target["PLAYER_ID"]
    target_date = pd.to_datetime(target["GAME_DATE"])

    # Modify current-game shooting outcomes.
    modified_shots = shots.copy()
    mask = (
        (modified_shots["GAME_ID"] == target_game)
        & (modified_shots["PLAYER_ID"] == target_player)
    )

    modified_shots.loc[
        mask, "SHOT_MADE_FLAG"
    ] = 0

    modified_logs = logs.copy()
    log_mask = (
        (modified_logs["GAME_ID"] == target_game)
        & (modified_logs["PLAYER_ID"] == target_player)
    )

    modified_logs.loc[log_mask, "FTM"] = 0

    # Recalculate features using modified outcomes.
    modified = build_epaa_features(
        modified_shots,
        modified_logs
    )

    keys = ["GAME_ID", "PLAYER_ID", "TEAM_ID"]

    feature_cols = [
        c for c in baseline.columns
        if c.startswith("v2_")
        or c == "spatial_epaa_v2_100"
    ]

    comparison = baseline.merge(
        modified,
        on=keys,
        suffixes=("_original", "_modified"),
        validate="one_to_one",
    )

    # The target player's current-game features must
    # remain unchanged.
    current = comparison.loc[
        (comparison["GAME_ID"] == target_game)
        & (comparison["PLAYER_ID"] == target_player)
    ]

    assert len(current) == 1

    for col in feature_cols:
        assert np.allclose(
            current[f"{col}_original"],
            current[f"{col}_modified"],
            equal_nan=True,
        ), f"Current-game leakage detected in {col}"

    print("PASS: Current-game outcomes do not affect pregame EPAA")

    # No features from earlier dates should change.
    earlier = comparison.loc[
        pd.to_datetime(comparison["GAME_DATE_original"])
        < target_date
    ]

    for col in feature_cols:
        assert np.allclose(
            earlier[f"{col}_original"],
            earlier[f"{col}_modified"],
            equal_nan=True,
        ), f"Historical leakage detected in {col}"

    print("PASS: Future outcomes do not affect earlier EPAA")


if __name__ == "__main__":
    test_current_game_and_future_leakage()
