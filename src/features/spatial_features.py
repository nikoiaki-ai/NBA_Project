
"""
Uses historical player shots and free throws, historical
league baselines, and empirical-Bayes-style smoothing.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SEASONS = [
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
]


def load_all_seasons():
    shot_frames = []
    player_frames = []

    for season in SEASONS:
        suffix = season.replace("-", "_")

        shot_path = (
            ROOT / "data/raw/shot_charts"
            / f"shot_charts_{suffix}.parquet"
        )

        player_path = (
            ROOT / "data/raw/player_game_logs"
            / f"player_game_logs_{suffix}.parquet"
        )

        shots = pd.read_parquet(shot_path)
        players = pd.read_parquet(player_path)

        shots["SEASON"] = season
        players["SEASON"] = season

        shot_frames.append(shots)
        player_frames.append(players)

        print(
            f"{season}: "
            f"{len(shots):,} shots, "
            f"{len(players):,} player-games"
        )

    all_shots = pd.concat(
        shot_frames,
        ignore_index=True,
    )

    all_players = pd.concat(
        player_frames,
        ignore_index=True,
    )

    for df in (all_shots, all_players):
        df["GAME_ID"] = (
            df["GAME_ID"].astype(str).str.zfill(10)
        )

        df["GAME_DATE"] = pd.to_datetime(
            df["GAME_DATE"]
        )

    all_shots = all_shots.sort_values(
        ["GAME_DATE", "GAME_ID", "GAME_EVENT_ID"]
    ).reset_index(drop=True)

    all_players = all_players.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    return all_shots, all_players

OUTPUT_PATH = (
    ROOT / "data/features"
    / "spatial_epaa_v2_all_seasons.parquet"
)

WINDOW = 10
STANDARD_ATTEMPTS = 100
SELECTION_PRIOR = 30
ACCURACY_PRIOR = 20

REGIONS = [
    "restricted",
    "paint",
    "midrange",
    "left_corner_3",
    "right_corner_3",
    "above_break_3",
    "free_throw",
]

POINT_VALUES = {
    "restricted": 2,
    "paint": 2,
    "midrange": 2,
    "left_corner_3": 3,
    "right_corner_3": 3,
    "above_break_3": 3,
    "free_throw": 1,
}

ZONE_MAP = {
    "Restricted Area": "restricted",
    "In The Paint (Non-RA)": "paint",
    "Mid-Range": "midrange",
    "Left Corner 3": "left_corner_3",
    "Right Corner 3": "right_corner_3",
    "Above the Break 3": "above_break_3",
}


def build_epaa_features(shots, logs):

    keys = ["GAME_ID", "PLAYER_ID", "TEAM_ID"]

    for data in (shots, logs):
        data["GAME_ID"] = (
            data["GAME_ID"].astype(str).str.zfill(10)
        )
        data["GAME_DATE"] = pd.to_datetime(
            data["GAME_DATE"]
        ).dt.normalize()

    # 1. Classify shots into regions.
    shots["region"] = shots["SHOT_ZONE_BASIC"].map(
        ZONE_MAP
    )

    unknown = shots.loc[
        shots["region"].isna(),
        "SHOT_ZONE_BASIC"
    ].value_counts()

    print("Excluded/unrecognized shot zones:")
    print(unknown)

    # Only backcourt shots are intentionally excluded.
    unexpected = set(unknown.index) - {"Backcourt"}
    if unexpected:
        raise ValueError(
            f"Unexpected shot regions: {unexpected}"
        )

    shots = shots.dropna(subset=["region"]).copy()

    # 2. Aggregate regional attempts and makes.
    grouped = (
        shots.groupby(keys + ["region"])
        .agg(
            attempts=("SHOT_ATTEMPTED_FLAG", "sum"),
            makes=("SHOT_MADE_FLAG", "sum"),
        )
        .reset_index()
    )

    wide = grouped.pivot(
        index=keys,
        columns="region",
        values=["attempts", "makes"],
    )

    wide.columns = [
        f"{'att' if stat == 'attempts' else 'made'}_{region}"
        for stat, region in wide.columns
    ]
    wide = wide.reset_index()

    # 3. Include free throws and zero-FGA games.
    df = logs[
        keys + ["GAME_DATE", "FGA", "FGM", "FTA", "FTM"]
    ].merge(
        wide,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    for region in REGIONS[:-1]:
        for prefix in ("att", "made"):
            col = f"{prefix}_{region}"
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)

    df["att_free_throw"] = df["FTA"].fillna(0)
    df["made_free_throw"] = df["FTM"].fillna(0)

    raw_cols = [
        f"{prefix}_{region}"
        for prefix in ("att", "made")
        for region in REGIONS
    ]

    # Backcourt shots explain any difference from total FGA.
    backcourt = (
        shots  # backcourt was excluded above
    )
    regional_fga = df[
        [f"att_{r}" for r in REGIONS[:-1]]
    ].sum(axis=1)

    KNOWN_EXCEPTIONS = {
    ("0022200271", 202322),
    ("0022300078", 1628991),
    ("0022300465", 202684),
    ("0022500792", 1631095),
    }

    excess = regional_fga > df["FGA"]

    if excess.any():
        exceptions = df.loc[
            excess, ["GAME_ID", "PLAYER_ID", "FGA"]
        ].copy()

        exceptions["regional_fga"] = regional_fga.loc[excess]

        for _, row in exceptions.iterrows():
            key = (
                str(row["GAME_ID"]).zfill(10),
                int(row["PLAYER_ID"]),
            )

        if (
            key not in KNOWN_EXCEPTIONS
            or row["regional_fga"] - row["FGA"] != 1
        ):
            raise ValueError(
                f"Unexpected regional FGA mismatch: {key}"
            )

        print(
            f"Accepted {len(exceptions)} documented "
            "regional FGA discrepancies."
        )

    # 4. League baselines from previous calendar dates.
    daily = (
        df.groupby("GAME_DATE", as_index=False)[raw_cols]
        .sum()
        .sort_values("GAME_DATE")
    )

    for col in raw_cols:
        daily[f"prior_{col}"] = (
            daily[col].cumsum() - daily[col]
        )

    league_prior_attempts = sum(
        daily[f"prior_att_{r}"] for r in REGIONS
    )

    for region in REGIONS:
        att = daily[f"prior_att_{region}"]
        made = daily[f"prior_made_{region}"]

        daily[f"league_select_{region}"] = (
            att / league_prior_attempts.replace(0, np.nan)
        )

        daily[f"league_accuracy_{region}"] = (
            made / att.replace(0, np.nan)
        )

    league_cols = [
        f"league_{metric}_{r}"
        for metric in ("select", "accuracy")
        for r in REGIONS
    ]

    df = df.merge(
        daily[["GAME_DATE"] + league_cols],
        on="GAME_DATE",
        how="left",
        validate="many_to_one",
    )

    # 5. Previous 10 player appearances only.
    df = df.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    grouped_players = df.groupby(
        "PLAYER_ID", sort=False
    )

    for col in raw_cols:
        df[f"hist10_{col}"] = (
    grouped_players[col].transform(
        lambda s: s.shift(1).rolling(
            WINDOW, min_periods=1
        ).sum()
    ).fillna(0)
)

    total_hist_attempts = sum(
        df[f"hist10_att_{r}"] for r in REGIONS
    )

    # 6. Smoothed shot selection and accuracy.
    expected_player = 0
    expected_league = 0

    for region in REGIONS:
        att = df[f"hist10_att_{region}"]
        made = df[f"hist10_made_{region}"]

        league_select = df[
            f"league_select_{region}"
        ]
        league_accuracy = df[
            f"league_accuracy_{region}"
        ]

        # Dirichlet-style smoothing for shot selection.
        player_select = (
            att + SELECTION_PRIOR * league_select
        ) / (
            total_hist_attempts + SELECTION_PRIOR
        )

        # Beta-style smoothing for shot accuracy.
        player_accuracy = (
            made + ACCURACY_PRIOR * league_accuracy
        ) / (
            att + ACCURACY_PRIOR
        )

        df[f"v2_select_{region}"] = player_select
        df[f"v2_accuracy_{region}"] = player_accuracy

        value = POINT_VALUES[region]

        expected_player = expected_player + (
            player_select * player_accuracy * value
        )

        expected_league = expected_league + (
            league_select * league_accuracy * value
        )

    # 7. Standardized expected points and EPAA.
    df["v2_expected_points_100"] = (
        STANDARD_ATTEMPTS * expected_player
    )
    df["v2_league_points_100"] = (
        STANDARD_ATTEMPTS * expected_league
    )
    df["spatial_epaa_v2_100"] = (
        df["v2_expected_points_100"]
        - df["v2_league_points_100"]
    )

    df["v2_prior_attempts_10"] = (
        total_hist_attempts
    )

    feature_cols = [
        col for col in df.columns
        if col.startswith("v2_")
        or col == "spatial_epaa_v2_100"
    ]

    output = df[
        keys + ["GAME_DATE"] + feature_cols
    ].copy()

    selection_cols = [
    f"v2_select_{r}" for r in REGIONS
]

    valid = output[selection_cols].notna().all(axis=1)

    selection_sums = (
        output.loc[valid, selection_cols].sum(axis=1)
    )

    print("Selection sums:")
    print(selection_sums.describe())

    assert np.allclose(selection_sums, 1.0, atol=1e-8)

    print("All shot-selection probabilities sum to 1.")

    if output.duplicated(keys).any():
        raise ValueError("Duplicate player-game keys")

    print("\nSaved:", OUTPUT_PATH)
    print("Rows:", len(output))

    print("\nEPAA v2 summary:")
    print(
        output[
            [
                "v2_expected_points_100",
                "v2_league_points_100",
                "spatial_epaa_v2_100",
            ]
        ].describe().round(3)
    )

    print("\nMissing values (%):")
    print(
        output[feature_cols]
        .isna().mean().mul(100).round(2)
    )

    return output


def main():
    shots, logs = load_all_seasons()

    output = build_epaa_features(
        shots.copy(),
        logs.copy()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    output.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()