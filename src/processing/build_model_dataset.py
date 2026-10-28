
"""Combine historical player logs with pregame team features."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAYER_DIR = PROJECT_ROOT / "data" / "raw" / "player_game_logs"
FEATURE_PATH = (
    PROJECT_ROOT / "data" / "features"
    / "team_matchup_features.parquet"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "player_model_dataset.parquet"
)

SEASONS = [
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
]


def load_player_logs() -> pd.DataFrame:
    """Combine player logs while preserving season identifiers."""

    frames = []

    for season in SEASONS:
        path = (
            PLAYER_DIR
            / f"player_game_logs_{season.replace('-', '_')}.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_parquet(path)
        df["SEASON"] = season
        frames.append(df)

    players = pd.concat(frames, ignore_index=True)
    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"])

    df["win"] = df["WL"].map({"W": 1, "L": 0}).astype("Int64")

    keys = ["SEASON", "GAME_ID", "PLAYER_ID"]

    if players.duplicated(keys).any():
        raise ValueError("Duplicate player-game records detected")

    return players


def build_model_dataset() -> pd.DataFrame:
    """Join player-game records with pregame matchup context."""

    players = load_player_logs()
    team_features = pd.read_parquet(FEATURE_PATH)

    join_keys = ["SEASON", "GAME_ID", "TEAM_ID"]

    if team_features.duplicated(join_keys).any():
        raise ValueError("Duplicate team-game feature keys")

    matchup_columns = [
        "SEASON",
        "GAME_ID",
        "TEAM_ID",
        "OPP_TEAM_ID",
        "pts_avg_10",
        "opp_pts_allowed_10",
        "opp_def_rating_10",
        "opp_pace_10",
        "opp_win_pct_10",
        "rest_days",
        "opp_rest_days",
    ]

    model_df = players.merge(
        team_features[matchup_columns],
        on=join_keys,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = (model_df["_merge"] != "both").sum()

    if unmatched:
        raise ValueError(
            f"{unmatched} player-games lack a team matchup record"
        )

    model_df = model_df.drop(columns="_merge")

    if len(model_df) != len(players):
        raise ValueError("Row count changed during matchup join")

    model_df = model_df.sort_values(
        ["GAME_DATE", "GAME_ID", "PLAYER_ID"]
    ).reset_index(drop=True)

    return model_df


def main():
    model_df = build_model_dataset()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved {len(model_df):,} player-game rows")
    print(f"Output: {OUTPUT_PATH}")

    print("\nRows per season:")
    print(
        model_df.groupby("SEASON")
        .size()
        .to_string()
    )

    print("\nMissing matchup features (%):")
    columns = [
        "opp_pts_allowed_10",
        "opp_def_rating_10",
        "opp_pace_10",
        "opp_win_pct_10",
        "rest_days",
        "opp_rest_days",
    ]

    print(
        (model_df[columns].isna().mean() * 100)
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
