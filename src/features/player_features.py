
"""Build leakage-free historical player features."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "player_model_dataset.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "features"
    / "player_features.parquet"
)

ROLLING_STATS = [
    "PTS", "MIN", "FGA", "FG3A", "FTA",
    "FGM", "FG3M", "FTM",
    "AST", "REB", "TOV",
]

SEASON_STATS = ["PTS", "MIN", "FGA", "FTA"]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean types and establish chronological player history."""

    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["win"] = df["WL"].map({"W": 1, "L": 0}).astype("Int64")

    # nba_api may provide minutes as numeric strings.
    for stat in ROLLING_STATS:
        df[stat] = pd.to_numeric(df[stat], errors="coerce")

    keys = ["SEASON", "GAME_ID", "PLAYER_ID"]

    if df.duplicated(keys).any():
        raise ValueError("Duplicate player-game rows detected")

    df = df.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate recent form across season boundaries."""

    df = df.copy()

    grouped = df.groupby("PLAYER_ID", sort=False)

    df["career_games_prior"] = grouped.cumcount()

    for stat in ROLLING_STATS:
        for window in [5, 10]:
            df[f"{stat.lower()}_avg_{window}"] = (
                grouped[stat].transform(
                    lambda s: (
                        s.shift(1)
                        .rolling(window, min_periods=1)
                        .mean()
                    )
                )
            )

    df["pts_std_10"] = grouped["PTS"].transform(
        lambda s: (
            s.shift(1)
            .rolling(10, min_periods=2)
            .std()
        )
    )

    return df


def add_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate season-to-date averages without current-game data."""

    df = df.copy()

    grouped = df.groupby(
        ["PLAYER_ID", "SEASON"], sort=False
    )

    df["season_games_prior"] = grouped.cumcount()

    for stat in SEASON_STATS:
        df[f"{stat.lower()}_season_avg"] = (
            grouped[stat].transform(
                lambda s: s.shift(1).expanding().mean()
            )
        )

    return df


def add_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute shooting percentages from prior 10-game totals."""

    df = df.copy()
    grouped = df.groupby("PLAYER_ID", sort=False)

    for stat in ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA"]:
        df[f"{stat.lower()}_sum_10"] = grouped[stat].transform(
            lambda s: (
                s.shift(1)
                .rolling(10, min_periods=1)
                .sum()
            )
        )

    df["fg_pct_10"] = (
        df["fgm_sum_10"]
        / df["fga_sum_10"].replace(0, np.nan)
    )

    df["fg3_pct_10"] = (
        df["fg3m_sum_10"]
        / df["fg3a_sum_10"].replace(0, np.nan)
    )

    df["ft_pct_10"] = (
        df["ftm_sum_10"]
        / df["fta_sum_10"].replace(0, np.nan)
    )

    return df


def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rest and home-court indicators."""

    df = df.copy()

    previous_date = df.groupby("PLAYER_ID")[
        "GAME_DATE"
    ].shift(1)

    df["player_rest_days"] = (
        (df["GAME_DATE"] - previous_date).dt.days - 1
    )

    df["is_home"] = (
        df["MATCHUP"].str.contains("vs.", regex=False)
    ).astype(int)

    return df


def build_features() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)

    df = prepare_data(df)
    df = add_rolling_features(df)
    df = add_season_features(df)
    df = add_efficiency_features(df)
    df = add_schedule_features(df)

    df = df.sort_values(
        ["GAME_DATE", "GAME_ID", "PLAYER_ID"]
    ).reset_index(drop=True)

    return df


def main():
    df = build_features()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved {len(df):,} player-game rows")
    print(f"Output: {OUTPUT_PATH}")

    feature_cols = [
        "pts_avg_5",
        "pts_avg_10",
        "min_avg_5",
        "pts_season_avg",
        "fg_pct_10",
        "player_rest_days",
        "opp_def_rating_10",
    ]

    print("\nMissing values (%):")
    print(
        (df[feature_cols].isna().mean() * 100)
        .round(2)
        .to_string()
    )

    print("\nSample:")
    print(
        df[
            ["PLAYER_NAME", "GAME_DATE", "PTS",
             "pts_avg_5", "pts_season_avg"]
        ].tail(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
