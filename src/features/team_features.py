
"""Generate leakage-safe NBA team and opponent features."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "team_game_logs"
FEATURE_DIR = PROJECT_ROOT / "data" / "features"

SEASONS = [
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
]


def load_team_logs() -> pd.DataFrame:
    """Load and combine team game logs across seasons."""

    frames = []

    for season in SEASONS:
        filename = f"team_game_logs_{season.replace('-', '_')}.parquet"
        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_parquet(path)
        df["SEASON"] = season
        frames.append(df)

    logs = pd.concat(frames, ignore_index=True)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

    return logs.sort_values(
        ["SEASON", "TEAM_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)


def build_team_features(logs: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly pregame rolling team statistics."""

    df = logs.copy()

    # Estimated possessions:
    # FGA + 0.44 * FTA - offensive rebounds + turnovers
    required = ["FGA", "FTA", "OREB", "TOV", "PTS", "WL"]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["possessions_est"] = (
        df["FGA"]
        + 0.44 * df["FTA"]
        - df["OREB"]
        + df["TOV"]
    )

    df["win"] = (df["WL"] == "W").astype(int)

    # Attach opposing team's game totals
    opponent = df[
        ["SEASON", "GAME_ID", "TEAM_ID", "PTS", "possessions_est"]
    ].rename(
        columns={
            "TEAM_ID": "OPP_TEAM_ID",
            "PTS": "pts_allowed",
            "possessions_est": "opp_possessions_est",
        }
    )

    df = df.merge(
        opponent,
        on=["SEASON", "GAME_ID"],
        how="inner",
        validate="many_to_many",
    )

    # Remove self-match introduced by the merge
    df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]].copy()

    # Approximate game possessions as the average
    # of both teams' estimated possessions.
    df["game_possessions_est"] = (
        df["possessions_est"]
        + df["opp_possessions_est"]
    ) / 2

    df["def_rating_est"] = (
        100 * df["pts_allowed"]
        / df["game_possessions_est"].replace(0, np.nan)
    )

    # Without minutes, possession count is only a pace proxy.
    df["pace_proxy"] = df["game_possessions_est"]

    df = df.sort_values(
        ["SEASON", "TEAM_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    group = df.groupby(["SEASON", "TEAM_ID"], sort=False)

    rolling_metrics = {
        "PTS": "pts_avg_10",
        "pts_allowed": "pts_allowed_10",
        "def_rating_est": "def_rating_10",
        "pace_proxy": "pace_10",
        "win": "win_pct_10",
    }

    for source, target in rolling_metrics.items():
        df[target] = group[source].transform(
            lambda s: s.shift(1).rolling(
                window=10,
                min_periods=3,
            ).mean()
        )

    df["previous_game_date"] = group["GAME_DATE"].shift(1)

    df["rest_days"] = (
        df["GAME_DATE"] - df["previous_game_date"]
    ).dt.days - 1

    # Attach opponent's PRE-game features to the current team
    opponent_features = df[
        [
            "SEASON",
            "GAME_ID",
            "TEAM_ID",
            "pts_allowed_10",
            "def_rating_10",
            "pace_10",
            "win_pct_10",
            "rest_days",
        ]
    ].rename(
        columns={
            "TEAM_ID": "OPP_TEAM_ID",
            "pts_allowed_10": "opp_pts_allowed_10",
            "def_rating_10": "opp_def_rating_10",
            "pace_10": "opp_pace_10",
            "win_pct_10": "opp_win_pct_10",
            "rest_days": "opp_rest_days",
        }
    )

    df = df.merge(
        opponent_features,
        on=["SEASON", "GAME_ID", "OPP_TEAM_ID"],
        how="left",
        validate="one_to_one",
    )

    output_columns = [
        "SEASON",
        "GAME_ID",
        "GAME_DATE",
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

    result = df[output_columns].copy()

    if result.duplicated(["SEASON", "GAME_ID", "TEAM_ID"]).any():
        raise ValueError("Duplicate team-game features detected")

    return result


def main():
    logs = load_team_logs()
    features = build_team_features(logs)

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FEATURE_DIR / "team_matchup_features.parquet"
    features.to_parquet(output_path, index=False)

    print(f"Saved {len(features):,} team-game feature rows")
    print(f"Output: {output_path}")

    print("\nMissing feature percentages:")
    print(
        (
            features.isna().mean() * 100
        ).round(2).to_string()
    )

    print("\nFeature preview:")
    print(features.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
