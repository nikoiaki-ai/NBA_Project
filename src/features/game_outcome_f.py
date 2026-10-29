
from pathlib import Path
from src.features.elo_features import (
    add_elo_features,
    load_elo_params,
)

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/raw/team_game_logs"
OUTPUT_PATH = ROOT / "data/processed/game_outcomes.parquet"

ROLLING_STATS = [
    #Scoring & Shooting
    "PTS",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",

    #Rebounds
    "OREB",
    "DREB",
    "REB",

    #Playmaking & Defense
    "STL",
    "AST",
    "BLK",

    #Negatives
    "TOV",
    "PF",

    #Result & Margin
    "PLUS_MINUS",
    "win",

    #Derived Metrics
    "off_rating",
    "def_rating",
    "net_rating",
    "pace",
    "efg_pct",
    "tov_pct",
    "orb_pct",
    "ft_rate",

]

WINDOWS = [5, 10]


def load_logs():
    paths = sorted(RAW_DIR.glob("team_game_logs_*.parquet"))

    if not paths:
        raise FileNotFoundError(
            f"No team game logs found in {RAW_DIR}"
        )

    df = pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
    )

    df["GAME_ID"] = (
        df["GAME_ID"].astype(str).str.zfill(10)
    )
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["TEAM_ID"] = df["TEAM_ID"].astype("int64")

    df = df.sort_values(
        ["GAME_DATE", "GAME_ID", "TEAM_ID"]
    ).reset_index(drop=True)

    if df.duplicated(["GAME_ID", "TEAM_ID"]).any():
        raise ValueError("Duplicate team-game rows")

    df["is_home"] = df["MATCHUP"].str.contains(
        "vs.", regex=False, na=False
    )

    if df["MATCHUP"].str.contains("@", regex=False, na=False).sum() + df["is_home"].sum() != len(df):
        raise ValueError("Unrecognized matchup format")

    df["win"] = df["WL"].map({"W": 1, "L": 0})

    if df["win"].isna().any():
        raise ValueError("Missing win/loss labels")

    return df


def add_team_features(df):
    df = df.copy()

    elo_params = load_elo_params()

    df = add_elo_features(
        df,
        elo_params,
    )

    possessions = 0.96 * (
    df["FGA"]
    + df["TOV"]
    + 0.44 * df["FTA"]
    - df["OREB"]
    )

    df["estimated_possessions"] = possessions.where(
        possessions > 0
    )

    df["off_rating"] = (
        100 * df["PTS"] / df["estimated_possessions"]
    )

    # Obtain opponent points through the two-row game pairing.
    game_totals = df.groupby("GAME_ID")[
        ["PTS", "estimated_possessions", "DREB"]
    ].transform("sum")

    game_counts = df.groupby("GAME_ID")["TEAM_ID"].transform("size")

    df["opp_pts"] = (
        game_totals["PTS"] - df["PTS"]
    ).where(game_counts == 2)

    df["opp_possessions"] = (
        game_totals["estimated_possessions"]
        - df["estimated_possessions"]
    ).where(game_counts == 2)

    df["opp_dreb"] = (
        game_totals["DREB"]
        - df["DREB"]
    ).where(game_counts == 2)

    df["def_rating"] = (
        100 * df["opp_pts"] / df["opp_possessions"]
    )
    df["net_rating"] = (
        df["off_rating"] - df["def_rating"]
    )
        
    game_minutes = df["MIN"] / 5

    df["pace"] = (
        df["estimated_possessions"]
        * 48
        / game_minutes
    )

    #Four Factors style stats
    df["efg_pct"] = (
        df["FGM"]
        + 0.5 * df["FG3M"]
    ) / df["FGA"].replace(0, np.nan)

    df["tov_pct"] = (
        df["TOV"]
        / (
            df["FGA"]
            + 0.44 * df["FTA"]
            + df["TOV"]
        ).replace(0, np.nan)
    )

    df["orb_pct"] = (
        df["OREB"]
        / (
            df["OREB"]
            + df["opp_dreb"]
        ).replace(0, np.nan)
    )

    df["ft_rate"] = (
        df["FTA"]
        / df["FGA"].replace(0, np.nan)
    )

    # Sort each team's games chronologically.
    df = df.sort_values(
        ["TEAM_ID", "GAME_DATE", "GAME_ID"]
    ).copy()

    groups = df.groupby("TEAM_ID", sort=False)

    # Current game must not enter its own rolling features.
    for stat in ROLLING_STATS:
        for window in WINDOWS:
            col = f"{stat.lower()}_avg_{window}"

            df[col] = groups[stat].transform(
                lambda s: s.shift(1).rolling(
                    window, min_periods=1
                ).mean()
            )

    df["games_prior"] = groups.cumcount()

    df["rest_days"] = (
        groups["GAME_DATE"].diff().dt.days - 1
    ).clip(lower=0)

    # Rest is known before tipoff.
    # Season is retained as metadata for splitting.
    return df


def build_matchups(df):
    # Exactly two teams per completed NBA game.
    counts = df.groupby("GAME_ID").size()

    valid_ids = counts[counts == 2].index

    print(
        f"Complete two-team games: {len(valid_ids):,}"
    )
    print(
        f"Excluded incomplete games: {(counts != 2).sum():,}"
    )

    df = df[df["GAME_ID"].isin(valid_ids)].copy()
    # Keep only games with exactly one identified home team.
    home_counts = df.groupby("GAME_ID")["is_home"].sum()

    valid_home_ids = home_counts[
        home_counts == 1
    ].index

    excluded_games = home_counts[
        home_counts != 1
    ]

    print(
        f"Games with ambiguous home/away assignments: "
        f"{len(excluded_games):,}"
    )

    df = df[
        df["GAME_ID"].isin(valid_home_ids)
    ].copy()

    home = df[df["is_home"]].copy()
    away = df[~df["is_home"]].copy()

    if home["GAME_ID"].duplicated().any():
        raise ValueError("Duplicate home-game entries")

    if away["GAME_ID"].duplicated().any():
        raise ValueError("Duplicate away-game entries")

    # Only retain metadata, target, and explicitly
    # selected pregame features.
    feature_cols = [
        col for col in df.columns
        if (
            "_avg_" in col
            or col in ["games_prior", "rest_days", "team_pregame_elo", "elo_win_prob"]
        )
    ]

    home_cols = [
        "GAME_ID", "GAME_DATE", "SEASON",
        "TEAM_ID", "TEAM_ABBREVIATION", "win"
    ] + feature_cols

    away_cols = [
        "GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"
    ] + feature_cols

    home = home[home_cols].rename(
        columns={
            "TEAM_ID": "home_team_id",
            "TEAM_ABBREVIATION": "home_team",
            "win": "home_win",
            **{
                col: f"home_{col}"
                for col in feature_cols
            },
        }
    )

    away = away[away_cols].rename(
        columns={
            "TEAM_ID": "away_team_id",
            "TEAM_ABBREVIATION": "away_team",
            **{
                col: f"away_{col}"
                for col in feature_cols
            },
        }
    )

    games = home.merge(
        away,
        on="GAME_ID",
        how="inner",
        validate="one_to_one",
    )

    # Verify that both teams have opposite results.
    away_results = df.loc[
        ~df["is_home"], ["GAME_ID", "win"]
    ].set_index("GAME_ID")["win"]

    away_wins = games["GAME_ID"].map(away_results)

    if not (games["home_win"] + away_wins == 1).all():
        raise ValueError(
            "Home and away results are inconsistent"
        )

    if len(games) != len(valid_home_ids):
        raise ValueError(
            "Home/away pairing failed for some games"
        )

    for col in feature_cols:
        games[f"diff_{col}"] = (
            games[f"home_{col}"]
            - games[f"away_{col}"]
        )

    games = games.sort_values(
        ["GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    return games


def validate(games):
    assert games["GAME_ID"].is_unique
    assert games["home_win"].isin([0, 1]).all()
    assert (
        games["home_team_id"] != games["away_team_id"]
    ).all()

    # Ensure no current-game box-score columns
    # accidentally entered the feature matrix.
    forbidden = [
        "home_PTS", "away_PTS",
        "home_WL", "away_WL",
        "home_PLUS_MINUS", "away_PLUS_MINUS",
    ]

    assert not any(
        col in games.columns for col in forbidden
    )

    print("\nGames by season:")
    print(games.groupby("SEASON").size())

    print("\nHome win rate:")
    print(round(games["home_win"].mean(), 4))

    print("\nMissing values (top 10):")
    print(
        games.isna().mean()
        .sort_values(ascending=False)
        .head(10)
    )


def main():
    logs = load_logs()
    featured = add_team_features(logs)
    games = build_matchups(featured)

    validate(games)

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    games.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Shape: {games.shape}")


if __name__ == "__main__":
    main()
