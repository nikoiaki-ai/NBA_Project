
"""Advanced pregame features for player-points prediction."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/features/player_features.parquet"
OUTPUT = ROOT / "data/features/player_features_advanced.parquet"


def build_advanced_features(df):
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df = df.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    # Derived shooting statistics for each historical game.
    df["FG2M"] = df["FGM"] - df["FG3M"]
    df["FG2A"] = df["FGA"] - df["FG3A"]

    stats = [
        "PTS", "MIN", "FGA", "FTA",
        "FG2M", "FG2A", "FG3M", "FG3A",
    ]

    grouped = df.groupby("PLAYER_ID", sort=False)

    # Historical 10-game totals, excluding the current game.
    for stat in stats:
        df[f"hist_{stat.lower()}_10"] = grouped[stat].transform(
            lambda s: s.shift(1).rolling(
                10, min_periods=1
            ).sum()
        )

    def safe_divide(a, b):
        return a / b.replace(0, np.nan)

    minutes = df["hist_min_10"]
    fga = df["hist_fga_10"]
    fta = df["hist_fta_10"]

    df["fga_per_min_10"] = safe_divide(fga, minutes)
    df["fta_per_min_10"] = safe_divide(fta, minutes)

    df["three_point_rate_10"] = safe_divide(
        df["hist_fg3a_10"], fga
    )

    df["ts_pct_10"] = safe_divide(
        df["hist_pts_10"],
        2 * (fga + 0.44 * fta),
    )

    # League shooting totals grouped by calendar date.
    # This prevents games played on the same day from
    # contributing information to each other's features.
    daily = (
        df.groupby(["SEASON", "GAME_DATE"], as_index=False)
        [["FG2M", "FG2A", "FG3M", "FG3A"]]
        .sum()
        .sort_values(["SEASON", "GAME_DATE"])
    )

    # League averages from earlier dates in the same season.
    for stat in ["FG2M", "FG2A", "FG3M", "FG3A"]:
        daily[f"prior_{stat.lower()}"] = (
            daily.groupby("SEASON")[stat].cumsum()
            - daily[stat]
        )

    daily["league_2p_pct"] = safe_divide(
        daily["prior_fg2m"], daily["prior_fg2a"]
    )
    daily["league_3p_pct"] = safe_divide(
        daily["prior_fg3m"], daily["prior_fg3a"]
    )

    df = df.merge(
        daily[
            ["SEASON", "GAME_DATE",
             "league_2p_pct", "league_3p_pct"]
        ],
        on=["SEASON", "GAME_DATE"],
        how="left",
        validate="many_to_one",
    )

    player_2p_pct = safe_divide(
        df["hist_fg2m_10"], df["hist_fg2a_10"]
    )
    player_3p_pct = safe_divide(
        df["hist_fg3m_10"], df["hist_fg3a_10"]
    )

    # Shooting points above league-average efficiency
    # across the previous 10-game shot volume.
    df["epaa_10"] = (
    2 * (
        df["hist_fg2m_10"]
        - df["hist_fg2a_10"] * df["league_2p_pct"]
    )
    + 3 * (
        df["hist_fg3m_10"]
        - df["hist_fg3a_10"] * df["league_3p_pct"]
    )
)
    total_attempts = (
        df["hist_fg2a_10"] + df["hist_fg3a_10"]
        )
    
    df["epaa_per_attempt_10"] = (
        df["epaa_10"] / total_attempts.replace(0, np.nan)
        )
    

    # Keep only the final engineered features.
    df = df.drop(
        columns=[
            c for c in df.columns
            if c.startswith("hist_")
            or c.startswith("prior_")
        ]
    )

    return df.sort_values(
        ["GAME_DATE", "GAME_ID", "PLAYER_ID"]
    ).reset_index(drop=True)

def main():
    df = pd.read_parquet(INPUT)
    original_rows = len(df)

    df = build_advanced_features(df)

    if len(df) != original_rows:
        raise ValueError("Row count changed during feature creation")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)

    features = [
        "fga_per_min_10",
        "fta_per_min_10",
        "three_point_rate_10",
        "ts_pct_10",
        "epaa_10",
    ]

    print(f"Saved {len(df):,} player-game rows")
    print(f"Output: {OUTPUT}")

    print("\nFeature summary:")
    print(df[features].describe().round(3))

    print("\nMissing values (%):")
    print((df[features].isna().mean() * 100).round(2))


if __name__ == "__main__":
    main()
