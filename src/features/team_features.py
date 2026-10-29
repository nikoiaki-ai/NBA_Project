"""Generate leakage-safe NBA team and opponent features."""

from pathlib import Path
from src.features.elo_features import (
    add_elo_features,
    load_elo_params,
)

import numpy as np
import pandas as pd


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "team_game_logs"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
)

SEASONS = [
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
]


def load_team_logs() -> pd.DataFrame:
    """Load and combine team logs."""

    frames = []

    for season in SEASONS:
        filename = (
            f"team_game_logs_"
            f"{season.replace('-', '_')}"
            f".parquet"
        )

        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        df = pd.read_parquet(path)

        df["SEASON"] = season

        frames.append(df)

    logs = pd.concat(
        frames,
        ignore_index=True,
    )

    logs["GAME_DATE"] = (
        pd.to_datetime(
            logs["GAME_DATE"]
        )
    )

    logs["GAME_ID"] = (
        logs["GAME_ID"]
        .astype(str)
        .str.zfill(10)
    )

    return logs.sort_values(
        [
            "TEAM_ID",
            "GAME_DATE",
            "GAME_ID",
        ]
    ).reset_index(drop=True)


def build_team_features(
    logs: pd.DataFrame,
) -> pd.DataFrame:
    """Compute leakage-safe team context."""

    df = logs.copy()

    elo_params = load_elo_params()

    df = add_elo_features(
    df,
    elo_params,
    )

    required = [
        "FGA",
        "FTA",
        "OREB",
        "DREB",
        "TOV",
        "PTS",
        "WL",
        "FGM",
        "FG3M",
        "AST",
        "REB",
        "STL",
        "BLK",
        "PF",
    ]

    missing = (
        set(required)
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{missing}"
        )

    df["win"] = (
        df["WL"] == "W"
    ).astype(int)

    # -------------------------
    # Game-level derived stats
    # -------------------------

    df["possessions_est"] = (
        0.96
        * (
            df["FGA"]
            + df["TOV"]
            + 0.44 * df["FTA"]
            - df["OREB"]
        )
    )

    game_counts = (
        df.groupby("GAME_ID")[
            "TEAM_ID"
        ]
        .transform("size")
    )

    totals = (
        df.groupby("GAME_ID")[
            [
                "PTS",
                "possessions_est",
                "DREB",
            ]
        ]
        .transform("sum")
    )

    df["pts_allowed"] = (
        totals["PTS"]
        - df["PTS"]
    ).where(game_counts == 2)

    df["opp_possessions_est"] = (
        totals["possessions_est"]
        - df["possessions_est"]
    ).where(game_counts == 2)

    df["opp_dreb"] = (
        totals["DREB"]
        - df["DREB"]
    ).where(game_counts == 2)

    df["off_rating_est"] = (
        100
        * df["PTS"]
        / df["possessions_est"]
        .replace(0, np.nan)
    )

    df["def_rating_est"] = (
        100
        * df["pts_allowed"]
        / df["opp_possessions_est"]
        .replace(0, np.nan)
    )

    df["net_rating_est"] = (
        df["off_rating_est"]
        - df["def_rating_est"]
    )

    df["pace_proxy"] = (
        (
            df["possessions_est"]
            + df["opp_possessions_est"]
        )
        / 2
    )

    df["efg_pct"] = (
        df["FGM"]
        + 0.5 * df["FG3M"]
    ) / df["FGA"].replace(
        0, np.nan
    )

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
        / df["FGA"].replace(
            0, np.nan
        )
    )

    # -------------------------
    # Chronological history
    # -------------------------

    df = df.sort_values(
        [
            "TEAM_ID",
            "GAME_DATE",
            "GAME_ID",
        ]
    ).reset_index(drop=True)

    # Important:
    # do NOT reset rolling form at season
    # boundaries.
    group = df.groupby(
        "TEAM_ID",
        sort=False,
    )

    rolling_metrics = {
        "PTS": "pts_avg_10",
        "pts_allowed":
            "pts_allowed_10",
        "off_rating_est":
            "off_rating_10",
        "def_rating_est":
            "def_rating_10",
        "net_rating_est":
            "net_rating_10",
        "pace_proxy":
            "pace_10",
        "win":
            "win_pct_10",
        "efg_pct":
            "efg_pct_10",
        "tov_pct":
            "tov_pct_10",
        "orb_pct":
            "orb_pct_10",
        "ft_rate":
            "ft_rate_10",
        "AST":
            "ast_avg_10",
        "REB":
            "reb_avg_10",
        "STL":
            "stl_avg_10",
        "BLK":
            "blk_avg_10",
        "PF":
            "pf_avg_10",
    }

    for source, target in (
        rolling_metrics.items()
    ):
        df[target] = (
            group[source].transform(
                lambda s: (
                    s.shift(1)
                    .rolling(
                        window=10,
                        min_periods=1,
                    )
                    .mean()
                )
            )
        )

    df["team_games_prior"] = (
        group.cumcount()
    )

    previous_date = (
        group["GAME_DATE"]
        .shift(1)
    )

    df["rest_days"] = (
        (
            df["GAME_DATE"]
            - previous_date
        ).dt.days
        - 1
    ).clip(
        lower=0,
        upper=10,
    )

    # -------------------------
    # Opponent pregame context
    # -------------------------

    opponent_features = df[
        [
            "GAME_ID",
            "TEAM_ID",

            "pts_avg_10",
            "pts_allowed_10",

            "off_rating_10",
            "def_rating_10",
            "net_rating_10",

            "pace_10",
            "win_pct_10",

            "efg_pct_10",
            "tov_pct_10",
            "orb_pct_10",
            "ft_rate_10",

            "ast_avg_10",
            "reb_avg_10",
            "stl_avg_10",
            "blk_avg_10",
            "pf_avg_10",

            "rest_days",
            "team_games_prior",
        ]
    ].rename(
        columns={
            "TEAM_ID":
                "OPP_TEAM_ID",

            "pts_avg_10":
                "opp_pts_avg_10",

            "pts_allowed_10":
                "opp_pts_allowed_10",

            "off_rating_10":
                "opp_off_rating_10",

            "def_rating_10":
                "opp_def_rating_10",

            "net_rating_10":
                "opp_net_rating_10",

            "pace_10":
                "opp_pace_10",

            "win_pct_10":
                "opp_win_pct_10",

            "efg_pct_10":
                "opp_efg_pct_10",

            "tov_pct_10":
                "opp_tov_pct_10",

            "orb_pct_10":
                "opp_orb_pct_10",

            "ft_rate_10":
                "opp_ft_rate_10",

            "ast_avg_10":
                "opp_ast_avg_10",

            "reb_avg_10":
                "opp_reb_avg_10",

            "stl_avg_10":
                "opp_stl_avg_10",

            "blk_avg_10":
                "opp_blk_avg_10",

            "pf_avg_10":
                "opp_pf_avg_10",

            "rest_days":
                "opp_rest_days",

            "team_games_prior":
                "opp_team_games_prior",
        }
    )

    # Determine opponent IDs directly.
    game_team_ids = (
        df.groupby("GAME_ID")[
            "TEAM_ID"
        ]
        .transform("sum")
    )

    df["OPP_TEAM_ID"] = (
        game_team_ids
        - df["TEAM_ID"]
    ).where(
        game_counts == 2
    )

    df = df.merge(
        opponent_features,
        on=[
            "GAME_ID",
            "OPP_TEAM_ID",
        ],
        how="left",
        validate="one_to_one",
    )

    output_columns = [
        "SEASON",
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ID",
        "OPP_TEAM_ID",

        # Team context
        "pts_avg_10",
        "pts_allowed_10",
        "off_rating_10",
        "def_rating_10",
        "net_rating_10",
        "pace_10",
        "win_pct_10",

        "efg_pct_10",
        "tov_pct_10",
        "orb_pct_10",
        "ft_rate_10",

        "ast_avg_10",
        "reb_avg_10",
        "stl_avg_10",
        "blk_avg_10",
        "pf_avg_10",

        "rest_days",
        "team_games_prior",

        # Opponent context
        "opp_pts_avg_10",
        "opp_pts_allowed_10",
        "opp_off_rating_10",
        "opp_def_rating_10",
        "opp_net_rating_10",
        "opp_pace_10",
        "opp_win_pct_10",

        "opp_efg_pct_10",
        "opp_tov_pct_10",
        "opp_orb_pct_10",
        "opp_ft_rate_10",

        "opp_ast_avg_10",
        "opp_reb_avg_10",
        "opp_stl_avg_10",
        "opp_blk_avg_10",
        "opp_pf_avg_10",

        "opp_rest_days",
        "opp_team_games_prior",

        "team_pregame_elo",
        "opp_pregame_elo",
        "diff_elo",
        "elo_win_prob",
    ]

    result = (
        df[output_columns]
        .copy()
    )

    if result.duplicated(
        [
            "SEASON",
            "GAME_ID",
            "TEAM_ID",
        ]
    ).any():
        raise ValueError(
            "Duplicate team-game "
            "features detected"
        )

    return result


def main():
    logs = load_team_logs()

    features = build_team_features(
        logs
    )

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FEATURE_DIR
        / "team_matchup_features.parquet"
    )

    features.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"Saved {len(features):,} "
        "team-game feature rows"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        "\nMissing feature percentages:"
    )

    print(
        (
            features.isna().mean()
            * 100
        )
        .round(2)
        .sort_values(
            ascending=False
        )
        .head(20)
        .to_string()
    )

    print(
        "\nFeature preview:"
    )

    print(
        features.tail(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()