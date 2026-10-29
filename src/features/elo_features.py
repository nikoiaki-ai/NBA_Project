from itertools import product
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data/raw/team_game_logs"
PARAM_PATH = ROOT / "models/game_outcomes/elo_params.json"


# Keep 2015-16 and 2016-17 as warm-up history.
# Elo parameters are judged on 2017-18 through 2023-24.
TUNE_START_YEAR = 2017
TUNE_END_YEAR = 2023


def build_elo_games(df):
    """
    Convert two-row team logs into one-row-per-game records.

    Games with ambiguous home/away metadata are excluded from
    Elo fitting because home-court advantage cannot be assigned
    reliably.
    """

    df = df.copy()

    df["GAME_ID"] = (
        df["GAME_ID"]
        .astype(str)
        .str.zfill(10)
    )

    df["GAME_DATE"] = pd.to_datetime(
        df["GAME_DATE"]
    )

    df["is_home"] = (
        df["MATCHUP"]
        .str.contains(
            "vs.",
            regex=False,
            na=False,
        )
    )

    counts = df.groupby("GAME_ID").size()

    valid_ids = counts[
        counts == 2
    ].index

    df = df[
        df["GAME_ID"].isin(valid_ids)
    ].copy()

    home_counts = (
        df.groupby("GAME_ID")[
            "is_home"
        ]
        .sum()
    )

    valid_home_ids = (
        home_counts[
            home_counts == 1
        ].index
    )

    df = df[
        df["GAME_ID"].isin(
            valid_home_ids
        )
    ].copy()

    home = df[
        df["is_home"]
    ][
        [
            "GAME_ID",
            "GAME_DATE",
            "SEASON",
            "TEAM_ID",
            "WL",
        ]
    ].copy()

    away = df[
        ~df["is_home"]
    ][
        [
            "GAME_ID",
            "TEAM_ID",
        ]
    ].copy()

    home = home.rename(
        columns={
            "TEAM_ID": "home_team_id",
            "WL": "home_wl",
        }
    )

    away = away.rename(
        columns={
            "TEAM_ID": "away_team_id",
        }
    )

    games = home.merge(
        away,
        on="GAME_ID",
        how="inner",
        validate="one_to_one",
    )

    games["home_win"] = (
        games["home_wl"] == "W"
    ).astype(int)

    return games.sort_values(
        [
            "GAME_DATE",
            "GAME_ID",
        ]
    ).reset_index(drop=True)


def run_elo(
    games,
    k_factor,
    carry_over,
    home_advantage,
    base_rating=1500.0,
):
    """
    Walk through games chronologically and produce
    strictly pregame Elo ratings.
    """

    ratings = {}
    last_season = {}

    records = []

    for row in games.itertuples(
        index=False
    ):
        home_id = row.home_team_id
        away_id = row.away_team_id
        season = row.SEASON

        # Regress each team's rating toward the
        # league mean when it first appears in a new season.
        for team_id in (
            home_id,
            away_id,
        ):
            if team_id not in ratings:
                ratings[team_id] = (
                    base_rating
                )

            elif (
                last_season.get(team_id)
                != season
            ):
                ratings[team_id] = (
                    carry_over
                    * ratings[team_id]
                    + (1 - carry_over)
                    * base_rating
                )

        home_elo = ratings[home_id]
        away_elo = ratings[away_id]

        adjusted_home = (
            home_elo
            + home_advantage
        )

        home_prob = 1 / (
            1
            + 10 ** (
                (
                    away_elo
                    - adjusted_home
                )
                / 400
            )
        )

        records.append(
            {
                "GAME_ID": row.GAME_ID,
                "GAME_DATE": row.GAME_DATE,
                "SEASON": season,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_pregame_elo": home_elo,
                "away_pregame_elo": away_elo,
                "diff_elo": (
                    home_elo
                    - away_elo
                ),
                "elo_home_win_prob": (
                    home_prob
                ),
                "home_win": row.home_win,
            }
        )

        # Update only AFTER recording the
        # pregame ratings.
        elo_shift = (
            k_factor
            * (
                row.home_win
                - home_prob
            )
        )

        ratings[home_id] = (
            home_elo
            + elo_shift
        )

        ratings[away_id] = (
            away_elo
            - elo_shift
        )

        last_season[home_id] = season
        last_season[away_id] = season

    return pd.DataFrame(records)


def tune_elo(games):
    """
    Tune K, season carryover, and home advantage
    using historical chronological predictions.
    """

    k_values = [
        10,
        15,
        20,
        25,
        30,
        40,
    ]

    carry_values = [
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.90,
    ]

    home_adv_values = [
        0,
        25,
        50,
        75,
        100,
    ]

    results = []

    for (
        k_factor,
        carry_over,
        home_advantage,
    ) in product(
        k_values,
        carry_values,
        home_adv_values,
    ):

        predictions = run_elo(
            games,
            k_factor=k_factor,
            carry_over=carry_over,
            home_advantage=home_advantage,
        )

        season_year = (
            predictions["SEASON"]
            .str[:4]
            .astype(int)
        )

        score_mask = (
            season_year
            >= TUNE_START_YEAR
        ) & (
            season_year
            <= TUNE_END_YEAR
        )

        scored = predictions.loc[
            score_mask
        ]

        loss = log_loss(
            scored["home_win"],
            np.clip(
                scored[
                    "elo_home_win_prob"
                ],
                1e-6,
                1 - 1e-6,
            ),
        )

        results.append(
            {
                "k_factor": k_factor,
                "carry_over": (
                    carry_over
                ),
                "home_advantage": (
                    home_advantage
                ),
                "log_loss": loss,
            }
        )

    results = (
        pd.DataFrame(results)
        .sort_values("log_loss")
        .reset_index(drop=True)
    )

    return results


def add_elo_features(
    team_logs,
    params,
):
    """
    Attach Elo values to both team rows of each game.
    """

    games = build_elo_games(
        team_logs
    )

    elo_games = run_elo(
        games,
        k_factor=params[
            "k_factor"
        ],
        carry_over=params[
            "carry_over"
        ],
        home_advantage=params[
            "home_advantage"
        ],
    )

    home_rows = elo_games[
        [
            "GAME_ID",
            "home_team_id",
            "home_pregame_elo",
            "away_pregame_elo",
            "diff_elo",
            "elo_home_win_prob",
        ]
    ].rename(
        columns={
            "home_team_id":
                "TEAM_ID",
            "home_pregame_elo":
                "team_pregame_elo",
            "away_pregame_elo":
                "opp_pregame_elo",
            "elo_home_win_prob":
                "elo_win_prob",
        }
    )

    away_rows = elo_games[
        [
            "GAME_ID",
            "away_team_id",
            "away_pregame_elo",
            "home_pregame_elo",
            "diff_elo",
            "elo_home_win_prob",
        ]
    ].rename(
        columns={
            "away_team_id":
                "TEAM_ID",
            "away_pregame_elo":
                "team_pregame_elo",
            "home_pregame_elo":
                "opp_pregame_elo",
        }
    )

    # Reverse perspective for away team.
    away_rows[
        "diff_elo"
    ] = -away_rows[
        "diff_elo"
    ]

    away_rows[
        "elo_win_prob"
    ] = (
        1
        - away_rows[
            "elo_home_win_prob"
        ]
    )

    away_rows = away_rows.drop(
        columns=[
            "elo_home_win_prob"
        ]
    )

    elo_team_rows = pd.concat(
        [
            home_rows,
            away_rows,
        ],
        ignore_index=True,
    )

    output = team_logs.merge(
        elo_team_rows,
        on=[
            "GAME_ID",
            "TEAM_ID",
        ],
        how="left",
        validate="one_to_one",
    )

    return output


def load_elo_params():
    with open(
        PARAM_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main():
    paths = sorted(
        RAW_DIR.glob(
            "team_game_logs_*.parquet"
        )
    )

    logs = pd.concat(
        [
            pd.read_parquet(path)
            for path in paths
        ],
        ignore_index=True,
    )

    games = build_elo_games(logs)

    results = tune_elo(games)

    print(
        "\n=== TOP ELO CONFIGURATIONS ==="
    )

    print(
        results.head(15)
        .to_string(index=False)
    )

    best = (
        results.iloc[0]
        .to_dict()
    )

    params = {
        "k_factor": float(
            best["k_factor"]
        ),
        "carry_over": float(
            best["carry_over"]
        ),
        "home_advantage": float(
            best["home_advantage"]
        ),
    }

    PARAM_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        PARAM_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            params,
            file,
            indent=2,
        )

    print(
        "\nSelected Elo parameters:"
    )
    print(params)

    print(
        f"\nSaved: {PARAM_PATH}"
    )


if __name__ == "__main__":
    main()