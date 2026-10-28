
"""Validate historical NBA player and team game logs."""

import argparse
import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

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

LOG = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "GAME_ID", "GAME_DATE", "PTS",
    "FGM", "FGA", "FG3M", "FG3A",
    "FTM", "FTA",
]


def validate_dataset(
    df: pd.DataFrame,
    season: str,
    data_type: str,
) -> list[dict]:
    """Validate one player or team game-log dataset."""

    issues = []

    def report(severity, check, details):
        issues.append({
            "season": season,
            "type": data_type,
            "severity": severity,
            "check": check,
            "details": details,
        })

    key = "PLAYER_ID" if data_type == "player" else "TEAM_ID"

    required = REQUIRED_COLUMNS + [key]
    missing = sorted(set(required) - set(df.columns))

    if missing:
        report("ERROR", "missing_columns", str(missing))
        return issues

    if df.empty:
        report("ERROR", "empty_dataset", "No records")
        return issues

    # Missing values in essential fields
    null_counts = df[required].isna().sum()
    null_counts = null_counts[null_counts > 0]

    if not null_counts.empty:
        report(
            "ERROR",
            "missing_values",
            null_counts.to_dict(),
        )

    # Duplicate player/team appearances within a game
    duplicate_count = df.duplicated(
        subset=[key, "GAME_ID"]
    ).sum()

    if duplicate_count:
        report(
            "ERROR",
            "duplicate_records",
            f"{duplicate_count} duplicates",
        )

    # Verify scoring arithmetic
    expected_pts = (
        2 * df["FGM"]
        + df["FG3M"]
        + df["FTM"]
    )

    bad_scores = (
        (df["PTS"] != expected_pts)
        & df[["PTS", "FGM", "FG3M", "FTM"]].notna().all(axis=1)
    ).sum()

    if bad_scores:
        report(
            "ERROR",
            "scoring_arithmetic",
            f"{bad_scores} inconsistent rows",
        )

    # Makes cannot exceed attempts
    for made, attempted in [
        ("FGM", "FGA"),
        ("FG3M", "FG3A"),
        ("FTM", "FTA"),
    ]:
        invalid = (
            (df[made] > df[attempted])
            & df[[made, attempted]].notna().all(axis=1)
        ).sum()

        if invalid:
            report(
                "ERROR",
                "invalid_shooting",
                f"{invalid} rows: {made} > {attempted}",
            )

    # Two-point shooting must also be consistent
    invalid_two_point = (
        (df["FG3M"] > df["FGM"])
        | (df["FG3A"] > df["FGA"])
    ).sum()

    if invalid_two_point:
        report(
            "ERROR",
            "invalid_two_point_totals",
            f"{invalid_two_point} rows",
        )

    # Dates must parse successfully
    dates = pd.to_datetime(
        df["GAME_DATE"], errors="coerce"
    )

    if dates.isna().any():
        report(
            "ERROR",
            "invalid_dates",
            f"{dates.isna().sum()} invalid dates",
        )
    else:
        start_year = int(season[:4])

        # Broad regular-season window, allowing unusual scheduling
        earliest = pd.Timestamp(start_year, 9, 1)
        latest = pd.Timestamp(start_year + 1, 7, 1)

        outside = (
            (dates < earliest)
            | (dates > latest)
        ).sum()

        if outside:
            report(
                "WARNING",
                "unexpected_dates",
                f"{outside} dates outside expected range",
            )

    # Team logs should have exactly two teams per game
    if data_type == "team":
        teams_per_game = df.groupby(
            "GAME_ID"
        )["TEAM_ID"].nunique()

        incomplete = (teams_per_game != 2).sum()

        if incomplete:
            report(
                "ERROR",
                "incomplete_team_games",
                f"{incomplete} games do not have two teams",
            )

    return issues


def validate_player_team_consistency(
    players: pd.DataFrame,
    teams: pd.DataFrame,
    season: str,
) -> list[dict]:
    """Compare player scoring totals with team game scores."""

    issues = []

    player_totals = (
        players.groupby(
            ["GAME_ID", "TEAM_ID"],
            as_index=False
        )["PTS"].sum()
        .rename(columns={"PTS": "player_points"})
    )

    team_totals = teams[
        ["GAME_ID", "TEAM_ID", "PTS"]
    ].rename(columns={"PTS": "team_points"})

    comparison = team_totals.merge(
        player_totals,
        on=["GAME_ID", "TEAM_ID"],
        how="left",
        validate="one_to_one",
    )

    missing = comparison["player_points"].isna().sum()

    if missing:
        issues.append({
            "season": season,
            "type": "cross_check",
            "severity": "ERROR",
            "check": "missing_player_team_matches",
            "details": f"{missing} unmatched team games",
        })

    mismatched = (
        comparison["player_points"].notna()
        & (
            comparison["player_points"]
            != comparison["team_points"]
        )
    ).sum()

    if mismatched:
        issues.append({
            "season": season,
            "type": "cross_check",
            "severity": "ERROR",
            "check": "player_team_score_mismatch",
            "details": f"{mismatched} mismatched team scores",
        })

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Validate NBA game logs"
    )

    parser.add_argument(
        "--seasons",
        nargs="+",
        default=SEASONS,
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    all_issues = []
    summary = []

    for season in args.seasons:

        datasets = {}

        for data_type in ["player", "team"]:

            filename = (
                f"{data_type}_game_logs_"
                f"{season.replace('-', '_')}.parquet"
            )

            path = (
                RAW_DIR
                / f"{data_type}_game_logs"
                / filename
            )

            if not path.exists():
                all_issues.append({
                    "season": season,
                    "type": data_type,
                    "severity": "ERROR",
                    "check": "missing_file",
                    "details": str(path),
                })
                continue

            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                all_issues.append({
                    "season": season,
                    "type": data_type,
                    "severity": "ERROR",
                    "check": "unreadable_file",
                    "details": str(exc),
                })
                continue

            datasets[data_type] = df

            all_issues.extend(
                validate_dataset(df, season, data_type)
            )

            summary.append({
                "season": season,
                "type": data_type,
                "rows": len(df),
                "games": df["GAME_ID"].nunique()
                if "GAME_ID" in df else None,
            })

        if {"player", "team"} <= datasets.keys():
            players = datasets["player"]
            teams = datasets["team"]

            needed_player = {"GAME_ID", "TEAM_ID", "PTS"}
            needed_team = {"GAME_ID", "TEAM_ID", "PTS"}

            if (
                needed_player <= set(players.columns)
                and needed_team <= set(teams.columns)
            ):
                all_issues.extend(
                    validate_player_team_consistency(
                        players, teams, season
                    )
                )

    print("\nDATASET SUMMARY")
    print(pd.DataFrame(summary).to_string(index=False))

    print("\nVALIDATION ISSUES")

    if all_issues:
        issue_df = pd.DataFrame(all_issues)
        print(issue_df.to_string(index=False))
    else:
        print("No issues found.")

    error_count = sum(
        issue["severity"] == "ERROR"
        for issue in all_issues
    )

    warning_count = sum(
        issue["severity"] == "WARNING"
        for issue in all_issues
    )

    print(
        f"\nErrors: {error_count} | "
        f"Warnings: {warning_count}"
    )

    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
