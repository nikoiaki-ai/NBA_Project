
"""Validate shot-chart records against player game logs."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SEASONS = [
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
]

KEYS = ["GAME_ID", "PLAYER_ID", "TEAM_ID"]


def validate_season(season):
    suffix = season.replace("-", "_")

    shot_path = (
        ROOT / "data/raw/shot_charts"
        / f"shot_charts_{suffix}.parquet"
    )

    log_path = (
        ROOT / "data/raw/player_game_logs"
        / f"player_game_logs_{suffix}.parquet"
    )

    if not shot_path.exists() or not log_path.exists():
        print(f"{season}: Missing input file")
        return None

    shots = pd.read_parquet(shot_path)
    logs = pd.read_parquet(log_path)

    for df in (shots, logs):
        df["GAME_ID"] = (
            df["GAME_ID"].astype(str).str.zfill(10)
        )

    duplicates = shots.duplicated(
        ["GAME_ID", "GAME_EVENT_ID"]
    ).sum()

    totals = (
        shots.groupby(KEYS)
        .agg(
            shot_fga=("SHOT_ATTEMPTED_FLAG", "sum"),
            shot_fgm=("SHOT_MADE_FLAG", "sum"),
        )
        .reset_index()
    )

    comparison = logs.merge(
        totals,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    comparison[["shot_fga", "shot_fgm"]] = (
        comparison[["shot_fga", "shot_fgm"]].fillna(0)
    )

    fga_match = (
        comparison["FGA"] == comparison["shot_fga"]
    )
    fgm_match = (
        comparison["FGM"] == comparison["shot_fgm"]
    )
    # Identify player-games with mismatched shot totals
    mismatches = comparison.loc[
        ~(fga_match & fgm_match)
    ]

    if not mismatches.empty:
        print(f"\n{season} mismatches:")
        print(
            mismatches[
            [
                "GAME_ID",
                "PLAYER_ID",
                "TEAM_ID",
                "FGA",
                "shot_fga",
                "FGM",
                "shot_fgm",
            ]
        ].to_string(index=False)
    )

    return {
        "season": season,
        "shots": len(shots),
        "games": shots["GAME_ID"].nunique(),
        "duplicate_events": int(duplicates),
        "fga_match_pct": round(
            fga_match.mean() * 100, 3
        ),
        "fgm_match_pct": round(
            fgm_match.mean() * 100, 3
        ),
        "mismatched_rows": int(
            (~(fga_match & fgm_match)).sum()
        ),
    }


def main():
    results = []

    for season in SEASONS:
        result = validate_season(season)

        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError(
            "No seasons available for validation."
        )

    report = pd.DataFrame(results)

    print("\nSPATIAL DATA VALIDATION")
    print(report.to_string(index=False))

    if (
        (report["mismatched_rows"] > 0).any()
        or (report["duplicate_events"] > 0).any()
    ):
        print(
            "\nWARNING: Resolve discrepancies "
            "before feature engineering."
        )
    else:
        print("\nAll available seasons passed validation.")


if __name__ == "__main__":
    main()
