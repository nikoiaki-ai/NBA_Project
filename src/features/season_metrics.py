"""Collect season-level NBA advanced team statistics."""

import time
from pathlib import Path

from nba_api.stats.endpoints import LeagueDashTeamStats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "team_advanced"

SEASONS = [
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
]


def collect_team_advanced(season: str):
    """Download advanced team statistics for one season."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR
        / f"team_advanced_{season.replace('-', '_')}.parquet"
    )

    if output_path.exists():
        print(f"Skipping {season}: already downloaded")
        return

    print(f"Downloading advanced team stats: {season}")

    response = LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
        timeout=60,
    )

    df = response.get_data_frames()[0]
    if df.empty:
        raise ValueError(f"Empty advanced team data: {season}")

    df["SEASON"] = season
    df.to_parquet(output_path, index=False)

    print(f"Saved {len(df)} teams to {output_path}")


def main():
    for season in SEASONS:
        try:
            collect_team_advanced(season)
        except Exception as exc:
            print(f"Failed {season}: {exc}")

        time.sleep(5)


if __name__ == "__main__":
    main()