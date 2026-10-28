
"""Collect historical NBA player and team game logs."""

import argparse
import logging
import random
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog


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


def fetch_game_logs(
    season: str,
    data_type: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch and validate player ('P') or team ('T') game logs."""

    if data_type not in {"P", "T"}:
        raise ValueError("data_type must be 'P' or 'T'")

    key = "PLAYER_ID" if data_type == "P" else "TEAM_ID"

    for attempt in range(1, max_retries + 1):
        try:
            response = LeagueGameLog(
                season=season,
                player_or_team_abbreviation=data_type,
                season_type_all_star="Regular Season",
                timeout=60,
            )

            df = response.get_data_frames()[0]

            required = {key, "GAME_ID", "GAME_DATE", "PTS"}
            missing = required - set(df.columns)

            if missing:
                raise ValueError(
                    f"{season}: missing columns {sorted(missing)}"
                )

            if df.empty:
                raise ValueError(f"{season}: empty API response")

            if df[list(required)].isna().any().any():
                raise ValueError(
                    f"{season}: null values in required columns"
                )

            if df.duplicated([key, "GAME_ID"]).any():
                raise ValueError(
                    f"{season}: duplicate game records"
                )

            df["GAME_DATE"] = pd.to_datetime(
                df["GAME_DATE"], errors="raise"
            )
            df["SEASON"] = season

            return df

        except Exception as exc:
            LOG.warning(
                "Attempt %s/%s failed for %s (%s): %s",
                attempt, max_retries, season, data_type, exc
            )

            if attempt == max_retries:
                raise

            delay = min(60, 5 * 2 ** (attempt - 1))
            time.sleep(delay + random.uniform(0, 2))

    raise RuntimeError("Unexpected retry failure")


def collect_season(
    season: str,
    data_type: str,
    refresh: bool = False,
) -> dict:
    """Download one season and save an immutable raw snapshot."""

    label = "player" if data_type == "P" else "team"
    folder = RAW_DIR / f"{label}_game_logs"
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"{label}_game_logs_{season.replace('-', '_')}.parquet"
    output_path = folder / filename

    if output_path.exists() and not refresh:
        LOG.info("Skipping existing file: %s", output_path)
        df = pd.read_parquet(output_path)
        status = "cached"
    else:
        LOG.info("Downloading %s logs for %s", label, season)
        df = fetch_game_logs(season, data_type)

        # Write to a temporary file first to avoid partial outputs.
        temp_path = output_path.with_suffix(".tmp.parquet")
        try:
            df.to_parquet(temp_path, index=False)
            temp_path.replace(output_path)
        finally:
            temp_path.unlink(missing_ok=True)

        status = "downloaded"

    return {
        "season": season,
        "type": label,
        "rows": len(df),
        "games": df["GAME_ID"].nunique(),
        "first_date": df["GAME_DATE"].min(),
        "last_date": df["GAME_DATE"].max(),
        "status": status,
    }


def main() -> None:
    """Run collection from the command line."""

    parser = argparse.ArgumentParser(
        description="Collect NBA historical game logs"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=SEASONS,
        help="NBA seasons, e.g. 2024-25 2025-26",
    )
    parser.add_argument(
        "--type",
        choices=["player", "team", "both"],
        default="both",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Replace existing snapshots with fresh downloads",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    types = {
        "player": ["P"],
        "team": ["T"],
        "both": ["P", "T"],
    }[args.type]

    results = []

    for season in args.seasons:
        for data_type in types:
            try:
                result = collect_season(
                    season,
                    data_type,
                    refresh=args.refresh,
                )
                results.append(result)
                LOG.info(
                    "%s %s: %s rows (%s)",
                    season,
                    result["type"],
                    result["rows"],
                    result["status"],
                )
            except Exception:
                LOG.exception(
                    "Collection failed: %s %s",
                    season,
                    data_type,
                )

            time.sleep(random.uniform(3, 6))

    if results:
        summary = pd.DataFrame(results)
        print("\nCollection summary:")
        print(summary.to_string(index=False))

    expected = len(args.seasons) * len(types)
    if len(results) != expected:
        raise SystemExit(
            f"Only {len(results)}/{expected} collections succeeded"
        )


if __name__ == "__main__":
    main()
