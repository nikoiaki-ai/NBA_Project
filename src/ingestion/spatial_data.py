
"""Collect NBA shot-location data, one team at a time."""

from pathlib import Path
import random
import time
import argparse

import pandas as pd
from nba_api.stats.endpoints import shotchartdetail


ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--season",
    type=str,
    default="2024-25",
    help="NBA season, e.g. 2022-23"
)
args = parser.parse_args()

SEASON = args.season
SEASON_FILE = SEASON.replace("-", "_")

LOG_PATH = (
    ROOT / "data/raw/team_game_logs"
    / f"team_game_logs_{SEASON_FILE}.parquet"
)

OUTPUT_DIR = ROOT / "data/raw/shot_charts" / SEASON
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    ROOT / "data/raw/shot_charts"
    / f"shot_charts_{SEASON_FILE}.parquet"
)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def fetch_team_shots(team_id, attempts=4):
    for attempt in range(attempts):
        try:
            response = shotchartdetail.ShotChartDetail(
                team_id=int(team_id),
                player_id=0,
                season_nullable=SEASON,
                season_type_all_star="Regular Season",
                context_measure_simple="FGA",
                timeout=60,
            )

            df = response.get_data_frames()[0]

            if df.empty:
                raise ValueError(
                    f"No shots returned for team {team_id}"
                )

            return df

        except Exception as exc:
            print(
                f"Team {team_id}, attempt "
                f"{attempt + 1}: {exc}"
            )

            if attempt == attempts - 1:
                raise

            time.sleep(10 * (attempt + 1))

    raise RuntimeError("Unreachable")


def main():
    team_logs = pd.read_parquet(LOG_PATH)

    team_ids = sorted(
        team_logs["TEAM_ID"].dropna().unique()
    )

    print(f"Collecting {SEASON}")
    print(f"Teams: {len(team_ids)}")

    failed = []

    for i, team_id in enumerate(team_ids, start=1):
        path = OUTPUT_DIR / f"team_{int(team_id)}.parquet"

        if path.exists():
            print(f"[{i}/{len(team_ids)}] Cached {team_id}")
            continue

        print(f"[{i}/{len(team_ids)}] Fetching {team_id}")

        try:
            shots = fetch_team_shots(team_id)
            shots.to_parquet(path, index=False)

            print(f"  Saved {len(shots):,} shots")

        except Exception as exc:
            failed.append((int(team_id), str(exc)))
            print(f"  Failed: {exc}")

        time.sleep(random.uniform(3, 6))

    files = sorted(OUTPUT_DIR.glob("team_*.parquet"))

    if not files:
        raise RuntimeError("No shot data collected")

    combined = pd.concat(
        [pd.read_parquet(path) for path in files],
        ignore_index=True,
    )

    if "GAME_EVENT_ID" in combined.columns:
        combined = combined.drop_duplicates(
            subset=["GAME_ID", "GAME_EVENT_ID"]
        )

    output = (
        ROOT / "data/raw/shot_charts"
        / "shot_charts_2024_25.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)

    print(f"\nTotal shots: {len(combined):,}")
    print(f"Saved: {output}")

    print("\nColumns:")
    print(combined.columns.tolist())

    if failed:
        print("\nFailed teams:")
        for team_id, error in failed:
            print(team_id, error)

        print("Rerun the script to retry missing teams.")


if __name__ == "__main__":
    main()
