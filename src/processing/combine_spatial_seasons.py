
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "data/raw/shot_charts"

SEASONS = [
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
]


def combine_season(season):
    season_dir = BASE_DIR / season
    files = sorted(season_dir.glob("team_*.parquet"))

    if not files:
        print(f"{season}: No team files found. Skipping.")
        return

    print(f"\n{season}: Found {len(files)} team files")

    if len(files) != 30:
        print(
            f"WARNING: Expected 30 team files, found {len(files)}."
        )
        print("Skipping to avoid saving an incomplete season.")
        return

    frames = [pd.read_parquet(file) for file in files]
    combined = pd.concat(frames, ignore_index=True)

    before = len(combined)

    combined["GAME_ID"] = (
        combined["GAME_ID"].astype(str).str.zfill(10)
    )

    combined = combined.drop_duplicates(
        subset=["GAME_ID", "GAME_EVENT_ID"]
    )

    combined = combined.sort_values(
        ["GAME_DATE", "GAME_ID", "GAME_EVENT_ID"]
    ).reset_index(drop=True)

    suffix = season.replace("-", "_")
    output = BASE_DIR / f"shot_charts_{suffix}.parquet"

    combined.to_parquet(output, index=False)

    print(f"Rows before deduplication: {before:,}")
    print(f"Rows after deduplication:  {len(combined):,}")
    print(f"Duplicate events removed: {before - len(combined):,}")
    print(f"Saved: {output.name}")


def main():
    for season in SEASONS:
        combine_season(season)


if __name__ == "__main__":
    main()
