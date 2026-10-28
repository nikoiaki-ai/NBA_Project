
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

shots = pd.read_parquet(
    ROOT / "data/raw/shot_charts/shot_charts_2024_25.parquet"
)

logs = pd.read_parquet(
    ROOT / "data/raw/player_game_logs/player_game_logs_2024_25.parquet"
)

shots["GAME_ID"] = shots["GAME_ID"].astype(str).str.zfill(10)
logs["GAME_ID"] = logs["GAME_ID"].astype(str).str.zfill(10)

keys = ["GAME_ID", "PLAYER_ID", "TEAM_ID"]

shot_totals = (
    shots.groupby(keys)
    .agg(
        shot_fga=("SHOT_ATTEMPTED_FLAG", "sum"),
        shot_fgm=("SHOT_MADE_FLAG", "sum"),
    )
    .reset_index()
)

comparison = logs.merge(
    shot_totals,
    on=keys,
    how="left",
    validate="one_to_one",
)

comparison[["shot_fga", "shot_fgm"]] = (
    comparison[["shot_fga", "shot_fgm"]].fillna(0)
)

comparison["fga_diff"] = (
    comparison["shot_fga"] - comparison["FGA"]
)

comparison["fgm_diff"] = (
    comparison["shot_fgm"] - comparison["FGM"]
)

print("Shot records:", len(shots))
print("Unique games:", shots["GAME_ID"].nunique())
print("Player-game rows:", len(comparison))

print("\nFGA agreement:",
      (comparison["fga_diff"] == 0).mean())

print("FGM agreement:",
      (comparison["fgm_diff"] == 0).mean())

print("\nTotal FGA difference:",
      comparison["fga_diff"].sum())

print("\nMost common shot zones:")
print(shots["SHOT_ZONE_BASIC"].value_counts())

print("\nRows with discrepancies:")
print(
    comparison.loc[
        (comparison["fga_diff"] != 0)
        | (comparison["fgm_diff"] != 0),
        keys + [
            "FGA", "shot_fga",
            "FGM", "shot_fgm",
        ],
    ].head(15)
)
