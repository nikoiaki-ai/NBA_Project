
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PATH = (
    ROOT / "data/features"
    / "spatial_epaa_v2_all_seasons.parquet"
)

df = pd.read_parquet(PATH)

df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

keys = ["GAME_ID", "PLAYER_ID", "TEAM_ID"]

# 1. Unique player-game records
assert not df.duplicated(keys).any(), (
    "Duplicate player-game records found"
)

print("PASS: Unique player-game keys")

# 2. Regional probabilities
regions = [
    "restricted",
    "paint",
    "midrange",
    "left_corner_3",
    "right_corner_3",
    "above_break_3",
    "free_throw",
]

selection_cols = [
    f"v2_select_{r}" for r in regions
]

accuracy_cols = [
    f"v2_accuracy_{r}" for r in regions
]

valid = df[selection_cols].notna().all(axis=1)

assert np.allclose(
    df.loc[valid, selection_cols].sum(axis=1),
    1.0,
    atol=1e-8,
)

print("PASS: Selection probabilities sum to 1")

# 3. Probability bounds
prob_cols = selection_cols + accuracy_cols

values = df[prob_cols].to_numpy(dtype=float)

assert not np.isinf(values).any()
assert np.nanmin(values) >= 0
assert np.nanmax(values) <= 1

print("PASS: All probabilities are between 0 and 1")

# 4. EPAA identity
valid_epaa = df[
    [
        "v2_expected_points_100",
        "v2_league_points_100",
        "spatial_epaa_v2_100",
    ]
].dropna()

assert np.allclose(
    valid_epaa["spatial_epaa_v2_100"],
    valid_epaa["v2_expected_points_100"]
    - valid_epaa["v2_league_points_100"],
)

print("PASS: EPAA calculation is consistent")

# 5. Inspect first appearance in each season
df["SEASON_START_YEAR"] = (
    df["GAME_DATE"].dt.year
    - (df["GAME_DATE"].dt.month < 7).astype(int)
)

df = df.sort_values(
    ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
)

first_games = (
    df.groupby(
        ["PLAYER_ID", "SEASON_START_YEAR"]
    )
    .head(1)
)

print("\nFirst-game historical attempts by season:")
print(
    first_games.groupby("SEASON_START_YEAR")[
        "v2_prior_attempts_10"
    ].agg(["count", "mean", "median"])
    .round(2)
)

print("\nValidation complete.")
