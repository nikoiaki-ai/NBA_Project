
import unittest

import pandas as pd

from src.features.player_features import (
    prepare_data,
    add_rolling_features,
    add_season_features,
)


class TestPlayerFeatures(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "PLAYER_ID": [1, 1, 1, 1],
            "GAME_ID": ["001", "002", "003", "004"],
            "GAME_DATE": pd.to_datetime([
                "2025-04-01",
                "2025-04-03",
                "2025-10-22",
                "2025-10-24",
            ]),
            "SEASON": [
                "2024-25", "2024-25",
                "2025-26", "2025-26",
            ],
            "PTS": [10, 20, 30, 40],
        })

        # Only the scoring columns are needed for these tests.
        # add_rolling_features also expects other box-score stats.
        for stat in [
            "MIN", "FGA", "FG3A", "FTA",
            "FGM", "FG3M", "FTM",
            "AST", "REB", "TOV",
        ]:
            self.df[stat] = 0

    def test_current_game_not_included(self):
        df = prepare_data(self.df)
        df = add_rolling_features(df)

        self.assertTrue(pd.isna(df.loc[0, "pts_avg_5"]))
        self.assertEqual(df.loc[1, "pts_avg_5"], 10)
        self.assertEqual(df.loc[2, "pts_avg_5"], 15)

    def test_season_average_resets(self):
        df = prepare_data(self.df)
        df = add_season_features(df)

        self.assertTrue(
            pd.isna(df.loc[2, "pts_season_avg"])
        )
        self.assertEqual(df.loc[3, "pts_season_avg"], 30)

    def test_current_game_change_does_not_leak(self):
        original = add_rolling_features(
            prepare_data(self.df)
        )

        modified = self.df.copy()
        modified.loc[2, "PTS"] = 100

        updated = add_rolling_features(
            prepare_data(modified)
        )

        self.assertEqual(
            original.loc[2, "pts_avg_5"],
            updated.loc[2, "pts_avg_5"],
        )

        self.assertNotEqual(
            original.loc[3, "pts_avg_5"],
            updated.loc[3, "pts_avg_5"],
        )


if __name__ == "__main__":
    unittest.main()
