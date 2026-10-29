import numpy as np
import pandas as pd

from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit


class GameDateSplit(BaseCrossValidator):
    def __init__(self, game_dates=None, n_splits=3):
        self.game_dates = game_dates
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if self.game_dates is None:
            raise ValueError("Game dates are required for calibration")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("GameDateSplit requires a pandas DataFrame")

        dates = pd.to_datetime(
            self.game_dates.loc[X.index]
        ).to_numpy()

        if pd.isna(dates).any():
            raise ValueError("Missing calibration game dates")

        unique_dates = np.unique(dates)

        if len(unique_dates) <= self.n_splits:
            raise ValueError("Not enough dates for calibration folds")

        splitter = TimeSeriesSplit(n_splits=self.n_splits)

        for fit_dates_idx, check_dates_idx in splitter.split(unique_dates):
            fit_dates = unique_dates[fit_dates_idx]
            check_dates = unique_dates[check_dates_idx]

            fit_idx = np.flatnonzero(np.isin(dates, fit_dates))
            check_idx = np.flatnonzero(np.isin(dates, check_dates))

            yield fit_idx, check_idx