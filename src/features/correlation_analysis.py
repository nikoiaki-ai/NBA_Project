
"""Pearson correlation analysis for NBA player-points features."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "features"
    / "player_features_advanced.parquet"
)

OUTPUT_DIR = ROOT / "reports" 

FEATURES = [
    # Scoring
    "pts_avg_5",
    "pts_avg_10",
    "pts_season_avg",
    "pts_std_10",

    # Minutes and opportunities
    "min_avg_5",
    "min_avg_10",
    "min_season_avg",
    "fga_avg_5",
    "fga_avg_10",
    "fga_season_avg",
    "fta_avg_10",
    "fga_per_min_10",
    "fta_per_min_10",

    # Shooting efficiency
    "fg_pct_10",
    "fg3_pct_10",
    "ft_pct_10",
    "three_point_rate_10",
    "ts_pct_10",
    "epaa_10",
    "epaa_per_attempt_10",

    # Matchup context
    "opp_pts_allowed_10",
    "opp_def_rating_10",
    "opp_pace_10",
    "opp_win_pct_10",

    # Schedule
    "player_rest_days",
    "is_home",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)

    # Only use historical training seasons for EDA.
    train = df[
        df["SEASON"].isin([
            "2022-23",
            "2023-24",
            "2024-25",
        ])
    ].copy()

    available_features = [
        feature for feature in FEATURES
        if feature in train.columns
    ]

    missing_features = sorted(
        set(FEATURES) - set(available_features)
    )

    if missing_features:
        print("Skipping unavailable features:", missing_features)

    numeric = train[
        available_features + ["PTS"]
    ].replace([np.inf, -np.inf], np.nan)

    # 1. Feature-to-feature Pearson correlations
    corr = numeric[available_features].corr(
        method="pearson"
    )

    fig, ax = plt.subplots(figsize=(15, 13))

    im = ax.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(len(available_features)))
    ax.set_yticks(range(len(available_features)))

    ax.set_xticklabels(
        available_features,
        rotation=90,
        fontsize=8,
    )
    ax.set_yticklabels(
        available_features,
        fontsize=8,
    )

    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("NBA Pregame Feature Correlation Matrix")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "feature_correlation.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # 2. Correlation with actual points
    target_corr = (
        numeric.corr(method="pearson")["PTS"]
        .drop("PTS")
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 9))

    target_corr.plot.barh(ax=ax)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with actual points")
    ax.set_title("Pregame Features vs Player Points")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "target_correlation.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # 3. Save correlation results
    corr.to_csv(
        OUTPUT_DIR / "correlation_matrix.csv"
    )

    target_corr.to_csv(
        OUTPUT_DIR / "target_correlations.csv"
    )

    print("\nCorrelations with actual points:")
    print(target_corr.sort_values(ascending=False).round(3))

    print("\nAnalysis complete.")
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
