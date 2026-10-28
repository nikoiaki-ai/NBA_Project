from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TEAM_DIR = ROOT / "data/raw/team_game_logs"

CORE_COLUMNS = [
    "SEASON",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
    "PLUS_MINUS",
]


def load_team_logs():
    paths = sorted(
        TEAM_DIR.glob("team_game_logs_*.parquet")
    )

    if not paths:
        raise FileNotFoundError(
            f"No team logs found in {TEAM_DIR}"
        )

    frames = []

    for path in paths:
        df = pd.read_parquet(path)
        df["SOURCE_FILE"] = path.name
        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined["GAME_ID"] = (
        combined["GAME_ID"]
        .astype(str)
        .str.zfill(10)
    )

    combined["GAME_DATE"] = pd.to_datetime(
        combined["GAME_DATE"]
    )

    return combined


def validate_schema(df):
    missing = [
        col for col in CORE_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("✓ Required columns present")


def validate_duplicates(df):
    dupes = df.duplicated(
        ["GAME_ID", "TEAM_ID"],
        keep=False,
    )

    print(
        f"Duplicate team-game rows: "
        f"{dupes.sum():,}"
    )

    if dupes.any():
        print(
            df.loc[
                dupes,
                [
                    "GAME_ID",
                    "GAME_DATE",
                    "TEAM_ABBREVIATION",
                    "SOURCE_FILE",
                ],
            ]
            .sort_values("GAME_DATE")
            .head(20)
            .to_string(index=False)
        )


def validate_game_pairing(df):
    counts = df.groupby("GAME_ID").size()

    bad = counts[counts != 2]

    print(
        f"Unique games: {len(counts):,}"
    )
    print(
        f"Games without exactly two team rows: "
        f"{len(bad):,}"
    )

    if not bad.empty:
        print("\nExamples:")
        print(bad.head(20))


def validate_results(df):
    invalid = ~df["WL"].isin(["W", "L"])

    print(
        f"Invalid W/L rows: {invalid.sum():,}"
    )

    # For complete games, one team should win
    # and the other should lose.
    wl_counts = (
        df.groupby("GAME_ID")["WL"]
        .agg(
            wins=lambda x: (x == "W").sum(),
            losses=lambda x: (x == "L").sum(),
        )
    )

    bad = wl_counts[
        (wl_counts["wins"] != 1)
        | (wl_counts["losses"] != 1)
    ]

    print(
        f"Games without exactly one W and one L: "
        f"{len(bad):,}"
    )


def validate_matchups(df):
    home = df["MATCHUP"].str.contains(
        "vs.", regex=False, na=False
    )

    away = df["MATCHUP"].str.contains(
        "@", regex=False, na=False
    )

    unknown = ~(home | away)

    print(
        f"Unrecognized MATCHUP rows: "
        f"{unknown.sum():,}"
    )

    home_counts = (
        pd.DataFrame({
            "GAME_ID": df["GAME_ID"],
            "home": home.astype(int),
        })
        .groupby("GAME_ID")["home"]
        .sum()
    )

    ambiguous = home_counts[
        home_counts != 1
    ]

    print(
        f"Games with ambiguous home/away: "
        f"{len(ambiguous):,}"
    )

    if not ambiguous.empty:
        examples = df[
            df["GAME_ID"].isin(
                ambiguous.index[:20]
            )
        ][
            [
                "GAME_ID",
                "GAME_DATE",
                "TEAM_ABBREVIATION",
                "MATCHUP",
            ]
        ]

        print("\nAmbiguous examples:")
        print(
            examples.to_string(index=False)
        )


def validate_missing(df):
    missing = (
        df[CORE_COLUMNS]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing = missing[missing > 0]

    print("\nMissing core values:")

    if missing.empty:
        print("None")
    else:
        print(missing.to_string())


def season_summary(df):
    summary = (
        df.groupby("SEASON")
        .agg(
            team_rows=("GAME_ID", "size"),
            games=("GAME_ID", "nunique"),
            first_game=("GAME_DATE", "min"),
            last_game=("GAME_DATE", "max"),
        )
        .sort_index()
    )

    print("\n=== SEASON SUMMARY ===")
    print(summary.to_string())


def validate_team_counts(df):
    team_counts = (
        df.groupby("SEASON")[
            "TEAM_ID"
        ]
        .nunique()
    )

    print("\nTeams by season:")
    print(team_counts.to_string())


def main():
    df = load_team_logs()

    print(
        f"Loaded {len(df):,} team-game rows"
    )
    print(
        f"Across {df['SEASON'].nunique()} seasons\n"
    )

    validate_schema(df)
    validate_duplicates(df)
    validate_game_pairing(df)
    validate_results(df)
    validate_matchups(df)
    validate_missing(df)
    validate_team_counts(df)
    season_summary(df)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()