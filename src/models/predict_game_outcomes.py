from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd

from src.features.game_outcome_f import (
    load_logs,
    add_team_features,
    ROLLING_STATS,
    WINDOWS,
)
from src.features.elo_features import load_elo_params


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models/game_outcomes"
STACK_DIR = MODEL_DIR / "stacking"
REPORT_DIR = ROOT / "reports/game_outcomes"


def season_for_date(game_date):
    start_year = (
        game_date.year
        if game_date.month >= 10
        else game_date.year - 1
    )
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def home_probability(model, X):
    """Read the probability for target class 1: home win."""
    classes = np.asarray(model.classes_)
    positions = np.flatnonzero(classes == 1)

    if len(positions) != 1:
        raise ValueError("Model must contain home-win class 1")

    probability = float(
        model.predict_proba(X)[0, positions[0]]
    )

    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid predicted probability")

    return probability


def load_context(game_date):
    logistic = joblib.load(
        STACK_DIR / "logistic_stack.joblib"
    )

    features = list(logistic["features"])
    base_names = list(logistic["base_names"])

    saved_models = {}

    for name in base_names:
        saved = joblib.load(
            MODEL_DIR / f"{name}.joblib"
        )

        if list(saved["features"]) != features:
            raise ValueError(
                f"Feature mismatch for {name}. "
                "Base models and stack must use the same features."
            )

        saved_models[name] = saved

    logs = load_logs()

    # Today's results must never enter today's predictions.
    logs = logs.loc[
        logs["GAME_DATE"] < game_date
    ].copy()

    if logs.empty:
        raise ValueError("No historical games before prediction date")

    latest_date = logs["GAME_DATE"].max()
    age_days = (game_date - latest_date).days

    print(
        f"Latest completed game in local logs: "
        f"{latest_date.date()}"
    )

    # A practical check for this regular-season prediction workflow.
    if age_days > 7:
        raise ValueError(
            f"Local logs are {age_days} days behind the prediction "
            "date. Refresh team game logs before predicting."
        )

    featured = add_team_features(logs)
    elo_params = load_elo_params()

    return {
        "logistic": logistic,
        "features": features,
        "base_names": base_names,
        "saved_models": saved_models,
        "featured": featured,
        "elo_params": elo_params,
    }


def build_team_snapshot(
    featured,
    team,
    game_date,
    elo_params,
):
    history = featured.loc[
        (featured["TEAM_ABBREVIATION"] == team)
        & (featured["GAME_DATE"] < game_date)
    ].sort_values(["GAME_DATE", "GAME_ID"])

    if history.empty:
        raise ValueError(f"No historical games found for {team}")

    latest = history.iloc[-1]
    snapshot = {}

    # Include the latest completed game in the upcoming game's
    # rolling window. Stored lagged averages exclude that game.
    for stat in ROLLING_STATS:
        for window in WINDOWS:
            snapshot[f"{stat.lower()}_avg_{window}"] = (
                history[stat].tail(window).mean()
            )

    snapshot["games_prior"] = len(history)

    days_since_last = (
        game_date - latest["GAME_DATE"]
    ).days

    snapshot["rest_days"] = max(days_since_last - 1, 0)

    required_elo = [
        latest["team_pregame_elo"],
        latest["elo_win_prob"],
        latest["win"],
    ]

    if not np.isfinite(
        np.asarray(required_elo, dtype=float)
    ).all():
        raise ValueError(
            f"Missing Elo state for {team}'s latest game"
        )

    # Recover the rating AFTER the latest completed game.
    # Its stored team_pregame_elo is from BEFORE that game.
    current_elo = float(latest["team_pregame_elo"]) + (
        float(elo_params["k_factor"])
        * (
            float(latest["win"])
            - float(latest["elo_win_prob"])
        )
    )

    # Match the season-boundary regression used in training.
    if str(latest["SEASON"]) != season_for_date(game_date):
        carry = float(elo_params["carry_over"])
        current_elo = carry * current_elo + (1 - carry) * 1500.0

    snapshot["team_pregame_elo"] = current_elo

    return snapshot, latest["GAME_DATE"]


def build_matchup_features(
    context,
    home_team,
    away_team,
    game_date,
):
    if home_team == away_team:
        raise ValueError("Home and away teams must differ")

    home, home_last = build_team_snapshot(
        context["featured"],
        home_team,
        game_date,
        context["elo_params"],
    )

    away, away_last = build_team_snapshot(
        context["featured"],
        away_team,
        game_date,
        context["elo_params"],
    )

    home_elo = home["team_pregame_elo"]
    away_elo = away["team_pregame_elo"]
    home_advantage = float(
        context["elo_params"]["home_advantage"]
    )

    elo_home_probability = 1 / (
        1 + 10 ** (
            (away_elo - home_elo - home_advantage) / 400
        )
    )

    home["elo_win_prob"] = elo_home_probability
    away["elo_win_prob"] = 1 - elo_home_probability

    row = {}

    for key in home:
        row[f"home_{key}"] = home[key]
        row[f"away_{key}"] = away[key]
        row[f"diff_{key}"] = home[key] - away[key]

    features = context["features"]

    missing = [
        feature for feature in features
        if feature not in row
    ]

    if missing:
        raise ValueError(
            f"Missing expected model features: {missing}"
        )

    X = pd.DataFrame([row])[features].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    metadata = {
        "home_last_game": home_last.date().isoformat(),
        "away_last_game": away_last.date().isoformat(),
        "home_rest_days": home["rest_days"],
        "away_rest_days": away["rest_days"],
        "home_elo": home_elo,
        "away_elo": away_elo,
    }

    return X, metadata


def predict_matchup(
    context,
    home_team,
    away_team,
    game_date,
):
    X, metadata = build_matchup_features(
        context,
        home_team,
        away_team,
        game_date,
    )

    probabilities = {}

    for name in context["base_names"]:
        probabilities[name] = home_probability(
            context["saved_models"][name]["pipeline"],
            X,
        )

    stack_input = pd.DataFrame(
        [probabilities],
        columns=context["base_names"],
    )

    probabilities["logistic_stack"] = home_probability(
        context["logistic"]["model"],
        stack_input,
    )

    return probabilities, metadata


def american_to_decimal(moneyline):
    moneyline = float(moneyline)

    if (
        not np.isfinite(moneyline)
        or abs(moneyline) < 100
    ):
        raise ValueError(
            f"Invalid American moneyline: {moneyline}"
        )

    if moneyline > 0:
        return 1 + moneyline / 100

    return 1 + 100 / abs(moneyline)


def load_odds(path, game_date):
    odds = pd.read_csv(path)

    required = {
        "game_date",
        "away_team",
        "home_team",
        "away_moneyline",
        "home_moneyline",
        "sportsbook",
    }

    missing = required - set(odds.columns)

    if missing:
        raise ValueError(
            f"Missing odds columns: {sorted(missing)}"
        )

    if odds.empty:
        raise ValueError("Odds CSV is empty")

    if odds[list(required)].isna().any().any():
        raise ValueError("Missing required values in odds CSV")

    odds["game_date"] = pd.to_datetime(
        odds["game_date"],
        format="%Y-%m-%d",
        errors="raise",
    ).dt.normalize()

    odds = odds.loc[
        odds["game_date"] == game_date
    ].copy()

    if odds.empty:
        raise ValueError(
            f"No odds found for {game_date.date()}"
        )

    for column in ["home_team", "away_team"]:
        odds[column] = (
            odds[column].astype(str).str.strip().str.upper()
        )

    odds["sportsbook"] = (
        odds["sportsbook"].astype(str).str.strip()
    )

    if (
        odds["home_team"].eq("")
        | odds["away_team"].eq("")
        | odds["sportsbook"].eq("")
    ).any():
        raise ValueError("Blank team or sportsbook")

    if odds["home_team"].eq(odds["away_team"]).any():
        raise ValueError("A team cannot play itself")

    keys = [
        "game_date",
        "home_team",
        "away_team",
        "sportsbook",
    ]

    if odds.duplicated(keys).any():
        raise ValueError(
            "Duplicate matchup/sportsbook rows in odds CSV"
        )

    for side in ["home", "away"]:
        column = f"{side}_moneyline"

        odds[column] = pd.to_numeric(
            odds[column],
            errors="raise",
        )

        odds[f"{side}_decimal"] = odds[column].map(
            american_to_decimal
        )

    return odds


def run_odds_report(game_date, odds_path, stake):
    if not np.isfinite(stake) or stake <= 0:
        raise ValueError("Illustrative stake must be positive")

    odds = load_odds(odds_path, game_date)
    context = load_context(game_date)

    generated_at = pd.Timestamp.now(tz="UTC").isoformat()

    prediction_rows = []
    ev_rows = []
    prediction_cache = {}

    for quote in odds.to_dict("records"):
        home = quote["home_team"]
        away = quote["away_team"]
        key = (home, away)

        if key not in prediction_cache:
            probabilities, metadata = predict_matchup(
                context,
                home,
                away,
                game_date,
            )

            prediction_cache[key] = probabilities

            home_prob = probabilities["logistic_stack"]

            prediction_rows.append({
                "game_date": game_date.date().isoformat(),
                "away_team": away,
                "home_team": home,
                "home_win_probability": home_prob,
                "away_win_probability": 1 - home_prob,
                "predicted_winner": (
                    home if home_prob >= 0.5 else away
                ),
                "probability_model": "logistic_stack",
                "generated_at_utc": generated_at,
                **metadata,
                **{
                    f"{name}_home_probability": probability
                    for name, probability in probabilities.items()
                },
            })

        probabilities = prediction_cache[key]
        home_prob = probabilities["logistic_stack"]

        home_implied = 1 / quote["home_decimal"]
        away_implied = 1 / quote["away_decimal"]
        implied_total = home_implied + away_implied

        # Normalize both sides from this same sportsbook quote.
        no_vig = {
            "home": home_implied / implied_total,
            "away": away_implied / implied_total,
        }

        for side, team, probability in [
            ("home", home, home_prob),
            ("away", away, 1 - home_prob),
        ]:
            decimal_odds = quote[f"{side}_decimal"]
            break_even = 1 / decimal_odds

            # Expected NET profit per unit staked.
            ev_fraction = probability * decimal_odds - 1

            ev_rows.append({
                "game_date": game_date.date().isoformat(),
                "away_team": away,
                "home_team": home,
                "team": team,
                "side": side,
                "sportsbook": quote["sportsbook"],
                "american_odds": quote[f"{side}_moneyline"],
                "decimal_odds": decimal_odds,
                "model_probability": probability,
                "break_even_probability": break_even,
                "market_no_vig_probability": no_vig[side],
                "edge_vs_break_even_pp": (
                    100 * (probability - break_even)
                ),
                "edge_vs_no_vig_pp": (
                    100 * (probability - no_vig[side])
                ),
                "ev_fraction": ev_fraction,
                "ev_pct": 100 * ev_fraction,
                "illustrative_stake": stake,
                "expected_net_profit": stake * ev_fraction,
                "positive_model_ev": ev_fraction > 0,
                "book_overround_pct": (
                    100 * (implied_total - 1)
                ),
                "probability_model": "logistic_stack",
                "odds_source_file": odds_path.name,
                # Blank unless a timestamp was supplied in the CSV.
                "odds_captured_at": quote.get("captured_at", ""),
                "generated_at_utc": generated_at,
            })

    predictions = pd.DataFrame(prediction_rows)

    results = pd.DataFrame(ev_rows).sort_values(
        "ev_pct",
        ascending=False,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    date_text = game_date.date().isoformat()

    predictions_path = (
        REPORT_DIR / f"live_predictions_{date_text}.csv"
    )
    ev_path = (
        REPORT_DIR / f"moneyline_ev_{date_text}.csv"
    )

    predictions.to_csv(predictions_path, index=False)
    results.to_csv(ev_path, index=False)

    print("\n=== GAME PREDICTIONS ===")

    summary = predictions[[
        "away_team",
        "home_team",
        "away_win_probability",
        "home_win_probability",
        "predicted_winner",
        "away_last_game",
        "home_last_game",
    ]].copy()

    for column in [
        "away_win_probability",
        "home_win_probability",
    ]:
        summary[column] = summary[column].map(
            lambda value: f"{value:.2%}"
        )

    print(summary.to_string(index=False))

    print("\n=== MONEYLINE EV: BOTH SIDES ===")

    display = results[[
        "team",
        "sportsbook",
        "american_odds",
        "model_probability",
        "break_even_probability",
        "market_no_vig_probability",
        "ev_pct",
        "expected_net_profit",
    ]].copy()

    for column in [
        "model_probability",
        "break_even_probability",
        "market_no_vig_probability",
    ]:
        display[column] = display[column].map(
            lambda value: f"{value:.2%}"
        )

    display["american_odds"] = display["american_odds"].map(
        lambda value: f"{value:+.0f}"
    )
    display["ev_pct"] = display["ev_pct"].map(
        lambda value: f"{value:+.2f}%"
    )
    display["expected_net_profit"] = (
        display["expected_net_profit"].map(
            lambda value: f"{value:+.2f}"
        )
    )

    print(display.to_string(index=False))

    print(
        f"\nExpected net profit uses an illustrative "
        f"${stake:.2f} stake per side."
    )
    print("Positive EV is conditional on the model probability.")
    print("Injury and depth adjustments are not included.")
    print(f"\nSaved predictions: {predictions_path}")
    print(f"Saved EV report: {ev_path}")

    return predictions, results


def predict(home_team, away_team, game_date):
    """Retain the previous single-matchup prediction entry point."""
    game_date = pd.Timestamp(game_date).normalize()

    home_team = home_team.upper()
    away_team = away_team.upper()

    context = load_context(game_date)

    probabilities, metadata = predict_matchup(
        context,
        home_team,
        away_team,
        game_date,
    )

    print(f"\n{away_team} at {home_team}")
    print(
        f"Latest games: {away_team} "
        f"{metadata['away_last_game']}; "
        f"{home_team} {metadata['home_last_game']}"
    )

    print("\nHOME WIN PROBABILITIES")

    for name, probability in probabilities.items():
        print(f"{name:25s} {probability:.2%}")

    return probabilities


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--date", required=True)
    parser.add_argument("--odds")
    parser.add_argument("--home")
    parser.add_argument("--away")
    parser.add_argument(
        "--stake",
        type=float,
        default=100.0,
        help="Illustrative stake for dollar EV; not a sizing recommendation",
    )

    args = parser.parse_args()

    game_date = pd.Timestamp(args.date).normalize()

    if args.odds:
        if args.home or args.away:
            parser.error(
                "Use --odds or --home/--away, not both"
            )

        odds_path = Path(args.odds)

        if not odds_path.is_absolute():
            odds_path = ROOT / odds_path

        run_odds_report(
            game_date,
            odds_path,
            args.stake,
        )

    elif args.home and args.away:
        predict(
            args.home,
            args.away,
            game_date,
        )

    else:
        parser.error(
            "Provide --odds, or both --home and --away"
        )


if __name__ == "__main__":
    main()