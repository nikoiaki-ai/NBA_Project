"""Niko's Picks — NBA predictions and DraftKings moneyline EV.

Place this file in the New_NBA project root and run:
    python -m streamlit run app.py

Uses existing src/features modules, saved model artifacts, and local game logs.
The live predictor and EV logic are included in this file; no changes to
src/models/predict_game_outcomes.py are required for this dashboard.

Manual odds path: data/raw/odds/draftkings_YYYY-MM-DD.csv
Columns: game_date,away_team,home_team,away_moneyline,home_moneyline,sportsbook
Optional column: captured_at (an ISO timestamp with timezone).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features.game_outcome_f import (
    ROLLING_STATS,
    WINDOWS,
    add_team_features,
    load_logs,
)


st.set_page_config(
    page_title="Niko's Picks | NBA Intelligence",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "reports" / "game_outcomes" / "stacking_validation_results.csv"
MODEL_DIR = ROOT / "models" / "game_outcomes"
STACK_DIR = MODEL_DIR / "stacking"
PREDICTION_LOG = ROOT / "reports" / "game_outcomes" / "live_predictions.csv"

BASE_NAMES = [
    "xgboost",
    "adaboost",
    "naive_bayes",
    "naive_bayes_calibrated",
    "knn",
    "knn_calibrated",
    "logistic_regression",
    "decision_tree",
    "mlp",
]

VALIDATION_SEASON = "2024-25"


def today_et():
    return pd.Timestamp.now(tz="America/New_York").date()


DISPLAY_NAMES = {
    "naive_bayes_calibrated": "Calibrated Naive Bayes",
    "knn_calibrated": "Calibrated KNN",
    "mlp": "MLP Base",
    "logistic_stack": "Logistic Stack",
    "simple_average": "Simple Average",
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "mlp_stack": "MLP Stack",
    "adaboost": "AdaBoost",
    "knn": "KNN",
    "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes",
}


st.markdown(
    """
<style>
:root {color-scheme: dark;}
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #080e19 !important;
    color: #e9f1fc !important;
}
[data-testid="stHeader"] {background: #080e19 !important;}
[data-testid="stToolbar"] {color: #c7d5ea !important;}
[data-testid="stSidebar"] {
    background: #0e1829 !important;
    border-right: 1px solid #243249 !important;
}
[data-testid="stSidebar"] * {color: #d9e5f5;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color: #a8bcd4;}
.block-container {padding: 2.25rem 2.5rem 4rem !important; max-width: 1420px;}
[data-testid="stMainBlockContainer"] {background: transparent;}
h1,h2,h3,h4,p,label,span {color: inherit;}
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p {color: #c9d7e9;}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {
    background: #15243a !important; border: 1px solid #35465e !important;
    color: #f4f7fd !important; border-radius: 10px !important;
}
[data-baseweb="select"] *, [data-baseweb="popover"] * {color: #f4f7fd !important;}
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul {background: #15243a !important;}
[data-testid="stRadio"] label {color: #dbe7f6 !important;}
[data-testid="stDataFrame"] {border: 1px solid #293c55; border-radius: 12px; overflow: hidden;}
[data-testid="stAlert"] {background: #13243a !important; color: #d6e7ff !important; border-radius: 12px;}
[data-testid="stAlert"] * {color: #d6e7ff !important;}
[data-testid="stButton"] button {
    background: linear-gradient(135deg,#2bcf9f,#3a9fe8) !important;
    color: #07111c !important; border: 0 !important; border-radius: 10px !important;
    font-weight: 800 !important; min-height: 44px;
}
hr {border-color: #283b52 !important;}
.hero {
    padding: 36px 40px; border-radius: 20px;
    background: radial-gradient(ellipse at 90% 0%, #173b43 0%, transparent 40%),
                linear-gradient(115deg, #16263e, #0e192b 78%);
    border: 1px solid #2b425a; margin-bottom: 26px;
}
.eyebrow {font-size: 11px; font-weight: 800; letter-spacing: 3px; color: #55e1b4;}
.hero h1 {font-size: clamp(32px, 3.4vw, 49px); color: #f5f8ff !important;
    letter-spacing: -1.8px; line-height: 1.1; margin: 15px 0 14px; font-weight: 800;}
.hero p {font-size: 14px; color: #b7c8df !important; margin: 0;}
.metric {padding: 22px 22px 20px; border: 1px solid #2a3c55;
    background: linear-gradient(150deg,#152238,#101a2b); border-radius: 15px; min-height: 137px;}
.metric .label {font-size: 10px; color: #aebfd6; font-weight: 800; letter-spacing: 1.3px;}
.metric .value {font-size: clamp(24px, 2.5vw, 34px); font-weight: 800;
    color: #f5f8ff; margin-top: 10px; letter-spacing: -1px;}
.metric .detail {font-size: 12px; color: #5ee3b7; margin-top: 7px; font-weight: 600;}
.section-label {font-size: 12px; letter-spacing: 2px; color: #7ce6c3;
    font-weight: 800; margin: 30px 0 14px;}
.panel {background: #111d30; border: 1px solid #2a3c55;
    border-radius: 16px; padding: 22px 24px; margin-top: 4px;}
.panel-title {font-size: 17px; color: #f2f7ff; font-weight: 750; margin-bottom: 14px;}
.panel-row {font-size: 13px; color: #b7c9df; padding: 10px 0;
    border-bottom: 1px solid #25354a; line-height: 1.6;}
.panel-row:last-child {border-bottom: 0;}
.panel-row strong {color: #f0f6ff;}
.pill {display: inline-block; border: 1px solid #286c61; background: #103a38;
    color: #78f2c9; border-radius: 999px; font-size: 11px; font-weight: 700;
    padding: 6px 12px; margin-top: 15px;}
.note {color: #9eb3cd; font-size: 12px; line-height: 1.7; margin-top: 10px;}
.brand {font-size: 22px; color: #f3f8ff; font-weight: 850; letter-spacing: 1.1px;}
.brand-sub {font-size: 10px; letter-spacing: 2px; color: #83a1be; margin-top: 4px;}
.prediction-hero {
    background: radial-gradient(circle at 50% 0%, #173e43, #101b2e 58%);
    border: 1px solid #315068; border-radius: 18px; padding: 30px 24px; text-align: center;
}
.prediction-team {font-size: clamp(30px, 4vw, 50px); font-weight: 850; color:#f5f8ff; letter-spacing:-1px;}
.prediction-prob {font-size: clamp(42px, 6vw, 72px); font-weight: 900; color:#67e7bd; line-height:1; margin:12px 0;}
.muted {color:#9db2ca; font-size:12px;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_results(signature=()) -> pd.DataFrame:
    if not RESULTS.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULTS)


@st.cache_data(show_spinner=False)
def read_logs(signature=()) -> pd.DataFrame:
    return load_logs()


def artifact_signature():
    paths = [
        RESULTS, MODEL_DIR / "elo_params.json",
        ROOT / "data/processed/game_outcomes.parquet",
        *MODEL_DIR.glob("*.joblib"), *STACK_DIR.glob("*.joblib"),
        *(ROOT / "data/raw/team_game_logs").glob("team_game_logs_*.parquet"),
    ]
    return tuple(
        (str(p), p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(set(paths)) if p.exists()
    )


@st.cache_resource(show_spinner=False, max_entries=2)
def load_saved_models(signature):
    logistic = joblib.load(STACK_DIR / "logistic_stack.joblib")
    names = list(logistic["base_names"])
    features = list(logistic["features"])
    if not names or len(names) != len(set(names)):
        raise ValueError("Invalid base-model list in logistic stack")
    base = {}
    for name in names:
        saved = joblib.load(MODEL_DIR / f"{name}.joblib")
        if list(saved["features"]) != features:
            raise ValueError(f"Feature order mismatch for {name}; retrain the stack.")
        base[name] = saved
    mlp_path = STACK_DIR / "mlp_stack.joblib"
    mlp = joblib.load(mlp_path) if mlp_path.exists() else None
    if mlp is not None:
        if list(mlp["features"]) != features:
            raise ValueError("MLP-stack feature mismatch; retrain it or remove the stale file.")
        if not set(mlp["base_names"]).issubset(base):
            raise ValueError("MLP stack requires base models absent from logistic stack.")
    return base, logistic, mlp


@st.cache_data(show_spinner=False, max_entries=4)
def read_validation_count(signature):
    path = ROOT / "data/processed/game_outcomes.parquet"
    if not path.exists():
        return None
    seasons = pd.read_parquet(path, columns=["SEASON"])
    return int(seasons["SEASON"].eq(VALIDATION_SEASON).sum())


def season_for_date(game_date):
    start = game_date.year if game_date.month >= 10 else game_date.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def home_probability(model, X):
    positions = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positions) != 1:
        raise ValueError("Model must contain home-win class 1")
    p = float(model.predict_proba(X)[0, positions[0]])
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Invalid model probability")
    return p


def get_prediction_context(game_date):
    game_date = pd.Timestamp(game_date).normalize()
    signature = artifact_signature()
    key = (game_date.isoformat(), signature)
    cached = st.session_state.get("_prediction_context")
    if cached is not None and cached["key"] == key:
        return cached["value"]
    base, logistic, mlp = load_saved_models(signature)
    logs = read_logs(signature)
    historical = logs.loc[logs["GAME_DATE"] < game_date].copy()
    if historical.empty:
        raise ValueError("No completed games before the selected date")
    latest = historical["GAME_DATE"].max()
    if (game_date - latest).days > 7:
        raise ValueError(
            f"Local logs end on {latest.date()}. Refresh your team game logs "
            "before using this regular-season forecast."
        )
    with (MODEL_DIR / "elo_params.json").open(encoding="utf-8") as f:
        params = json.load(f)
    value = {
        "base": base, "logistic": logistic, "mlp": mlp,
        "featured": add_team_features(historical), "params": params,
    }
    st.session_state["_prediction_context"] = {"key": key, "value": value}
    return value


def build_team_snapshot(featured, team, game_date, params):
    history = featured.loc[
        featured["TEAM_ABBREVIATION"].eq(team)
        & (featured["GAME_DATE"] < game_date)
    ].sort_values(["GAME_DATE", "GAME_ID"])
    if history.empty:
        raise ValueError(f"No completed history for {team} before {game_date.date()}")
    latest = history.iloc[-1]
    snapshot = {
        f"{stat.lower()}_avg_{window}": history[stat].tail(window).mean()
        for stat in ROLLING_STATS for window in WINDOWS
    }
    snapshot["games_prior"] = len(history)
    snapshot["rest_days"] = max((game_date - latest["GAME_DATE"]).days - 1, 0)
    state = np.asarray([
        latest["team_pregame_elo"], latest["elo_win_prob"], latest["win"]
    ], dtype=float)
    if not np.isfinite(state).all():
        raise ValueError(f"Missing Elo state for {team}'s latest completed game")
    # Recover the postgame rating using exactly the training update rule.
    rating = state[0] + float(params["k_factor"]) * (state[2] - state[1])
    if str(latest["SEASON"]) != season_for_date(game_date):
        carry = float(params["carry_over"])
        rating = carry * rating + (1 - carry) * 1500.0
    snapshot["team_pregame_elo"] = rating
    return snapshot, latest["GAME_DATE"]


def build_live_features(model_context, home_team, away_team, game_date):
    if home_team == away_team:
        raise ValueError("Home and away teams must differ")
    params = model_context["params"]
    home, home_last = build_team_snapshot(
        model_context["featured"], home_team, game_date, params
    )
    away, away_last = build_team_snapshot(
        model_context["featured"], away_team, game_date, params
    )
    p_elo = 1 / (1 + 10 ** (
        (away["team_pregame_elo"] - home["team_pregame_elo"]
         - float(params["home_advantage"])) / 400
    ))
    home["elo_win_prob"] = p_elo
    away["elo_win_prob"] = 1 - p_elo
    row = {}
    for key in home:
        row[f"home_{key}"] = home[key]
        row[f"away_{key}"] = away[key]
        row[f"diff_{key}"] = home[key] - away[key]
    features = list(model_context["logistic"]["features"])
    missing = [f for f in features if f not in row]
    if missing:
        raise ValueError(f"Live feature builder is missing trained features: {missing}")
    X = pd.DataFrame([row])[features].replace([np.inf, -np.inf], np.nan)
    context = {
        "home_last_game": home_last, "away_last_game": away_last,
        "home_rest_days": home["rest_days"], "away_rest_days": away["rest_days"],
        "home_elo": home["team_pregame_elo"],
        "away_elo": away["team_pregame_elo"],
    }
    return X, context


def run_prediction(home_team, away_team, game_date):
    game_date = pd.Timestamp(game_date).normalize()
    models = get_prediction_context(game_date)
    X, context = build_live_features(models, home_team, away_team, game_date)
    probs = {
        name: home_probability(saved["pipeline"], X)
        for name, saved in models["base"].items()
    }
    base_frame = pd.DataFrame([probs], columns=models["logistic"]["base_names"])
    probs["simple_average"] = float(base_frame.mean(axis=1).iloc[0])
    probs["logistic_stack"] = home_probability(
        models["logistic"]["model"], base_frame
    )
    if models["mlp"] is not None:
        probs["mlp_stack"] = home_probability(
            models["mlp"]["model"], base_frame[models["mlp"]["base_names"]]
        )
    return probs, context


def american_to_decimal(moneyline):
    value = float(moneyline)
    if not np.isfinite(value) or abs(value) < 100 or not value.is_integer():
        raise ValueError(f"Invalid American moneyline: {moneyline}")
    return 1 + value / 100 if value > 0 else 1 + 100 / abs(value)


def load_odds(path, game_date):
    odds = pd.read_csv(path)
    required = {
        "game_date", "home_team", "away_team", "home_moneyline",
        "away_moneyline", "sportsbook",
    }
    missing = required - set(odds.columns)
    if missing:
        raise ValueError(f"Missing odds columns: {sorted(missing)}")
    if odds.empty or odds[list(required)].isna().any().any():
        raise ValueError("Odds file is empty or contains missing required values")
    odds["game_date"] = pd.to_datetime(
        odds["game_date"], format="%Y-%m-%d", errors="raise"
    ).dt.normalize()
    odds = odds.loc[odds["game_date"].eq(pd.Timestamp(game_date).normalize())].copy()
    if odds.empty:
        raise ValueError("The odds file has no rows for the selected date")
    for col in ["home_team", "away_team", "sportsbook"]:
        odds[col] = odds[col].astype(str).str.strip()
        if odds[col].eq("").any():
            raise ValueError(f"Blank values in {col}")
    for col in ["home_team", "away_team"]:
        odds[col] = odds[col].str.upper()
    if odds["home_team"].eq(odds["away_team"]).any():
        raise ValueError("A team cannot play itself")
    keys = ["game_date", "home_team", "away_team", "sportsbook"]
    if odds.duplicated(keys).any():
        raise ValueError("Duplicate matchup/sportsbook quotes in odds file")
    for side in ["home", "away"]:
        col = f"{side}_moneyline"
        odds[col] = pd.to_numeric(odds[col], errors="raise")
        odds[f"{side}_decimal"] = odds[col].map(american_to_decimal)
    return odds


def quote_ev_rows(quote, home_prob, generated_at):
    if not np.isfinite(home_prob) or not 0 <= home_prob <= 1:
        raise ValueError("Invalid home-win probability")
    implied = {side: 1 / quote[f"{side}_decimal"] for side in ["home", "away"]}
    total = sum(implied.values())
    rows = []
    for side, p in [("home", home_prob), ("away", 1 - home_prob)]:
        decimal = quote[f"{side}_decimal"]
        no_vig = implied[side] / total
        ev = p * decimal - 1
        captured = quote.get("captured_at", "")
        rows.append({
            "game_date": pd.Timestamp(quote["game_date"]).date().isoformat(),
            "home_team": quote["home_team"], "away_team": quote["away_team"],
            "team": quote[f"{side}_team"], "side": side,
            "sportsbook": quote["sportsbook"],
            "american_odds": quote[f"{side}_moneyline"], "decimal_odds": decimal,
            "model_probability": p, "break_even_probability": implied[side],
            "market_no_vig_probability": no_vig,
            "edge_vs_no_vig_pp": 100 * (p - no_vig),
            "ev_fraction": ev, "ev_pct": 100 * ev,
            "book_overround_pct": 100 * (total - 1),
            "probability_model": "logistic_stack",
            "odds_captured_at": captured if pd.notna(captured) else "",
            "generated_at_utc": generated_at,
        })
    return rows


def generate_slate(game_date, odds):
    generated = pd.Timestamp.now(tz="UTC").isoformat()
    prediction_rows, ev_rows = [], []
    cache = {}
    for quote in odds.to_dict("records"):
        home, away = quote["home_team"], quote["away_team"]
        key = (home, away)
        if key not in cache:
            probs, context = run_prediction(home, away, pd.Timestamp(game_date))
            cache[key] = probs
            p = probs["logistic_stack"]
            prediction_rows.append({
                "game_date": pd.Timestamp(game_date).date().isoformat(),
                "home_team": home, "away_team": away,
                "home_win_probability": p, "away_win_probability": 1 - p,
                "predicted_winner": home if p >= 0.5 else away,
                "probability_model": "logistic_stack",
                "generated_at_utc": generated, **context,
                **{f"{name}_home_probability": value for name, value in probs.items()},
            })
        ev_rows.extend(quote_ev_rows(quote, cache[key]["logistic_stack"], generated))
    return pd.DataFrame(prediction_rows), pd.DataFrame(ev_rows).sort_values("ev_fraction", ascending=False)


def odds_signature(path, game_date):
    return (
        str(path), pd.Timestamp(game_date).isoformat(),
        path.stat().st_mtime_ns, path.stat().st_size, artifact_signature(),
    )



def save_prediction(home_team, away_team, game_date, probs, generated_at=""):
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "recorded_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "probability_model": "logistic_stack",
        "forecast_generated_at": generated_at,
        "game_date": game_date.date().isoformat(),
        "away_team": away_team,
        "home_team": home_team,
        **probs,
    }
    new = pd.DataFrame([row])

    if PREDICTION_LOG.exists():
        old = pd.read_csv(PREDICTION_LOG)
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new

    out.to_csv(PREDICTION_LOG, index=False)


if st.sidebar.button("Refresh local data & models", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    for key in ["_prediction_context", "latest_prediction", "ev_snapshot"]:
        st.session_state.pop(key, None)
    st.rerun()


results = read_results(artifact_signature())
if not results.empty:
    required = {"model", "accuracy", "roc_auc", "log_loss", "brier"}
    if not required.issubset(results.columns):
        st.error(f"Validation report is missing columns: {sorted(required - set(results.columns))}")
        st.stop()
    results["display_name"] = results["model"].map(
        lambda s: DISPLAY_NAMES.get(s, s.replace("_", " ").title())
    )


st.sidebar.markdown(
    '<div class="brand">🏀 Niko’s Picks</div><div class="brand-sub">NBA PREDICTIVE INTELLIGENCE</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "WORKSPACE",
    ["Overview", "Model Lab", "Matchup Explorer", "Odds & EV", "Prediction History"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("RESEARCH BUILD  ·  V0.4")
st.sidebar.markdown('<span class="pill">● LOCAL MODEL ARTIFACTS</span>', unsafe_allow_html=True)

# Artifact-health status in the sidebar.
artifact_checks = {
    "Validation report": RESULTS.exists(),
    "Base models": all((MODEL_DIR / f"{m}.joblib").exists() for m in BASE_NAMES),
    "Logistic stack": (STACK_DIR / "logistic_stack.joblib").exists(),
    "MLP stack": (STACK_DIR / "mlp_stack.joblib").exists(),
}
for label, ok in artifact_checks.items():
    icon = "✓" if ok else "×"
    st.sidebar.caption(f"{icon} {label}")


st.markdown(
    '''<div class="hero"><div class="eyebrow">NBA / PREDICTIVE ANALYTICS</div>
<h1>Every game has a probability.</h1>
<p>Pregame modeling &nbsp;·&nbsp; Nine base-model candidates &nbsp;·&nbsp; Chronological validation &nbsp;·&nbsp; Stacked ensembles</p></div>''',
    unsafe_allow_html=True,
)


if page in ("Overview", "Model Lab"):
    if results.empty:
        st.error(
            "No validation report was found at "
            "`reports/game_outcomes/stacking_validation_results.csv`. "
            "Run the stacking training script first."
        )
        st.stop()

    validation_count = read_validation_count(artifact_signature())
    best_accuracy = results.loc[results["accuracy"].idxmax()]
    best_loss = results.loc[results["log_loss"].idxmin()]
    best_auc = results.loc[results["roc_auc"].idxmax()]
    cards = [
        ("BEST ACCURACY", f'{best_accuracy["accuracy"]:.2%}', best_accuracy["display_name"]),
        ("LOWEST LOG LOSS", f'{best_loss["log_loss"]:.4f}', best_loss["display_name"]),
        ("BEST ROC-AUC", f'{best_auc["roc_auc"]:.4f}', best_auc["display_name"]),
        ("VALIDATION GAMES", f"{validation_count:,}" if validation_count is not None else "—", "2024–25 season"),
    ]
    for column, (label, value, detail) in zip(st.columns(4, gap="medium"), cards):
        column.markdown(
            f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="detail">{detail}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">MODEL PERFORMANCE / 2024–25</div>', unsafe_allow_html=True)
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        metric = st.selectbox(
            "Compare metric",
            ["accuracy", "roc_auc", "log_loss", "brier"],
            format_func=lambda v: {
                "accuracy": "Accuracy",
                "roc_auc": "ROC-AUC",
                "log_loss": "Log loss",
                "brier": "Brier score",
            }[v],
        )
        smaller_is_better = metric in {"log_loss", "brier"}
        chart_df = results.sort_values(metric, ascending=smaller_is_better).iloc[::-1].copy()
        colors = ["#5be3b5" if m == "logistic_stack" else "#5087c9" for m in chart_df["model"]]
        values = chart_df[metric] * (100 if metric == "accuracy" else 1)
        suffix = "%" if metric == "accuracy" else ""
        fig = go.Figure(
            go.Bar(
                x=values,
                y=chart_df["display_name"],
                orientation="h",
                marker_color=colors,
                marker_line_width=0,
                text=[
                    f"{v:.2f}{suffix}" if metric == "accuracy" else f"{v:.4f}"
                    for v in values
                ],
                textposition="outside",
                textfont=dict(color="#eaf3ff", size=12),
                hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            paper_bgcolor="#111d30",
            plot_bgcolor="#111d30",
            font=dict(color="#cbd9ec", family="Arial", size=12),
            margin=dict(l=10, r=70, t=20, b=32),
            height=465,
            xaxis=dict(showgrid=True, gridcolor="#263850", zeroline=False, title=None),
            yaxis=dict(showgrid=False, title=None),
            bargap=0.38,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Green = primary logistic stacking model. Lower is better for log loss and Brier score.")

    with right:
        st.markdown(
            '''<div class="panel"><div class="panel-title">Research snapshot</div>
<div class="panel-row"><strong>Primary predictor</strong><br>Logistic Regression stack</div>
<div class="panel-row"><strong>Research architecture</strong><br>MLP stacking ensemble</div>
<div class="panel-row"><strong>Base classifiers</strong><br>XGBoost · AdaBoost · GaussianNB · Calibrated GaussianNB · KNN · Calibrated KNN · Logistic Regression · Decision Tree · MLP</div>
<div class="panel-row"><strong>Evaluation</strong><br>Chronological split · 2024–25 validation · 2025–26 held out</div>
<div class="note">The primary predictor is chosen from validation results. The 2026–27 season remains prospective.</div>
</div>''',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">FULL VALIDATION SCOREBOARD</div>', unsafe_allow_html=True)
    table = results.sort_values("log_loss")[["display_name", "accuracy", "roc_auc", "log_loss", "brier"]]
    table = table.rename(
        columns={
            "display_name": "Model",
            "accuracy": "Accuracy",
            "roc_auc": "ROC-AUC",
            "log_loss": "Log loss",
            "brier": "Brier",
        }
    )
    st.dataframe(
        table.style.format(
            {"Accuracy": "{:.2%}", "ROC-AUC": "{:.4f}", "Log loss": "{:.4f}", "Brier": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )


if page == "Matchup Explorer":
    st.markdown('<div class="section-label">LIVE MATCHUP EXPLORER</div>', unsafe_allow_html=True)

    try:
        logs = read_logs(artifact_signature())
        teams = sorted(logs["TEAM_ABBREVIATION"].dropna().unique().tolist())
        latest_log_date = logs["GAME_DATE"].max().date()
    except Exception as exc:
        st.error(f"Could not load team game logs: {exc}")
        st.stop()

    top_left, top_right = st.columns([2, 1], gap="large")
    with top_left:
        st.markdown("### Build a pregame prediction")
        st.caption(
            f"Latest completed game in the local data: {latest_log_date}. "
            "Only games before the selected date are used."
        )
    with top_right:
        prediction_date = st.date_input("Game date", value=today_et())

    away_col, home_col = st.columns(2, gap="large")
    with away_col:
        default_away = teams.index("DET") if "DET" in teams else 0
        away = st.selectbox("Away team", teams, index=default_away)
    with home_col:
        default_home = teams.index("NYK") if "NYK" in teams else min(1, len(teams) - 1)
        home = st.selectbox("Home team", teams, index=default_home)

    if away == home:
        st.warning("Select two different teams.")
    else:
        st.markdown(
            f'''<div class="panel" style="text-align:center;padding:28px 20px;">
<div class="eyebrow">PREGAME MATCHUP</div>
<div style="font-size:40px;font-weight:800;color:#f5f8ff;margin:14px 0;">{away} <span style="color:#6fe4bc">@</span> {home}</div>
<div class="muted">{prediction_date.strftime('%B %d, %Y')}</div>
</div>''',
            unsafe_allow_html=True,
        )

        if st.button("Run model prediction", use_container_width=True):
            st.session_state.pop("latest_prediction", None)
            with st.spinner("Building pregame features and running the ensemble..."):
                try:
                    probs, context = run_prediction(home, away, pd.Timestamp(prediction_date))
                    st.session_state["latest_prediction"] = {
                        "home": home,
                        "away": away,
                        "date": pd.Timestamp(prediction_date),
                        "probs": probs,
                        "context": context,
                        "signature": artifact_signature(),
                        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    }
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        latest = st.session_state.get("latest_prediction")
        if latest and latest.get("signature") == artifact_signature() and latest["home"] == home and latest["away"] == away and latest["date"].date() == prediction_date:
            probs = latest["probs"]
            context = latest["context"]
            primary = probs["logistic_stack"]
            winner = home if primary >= 0.5 else away
            confidence = max(primary, 1 - primary)

            st.markdown('<div class="section-label">PRIMARY FORECAST</div>', unsafe_allow_html=True)
            main, context_col = st.columns([1.35, 1], gap="large")
            with main:
                st.markdown(
                    f'''<div class="prediction-hero">
<div class="eyebrow">LOGISTIC STACK</div>
<div class="prediction-team">{winner}</div>
<div class="prediction-prob">{confidence:.1%}</div>
<div class="muted">predicted win probability</div>
</div>''',
                    unsafe_allow_html=True,
                )
            with context_col:
                st.markdown(
                    f'''<div class="panel"><div class="panel-title">Pregame context</div>
<div class="panel-row"><strong>{home} last game</strong><br>{context['home_last_game'].date()}</div>
<div class="panel-row"><strong>{away} last game</strong><br>{context['away_last_game'].date()}</div>
<div class="panel-row"><strong>Rest days</strong><br>{home}: {int(context['home_rest_days'])} &nbsp;·&nbsp; {away}: {int(context['away_rest_days'])}</div>
<div class="panel-row"><strong>Pregame Elo</strong><br>{home}: {context['home_elo']:.0f} &nbsp;·&nbsp; {away}: {context['away_elo']:.0f}</div>
<div class="note">The displayed forecast is from the validation-winning logistic stack. The MLP remains visible below for research comparison.</div></div>''',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="section-label">MODEL CONSENSUS</div>', unsafe_allow_html=True)
            probability_df = pd.DataFrame(
                {
                    "model": [DISPLAY_NAMES.get(k, k) for k in probs],
                    "home_win_probability": list(probs.values()),
                    "key": list(probs.keys()),
                }
            ).sort_values("home_win_probability")

            bar_colors = [
                "#5be3b5" if key == "logistic_stack" else
                "#a789ff" if key == "mlp_stack" else "#5087c9"
                for key in probability_df["key"]
            ]
            fig = go.Figure(
                go.Bar(
                    x=probability_df["home_win_probability"] * 100,
                    y=probability_df["model"],
                    orientation="h",
                    marker_color=bar_colors,
                    text=[f"{p:.1%}" for p in probability_df["home_win_probability"]],
                    textposition="outside",
                    hovertemplate="%{y}<br>Home win: %{x:.1f}%<extra></extra>",
                )
            )
            fig.add_vline(x=50, line_dash="dash", line_color="#73869d", opacity=0.75)
            fig.update_layout(
                paper_bgcolor="#111d30",
                plot_bgcolor="#111d30",
                font=dict(color="#cbd9ec"),
                xaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="#263850", title=f"{home} win probability"),
                yaxis=dict(title=None),
                margin=dict(l=10, r=55, t=20, b=35),
                height=470,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            left_save, right_save = st.columns([1, 2])
            with left_save:
                if st.button("Save prediction to history", use_container_width=True):
                    save_prediction(home, away, pd.Timestamp(prediction_date), probs, latest["generated_at"])
                    st.success(f"Saved to {PREDICTION_LOG.relative_to(ROOT)}")
            with right_save:
                st.caption(
                    "Save before tipoff to retain a prospective record. Injury/depth adjustments are not included. This page does not verify whether the game has started."
                )


if page == "Prediction History":
    st.markdown('<div class="section-label">PROSPECTIVE PREDICTION LOG</div>', unsafe_allow_html=True)
    if not PREDICTION_LOG.exists():
        st.info("No live predictions have been saved yet. Use Matchup Explorer → Save prediction to history.")
    else:
        history = pd.read_csv(PREDICTION_LOG)
        st.metric("Recorded predictions", len(history))
        visible = history.copy()
        probability_cols = [c for c in [*BASE_NAMES, "simple_average", "logistic_stack", "mlp_stack"] if c in visible]
        for c in probability_cols:
            visible[c] = visible[c].map(lambda x: f"{float(x):.1%}")
        st.dataframe(visible.iloc[::-1], use_container_width=True, hide_index=True)
        st.caption(f"Stored locally at {PREDICTION_LOG.relative_to(ROOT)}")


if page == "Odds & EV":
    st.markdown('<div class="section-label">DRAFTKINGS / MONEYLINE EV</div>', unsafe_allow_html=True)
    st.markdown("### Game probabilities and market prices")
    st.caption("Logistic-stack forecasts with your saved DraftKings prices. Injury and depth adjustments are not included.")
    left, right = st.columns(2)
    with left:
        ev_date = st.date_input("Slate date", value=today_et(), key="ev_date")
    with right:
        stake = st.number_input("Illustrative stake ($)", min_value=1.0, value=100.0, step=10.0,
                                help="Converts EV to dollars; this is not a bankroll-sizing recommendation.")
    odds_path = ROOT / "data/raw/odds" / f"draftkings_{ev_date.isoformat()}.csv"
    if not odds_path.exists():
        st.info(f"No price snapshot for this date. Add {odds_path.relative_to(ROOT)} to your project.")
    else:
        try:
            odds = load_odds(odds_path, ev_date)
        except Exception as exc:
            st.error(f"Could not read the odds snapshot: {exc}")
            st.stop()
        signature = odds_signature(odds_path, ev_date)
        st.caption(f"Manual price snapshot: {odds_path.name}. These are saved prices, not a live odds feed.")
        if "captured_at" not in odds or odds["captured_at"].fillna("").eq("").any():
            st.caption("A capture timestamp was not recorded for some or all quotes.")
        with st.expander("DraftKings input prices"):
            st.dataframe(odds[["away_team", "home_team", "away_moneyline", "home_moneyline", "sportsbook"]],
                         use_container_width=True, hide_index=True)
        if st.button("Generate predictions & EV", use_container_width=True):
            st.session_state.pop("ev_snapshot", None)
            try:
                with st.spinner("Building pregame features and evaluating the slate..."):
                    predictions, ev_results = generate_slate(pd.Timestamp(ev_date), odds)
                ev_results["odds_source_file"] = odds_path.name
                st.session_state["ev_snapshot"] = {
                    "signature": signature, "predictions": predictions, "results": ev_results,
                    "generated_at": pd.Timestamp.now(tz="America/New_York").isoformat(timespec="seconds"),
                }
            except Exception as exc:
                st.error(f"Could not generate predictions: {exc}")
        snapshot = st.session_state.get("ev_snapshot")
        if snapshot is not None:
            if snapshot["signature"] != signature:
                st.info("The date or input files changed. Generate predictions again to refresh these results.")
            else:
                predictions = snapshot["predictions"].copy()
                ev_results = snapshot["results"].copy()
                for frame in [predictions, ev_results]:
                    frame["matchup"] = frame["away_team"] + " @ " + frame["home_team"]
                ev_results["illustrative_stake"] = stake
                ev_results["expected_net_profit"] = ev_results["ev_fraction"] * stake
                st.caption(f"Forecast generated: {snapshot['generated_at']}")
                matchups = predictions["matchup"].tolist()
                selected = st.multiselect("Games to display", options=matchups, default=matchups,
                                         key="ev_games_" + snapshot["generated_at"])
                positive_only = st.checkbox("Show only positive model EV", value=False)
                shown_predictions = predictions.loc[predictions["matchup"].isin(selected)]
                sides = ev_results.loc[ev_results["matchup"].isin(selected)]
                shown = sides.loc[sides["ev_fraction"] > 0] if positive_only else sides
                shown = shown.sort_values("ev_fraction", ascending=False)
                one, two, three = st.columns(3)
                one.metric("Games displayed", len(shown_predictions))
                two.metric("Sides with positive model EV", int((sides["ev_fraction"] > 0).sum()))
                three.metric("Highest model EV", f"{sides['ev_pct'].max():+.2f}%" if not sides.empty else "—")
                st.markdown("### Game forecasts")
                forecast = shown_predictions[["matchup", "away_win_probability", "home_win_probability", "predicted_winner"]].rename(columns={
                    "matchup": "Matchup", "away_win_probability": "Away win", "home_win_probability": "Home win", "predicted_winner": "Predicted winner",
                })
                st.dataframe(forecast.style.format({"Away win": "{:.2%}", "Home win": "{:.2%}"}), use_container_width=True, hide_index=True)
                st.markdown("### Moneyline EV")
                if shown.empty:
                    st.info("No sides match the current filters.")
                else:
                    table = shown[["matchup", "team", "sportsbook", "american_odds", "model_probability", "break_even_probability",
                                   "market_no_vig_probability", "edge_vs_no_vig_pp", "ev_pct", "expected_net_profit"]].rename(columns={
                        "matchup": "Matchup", "team": "Team", "sportsbook": "Sportsbook", "american_odds": "Moneyline",
                        "model_probability": "Model win", "break_even_probability": "Break-even", "market_no_vig_probability": "Market no-vig",
                        "edge_vs_no_vig_pp": "Edge vs market (pp)", "ev_pct": "EV %", "expected_net_profit": "Expected net profit",
                    })
                    st.dataframe(table.style.format({
                        "Moneyline": "{:+.0f}", "Model win": "{:.2%}", "Break-even": "{:.2%}", "Market no-vig": "{:.2%}",
                        "Edge vs market (pp)": "{:+.2f}", "EV %": "{:+.2f}%", "Expected net profit": "${:+.2f}",
                    }), use_container_width=True, hide_index=True)
                st.caption("EV is expected net profit as a percentage of stake, conditional on model accuracy. Market no-vig proportionally normalizes both prices from the same sportsbook quote.")
                with st.expander("Elo, rest, and data freshness"):
                    context_table = shown_predictions[["matchup", "away_elo", "home_elo", "away_rest_days", "home_rest_days", "away_last_game", "home_last_game"]].copy()
                    for col in ["away_last_game", "home_last_game"]:
                        context_table[col] = pd.to_datetime(context_table[col]).dt.strftime("%Y-%m-%d")
                    context_table = context_table.rename(columns={
                        "matchup": "Matchup", "away_elo": "Away Elo", "home_elo": "Home Elo", "away_rest_days": "Away rest",
                        "home_rest_days": "Home rest", "away_last_game": "Away latest game", "home_last_game": "Home latest game",
                    })
                    st.dataframe(context_table.style.format({"Away Elo": "{:.1f}", "Home Elo": "{:.1f}", "Away rest": "{:.0f}", "Home rest": "{:.0f}"}),
                                 use_container_width=True, hide_index=True)
                with st.expander("Model home-win probabilities"):
                    columns = [c for c in predictions if c.endswith("_home_probability")]
                    names = {c: DISPLAY_NAMES.get(c.removesuffix("_home_probability"), c) for c in columns}
                    model_table = shown_predictions[["matchup", *columns]].rename(columns={"matchup": "Matchup", **names})
                    st.dataframe(model_table.style.format({name: "{:.2%}" for name in names.values()}), use_container_width=True, hide_index=True)
                st.markdown("### Save this forecast")
                st.caption("The page works without exporting. Saving keeps a timestamped snapshot of all games and prices in this run.")
                if st.button("Save slate snapshot", use_container_width=True):
                    try:
                        output_dir = ROOT / "reports/game_outcomes"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
                        predictions.to_csv(output_dir / f"live_predictions_{ev_date}_{stamp}.csv", index=False)
                        ev_results.to_csv(output_dir / f"moneyline_ev_{ev_date}_{stamp}.csv", index=False)
                        st.success("Saved the complete forecast and EV snapshot.")
                    except Exception as exc:
                        st.error(f"Could not save the snapshot: {exc}")

