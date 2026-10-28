"""COURTSIDE — NBA predictive intelligence dashboard.

Run from the New_NBA project root:
    python -m streamlit run app.py

Expected project artifacts:
    data/raw/team_game_logs/team_game_logs_*.parquet
    reports/game_outcomes/stacking_validation_results.csv
    models/game_outcomes/{base_model}.joblib
    models/game_outcomes/stacking/logistic_stack.joblib
    models/game_outcomes/stacking/mlp_stack.joblib
"""
from __future__ import annotations

from datetime import date
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
    "knn",
    "logistic_regression",
    "decision_tree",
]

DISPLAY_NAMES = {
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
def read_results() -> pd.DataFrame:
    if not RESULTS.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULTS)


@st.cache_data(show_spinner=False)
def read_logs() -> pd.DataFrame:
    return load_logs()


@st.cache_resource(show_spinner=False)
def load_saved_models():
    missing = []
    base = {}

    for name in BASE_NAMES:
        path = MODEL_DIR / f"{name}.joblib"
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
        else:
            base[name] = joblib.load(path)

    logistic_path = STACK_DIR / "logistic_stack.joblib"
    mlp_path = STACK_DIR / "mlp_stack.joblib"

    if not logistic_path.exists():
        missing.append(str(logistic_path.relative_to(ROOT)))
    if not mlp_path.exists():
        missing.append(str(mlp_path.relative_to(ROOT)))

    if missing:
        raise FileNotFoundError("Missing model artifacts:\n" + "\n".join(missing))

    return base, joblib.load(logistic_path), joblib.load(mlp_path)


def build_team_snapshot(featured: pd.DataFrame, team: str, game_date: pd.Timestamp):
    history = featured[
        (featured["TEAM_ABBREVIATION"] == team)
        & (featured["GAME_DATE"] < game_date)
    ].sort_values(["GAME_DATE", "GAME_ID"])

    if history.empty:
        raise ValueError(f"No completed history was found for {team} before {game_date.date()}.")

    latest = history.iloc[-1]
    snapshot = {}

    # These use the completed historical box scores themselves so the most recent
    # game is included in the live rolling window. No current/future game is used.
    for stat in ROLLING_STATS:
        for window in WINDOWS:
            col = f"{stat.lower()}_avg_{window}"
            snapshot[col] = history[stat].tail(window).mean()

    snapshot["games_prior"] = len(history)
    snapshot["rest_days"] = max((game_date - latest["GAME_DATE"]).days - 1, 0)

    return snapshot, latest["GAME_DATE"]


def build_live_features(
    logs: pd.DataFrame,
    home_team: str,
    away_team: str,
    game_date: pd.Timestamp,
    features: list[str],
):
    historical_logs = logs[logs["GAME_DATE"] < game_date].copy()
    if historical_logs.empty:
        raise ValueError("No completed games exist before the selected prediction date.")

    featured = add_team_features(historical_logs)
    home, home_last = build_team_snapshot(featured, home_team, game_date)
    away, away_last = build_team_snapshot(featured, away_team, game_date)

    row = {}
    for key in home:
        row[f"home_{key}"] = home[key]
        row[f"away_{key}"] = away[key]
        row[f"diff_{key}"] = home[key] - away[key]

    X = pd.DataFrame([row])
    missing = [col for col in features if col not in X.columns]
    if missing:
        raise ValueError(f"Live feature builder is missing trained features: {missing}")

    X = X[features].replace([np.inf, -np.inf], np.nan)

    context = {
        "home_last_game": home_last,
        "away_last_game": away_last,
        "home_rest_days": home["rest_days"],
        "away_rest_days": away["rest_days"],
    }
    return X, context


def run_prediction(home_team: str, away_team: str, game_date: pd.Timestamp):
    logs = read_logs()
    base_models, logistic_stack, mlp_stack = load_saved_models()

    features = base_models["xgboost"]["features"]
    for name, saved in base_models.items():
        if list(saved["features"]) != list(features):
            raise ValueError(f"Feature order mismatch between XGBoost and {name}.")

    X, context = build_live_features(logs, home_team, away_team, game_date, features)

    probs = {}
    for name, saved in base_models.items():
        probs[name] = float(saved["pipeline"].predict_proba(X)[0, 1])

    base_frame = pd.DataFrame([{name: probs[name] for name in BASE_NAMES}])
    probs["simple_average"] = float(base_frame.mean(axis=1).iloc[0])
    probs["logistic_stack"] = float(
        logistic_stack["model"].predict_proba(base_frame[logistic_stack["base_names"]])[0, 1]
    )
    probs["mlp_stack"] = float(
        mlp_stack["model"].predict_proba(base_frame[mlp_stack["base_names"]])[0, 1]
    )

    return probs, context


def save_prediction(home_team, away_team, game_date, probs):
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "recorded_at": pd.Timestamp.now().isoformat(timespec="seconds"),
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


results = read_results()
if not results.empty:
    required = {"model", "accuracy", "roc_auc", "log_loss", "brier"}
    if not required.issubset(results.columns):
        st.error(f"Validation report is missing columns: {sorted(required - set(results.columns))}")
        st.stop()
    results["display_name"] = results["model"].map(
        lambda s: DISPLAY_NAMES.get(s, s.replace("_", " ").title())
    )


st.sidebar.markdown(
    '<div class="brand">🏀 COURTSIDE</div><div class="brand-sub">NBA PREDICTIVE INTELLIGENCE</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "WORKSPACE",
    ["Overview", "Model Lab", "Matchup Explorer", "Prediction History"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("RESEARCH BUILD  ·  V0.3")
st.sidebar.markdown('<span class="pill">● LIVE MODEL CONNECTION</span>', unsafe_allow_html=True)

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
<p>Pregame modeling &nbsp;·&nbsp; Six base learners &nbsp;·&nbsp; Chronological validation &nbsp;·&nbsp; Stacked ensembles</p></div>''',
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

    best_accuracy = results.loc[results["accuracy"].idxmax()]
    best_loss = results.loc[results["log_loss"].idxmin()]
    best_auc = results.loc[results["roc_auc"].idxmax()]
    cards = [
        ("BEST ACCURACY", f'{best_accuracy["accuracy"]:.2%}', best_accuracy["display_name"]),
        ("LOWEST LOG LOSS", f'{best_loss["log_loss"]:.4f}', best_loss["display_name"]),
        ("BEST ROC-AUC", f'{best_auc["roc_auc"]:.4f}', best_auc["display_name"]),
        ("VALIDATION GAMES", "1,225", "2025–26 season"),
    ]
    for column, (label, value, detail) in zip(st.columns(4, gap="medium"), cards):
        column.markdown(
            f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="detail">{detail}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">MODEL PERFORMANCE / 2025–26</div>', unsafe_allow_html=True)
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
<div class="panel-row"><strong>Base classifiers</strong><br>XGBoost · AdaBoost · GaussianNB · KNN · Logistic Regression · Decision Tree</div>
<div class="panel-row"><strong>Evaluation</strong><br>Chronological split · 2025–26 validation</div>
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
        logs = read_logs()
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
        prediction_date = st.date_input("Game date", value=date.today())

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
            with st.spinner("Building leakage-free pregame features and running the ensemble..."):
                try:
                    probs, context = run_prediction(home, away, pd.Timestamp(prediction_date))
                    st.session_state["latest_prediction"] = {
                        "home": home,
                        "away": away,
                        "date": pd.Timestamp(prediction_date),
                        "probs": probs,
                        "context": context,
                    }
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        latest = st.session_state.get("latest_prediction")
        if latest and latest["home"] == home and latest["away"] == away and latest["date"].date() == prediction_date:
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
                    save_prediction(home, away, pd.Timestamp(prediction_date), probs)
                    st.success(f"Saved to {PREDICTION_LOG.relative_to(ROOT)}")
            with right_save:
                st.caption(
                    "Saving before tipoff creates a prospective record that can later be scored without rewriting history."
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
