#!/usr/bin/env python3
"""Streamlit interface for football match analysis and outcome estimates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from football_analysis.data import PROCESSED_FILE, load_matches  # noqa: E402
from football_analysis.modeling import prediction_row  # noqa: E402


MODEL_FILE = ROOT / "models" / "outcome_model.joblib"
METRICS_FILE = ROOT / "reports" / "metrics" / "evaluation.json"

st.set_page_config(page_title="Football Match Analysis", page_icon="⚽", layout="wide")


@st.cache_data
def get_matches() -> pd.DataFrame:
    return load_matches(PROCESSED_FILE)


@st.cache_resource
def get_model_bundle() -> dict:
    return joblib.load(MODEL_FILE)


def show_dashboard(matches: pd.DataFrame) -> None:
    st.title("Match analysis dashboard")
    st.caption("Explore completed Premier League matches and their recorded statistics.")

    teams = sorted(set(matches["home_team"]).union(matches["away_team"]))
    selected_team = st.selectbox("Team", ["All teams", *teams])
    date_min, date_max = matches["date"].min().date(), matches["date"].max().date()
    selected_dates = st.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
    filtered = matches.copy()
    if selected_team != "All teams":
        filtered = filtered.loc[
            (filtered["home_team"] == selected_team) | (filtered["away_team"] == selected_team)
        ]
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start, end = selected_dates
        filtered = filtered.loc[
            (filtered["date"].dt.date >= start) & (filtered["date"].dt.date <= end)
        ]

    if filtered.empty:
        st.info("No matches match those filters.")
        return

    home_wins = int((filtered["result"] == "H").sum())
    draws = int((filtered["result"] == "D").sum())
    away_wins = int((filtered["result"] == "A").sum())
    goals = filtered["home_goals"].sum() + filtered["away_goals"].sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Matches", f"{len(filtered):,}")
    k2.metric("Goals", f"{int(goals):,}")
    k3.metric("Goals per match", f"{goals / len(filtered):.2f}")
    k4.metric("Home wins", f"{home_wins} ({home_wins / len(filtered):.0%})")

    left, right = st.columns(2)
    with left:
        st.subheader("Match outcomes")
        st.bar_chart(pd.Series({"Home win": home_wins, "Draw": draws, "Away win": away_wins}, name="Matches"))
    with right:
        st.subheader("Goals over time")
        goal_trend = filtered.set_index("date")[["home_goals", "away_goals"]].rename(
            columns={"home_goals": "Home goals", "away_goals": "Away goals"}
        ).resample("MS").sum()
        st.line_chart(goal_trend)

    st.subheader("Match records")
    display_columns = [
        "date", "home_team", "away_team", "home_goals", "away_goals", "result",
        "hs", "as", "hst", "ast", "hc", "ac",
    ]
    st.dataframe(filtered[display_columns].sort_values("date", ascending=False), use_container_width=True, hide_index=True)


def show_comparison(matches: pd.DataFrame) -> None:
    st.title("Team comparison")
    st.caption("Compare season results and match statistics across the selected teams.")
    teams = sorted(set(matches["home_team"]).union(matches["away_team"]))
    first, second = st.columns(2)
    home_team = first.selectbox("First team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
    away_team = second.selectbox("Second team", teams, index=teams.index("Liverpool") if "Liverpool" in teams else min(1, len(teams) - 1))

    def team_summary(team: str) -> dict[str, float | int | str]:
        home = matches.loc[matches["home_team"] == team]
        away = matches.loc[matches["away_team"] == team]
        won = int((home["result"] == "H").sum() + (away["result"] == "A").sum())
        drawn = int((home["result"] == "D").sum() + (away["result"] == "D").sum())
        lost = int(len(home) + len(away) - won - drawn)
        gf = int(home["home_goals"].sum() + away["away_goals"].sum())
        ga = int(home["away_goals"].sum() + away["home_goals"].sum())
        played = len(home) + len(away)
        points = won * 3 + drawn
        return {
            "Team": team, "Played": played, "Points": points,
            "Wins": won, "Draws": drawn, "Losses": lost,
            "Goals for": gf, "Goals against": ga, "Goal difference": gf - ga,
            "Avg shots": pd.concat([home["hs"], away["as"]]).mean(),
            "Avg shots on target": pd.concat([home["hst"], away["ast"]]).mean(),
        }

    comparison = pd.DataFrame([team_summary(home_team), team_summary(away_team)]).set_index("Team")
    st.dataframe(comparison.T.rename(columns={home_team: home_team, away_team: away_team}), use_container_width=True)
    chart_metrics = ["Points", "Goals for", "Goals against", "Avg shots", "Avg shots on target"]
    st.bar_chart(comparison[chart_metrics].T)


def show_prediction(matches: pd.DataFrame) -> None:
    st.title("Outcome prediction")
    st.caption("Estimate home/draw/away probabilities for a hypothetical fixture after the latest match in the dataset.")
    if not MODEL_FILE.exists():
        st.warning("The saved model is not available yet. Run `python3 src/train_model.py` from the project folder.")
        return

    bundle = get_model_bundle()
    leagues = bundle.get("league") or sorted(matches["league"].dropna().astype(str).unique())
    league = st.selectbox("League", leagues)
    league_matches = matches.loc[matches["league"].astype(str) == str(league)]
    teams = sorted(set(league_matches["home_team"]).union(league_matches["away_team"]))
    left, right = st.columns(2)
    home_team = left.selectbox("Home team", teams, key="predict_home")
    away_options = [team for team in teams if team != home_team]
    away_team = right.selectbox("Away team", away_options, key="predict_away")
    st.info(f"Team form uses the latest completed matches available through {bundle['trained_through']}.")

    if st.button("Estimate outcome", type="primary"):
        try:
            features = prediction_row(matches, home_team, away_team, str(league))
            model = bundle["model"]
            probabilities = model.predict_proba(features)[0]
            classes = model.named_steps["classifier"].classes_
            result = pd.Series(probabilities, index=classes).reindex(["H", "D", "A"], fill_value=0.0)
            result.index = ["Home win", "Draw", "Away win"]
            st.subheader("Estimated probabilities")
            st.bar_chart((result * 100).rename("Probability (%)"))
            st.dataframe((result * 100).round(1).rename("Probability (%)"), use_container_width=True)
            st.caption("This is a model estimate for analysis, not a guaranteed result.")
        except ValueError as error:
            st.error(str(error))


def show_evaluation() -> None:
    st.title("Model evaluation")
    st.caption("Metrics come from a chronological holdout: earlier dates for training, later dates for evaluation.")
    if not METRICS_FILE.exists():
        st.warning("Saved evaluation results are not available yet. Run `python3 src/train_model.py` from the project folder.")
        return

    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Holdout accuracy", f"{metrics['accuracy']:.1%}")
    k2.metric("Training matches", metrics["train_matches"])
    k3.metric("Test matches", metrics["test_matches"])
    k4.metric("Train/test cutoff", metrics["cutoff_date"])
    st.write(metrics["evaluation_method"])

    st.subheader("Per-outcome metrics")
    metric_table = pd.DataFrame(metrics["class_metrics"]).T.rename(
        index={"H": "Home win", "D": "Draw", "A": "Away win"}
    )
    st.dataframe(metric_table.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1": "{:.2f}"}), use_container_width=True)

    st.subheader("Confusion matrix")
    labels = metrics["confusion_matrix_labels"]
    names = {"H": "Home win", "D": "Draw", "A": "Away win"}
    confusion = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"Actual {names[label]}" for label in labels],
        columns=[f"Predicted {names[label]}" for label in labels],
    )
    st.dataframe(confusion, use_container_width=True)


def main() -> None:
    st.sidebar.title("Football analysis")
    page = st.sidebar.radio(
        "Section",
        ["Match dashboard", "Team comparison", "Outcome prediction", "Model evaluation"],
    )
    try:
        matches = get_matches()
    except (OSError, ValueError) as error:
        st.error(str(error))
        st.stop()

    if page == "Match dashboard":
        show_dashboard(matches)
    elif page == "Team comparison":
        show_comparison(matches)
    elif page == "Outcome prediction":
        show_prediction(matches)
    else:
        show_evaluation()


if __name__ == "__main__":
    main()
