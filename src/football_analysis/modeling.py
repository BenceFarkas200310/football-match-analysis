"""Leakage-aware model training, temporal evaluation, and future-match features."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABELS = ["H", "D", "A"]
FEATURES = [
    f"{side}_{metric}_avg_last5"
    for side in ("home", "away")
    for metric in (
        "points", "goals_for", "goals_against", "goal_difference",
        "shots_for", "shots_against", "shots_on_target_for", "shots_on_target_against",
    )
]


def eligible_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Use rows where both clubs have five prior matches for stable comparisons."""
    return matches.loc[
        (matches["home_matches_last5"] >= 5)
        & (matches["away_matches_last5"] >= 5)
        & matches["result"].isin(LABELS)
    ].copy()


def build_model() -> Pipeline:
    """Return a reproducible multiclass baseline with scaled numeric inputs."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def temporal_evaluation(matches: pd.DataFrame) -> tuple[dict[str, Any], Pipeline, pd.DataFrame]:
    """Evaluate on the latest 20% of eligible match dates, then fit final model."""
    data = eligible_matches(matches)
    dates = sorted(data["date"].drop_duplicates())
    if len(dates) < 10:
        raise ValueError("At least 10 eligible match dates are needed for temporal evaluation.")

    split_at = max(1, int(len(dates) * 0.8))
    cutoff = dates[split_at - 1]
    train = data.loc[data["date"] <= cutoff]
    test = data.loc[data["date"] > cutoff]
    if train.empty or test.empty or train["result"].nunique() < 2:
        raise ValueError("The chronological split does not contain enough result classes to evaluate.")

    evaluation_model = build_model()
    evaluation_model.fit(train[FEATURES], train["result"])
    predictions = evaluation_model.predict(test[FEATURES])
    precision, recall, f1, support = precision_recall_fscore_support(
        test["result"], predictions, labels=LABELS, zero_division=0
    )
    matrix = confusion_matrix(test["result"], predictions, labels=LABELS)

    # Persist a separate final model trained on every eligible historical match.
    final_model = build_model()
    final_model.fit(data[FEATURES], data["result"])

    metrics: dict[str, Any] = {
        "evaluation_method": "Chronological holdout by match date (first 80% train, latest 20% test)",
        "cutoff_date": cutoff.strftime("%Y-%m-%d"),
        "train_matches": int(len(train)),
        "test_matches": int(len(test)),
        "eligible_matches": int(len(data)),
        "accuracy": float(accuracy_score(test["result"], predictions)),
        "class_metrics": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": matrix.astype(int).tolist(),
    }
    return metrics, final_model, data


def latest_team_form(matches: pd.DataFrame, team: str, league: str) -> dict[str, float]:
    """Calculate a team's latest five-match form for a hypothetical next fixture."""
    subset = matches.loc[matches["league"].astype(str) == str(league)]
    team_rows: list[dict[str, float | pd.Timestamp]] = []

    for _, row in subset.iterrows():
        if row["home_team"] == team:
            points = 3.0 if row["home_goals"] > row["away_goals"] else 1.0 if row["home_goals"] == row["away_goals"] else 0.0
            record = {
                "date": row["date"],
                "points": points,
                "goals_for": row["home_goals"],
                "goals_against": row["away_goals"],
                "goal_difference": row["home_goals"] - row["away_goals"],
                "shots_for": row.get("hs"),
                "shots_against": row.get("as"),
                "shots_on_target_for": row.get("hst"),
                "shots_on_target_against": row.get("ast"),
            }
            team_rows.append(record)
        elif row["away_team"] == team:
            points = 3.0 if row["away_goals"] > row["home_goals"] else 1.0 if row["away_goals"] == row["home_goals"] else 0.0
            record = {
                "date": row["date"],
                "points": points,
                "goals_for": row["away_goals"],
                "goals_against": row["home_goals"],
                "goal_difference": row["away_goals"] - row["home_goals"],
                "shots_for": row.get("as"),
                "shots_against": row.get("hs"),
                "shots_on_target_for": row.get("ast"),
                "shots_on_target_against": row.get("hst"),
            }
            team_rows.append(record)

    if len(team_rows) < 5:
        raise ValueError(f"{team} has fewer than five recorded matches in this dataset.")

    history = pd.DataFrame(team_rows).sort_values("date").tail(5)
    return {
        f"{metric}_avg_last5": float(history[metric].mean())
        for metric in (
            "points", "goals_for", "goals_against", "goal_difference",
            "shots_for", "shots_against", "shots_on_target_for", "shots_on_target_against",
        )
    }


def prediction_row(matches: pd.DataFrame, home_team: str, away_team: str, league: str) -> pd.DataFrame:
    if home_team == away_team:
        raise ValueError("Choose two different teams.")
    home_form = latest_team_form(matches, home_team, league)
    away_form = latest_team_form(matches, away_team, league)
    values = {f"home_{key}": value for key, value in home_form.items()}
    values.update({f"away_{key}": value for key, value in away_form.items()})
    return pd.DataFrame([[values[column] for column in FEATURES]], columns=FEATURES)
