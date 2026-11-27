#!/usr/bin/env python3
"""Prepare football-data.co.uk match CSVs for analysis and prediction.

Reads every CSV in ``data/raw`` and writes a combined, model-ready file to
``data/processed/matches_processed.csv``. The original downloads are untouched.

Pre-match form features are calculated only from matches on earlier dates. This
avoids using a result from another fixture on the same date as prior knowledge.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "matches_processed.csv"
FORM_WINDOW = 5

REQUIRED_COLUMNS = {
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
}

# These are retained for descriptive analysis of completed matches. They are
# deliberately not used as features for predicting that same match's result.
MATCH_STAT_COLUMNS = (
    "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST", "HF", "AF",
    "HC", "AC", "HY", "AY", "HR", "AR",
)

HISTORY_METRICS = (
    "points",
    "goals_for",
    "goals_against",
    "goal_difference",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
)


def parse_number(value: str | None) -> float | None:
    """Convert a CSV numeric value to float, preserving blanks as None."""
    if value is None or not value.strip():
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def parse_date(value: str, path: Path, line_number: int) -> datetime:
    """Parse the source's day/month/year date, with a year-first fallback."""
    for date_format in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), date_format)
        except ValueError:
            continue
    raise ValueError(f"{path}:{line_number}: unrecognized Date value {value!r}")


def load_matches(paths: Iterable[Path]) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValueError(f"{path}: CSV has no header row")
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"{path}: missing required columns: {', '.join(sorted(missing))}"
                )

            for line_number, raw in enumerate(reader, start=2):
                home = (raw.get("HomeTeam") or "").strip()
                away = (raw.get("AwayTeam") or "").strip()
                if not home or not away:
                    raise ValueError(
                        f"{path}:{line_number}: HomeTeam and AwayTeam must be set"
                    )
                match_date = parse_date(raw.get("Date") or "", path, line_number)
                row: dict[str, str] = {
                    "date": match_date.strftime("%Y-%m-%d"),
                    "time": (raw.get("Time") or "").strip(),
                    "league": (raw.get("Div") or "").strip(),
                    "home_team": home,
                    "away_team": away,
                    "home_goals": (raw.get("FTHG") or "").strip(),
                    "away_goals": (raw.get("FTAG") or "").strip(),
                    "result": (raw.get("FTR") or "").strip().upper(),
                }
                if row["result"] not in {"H", "D", "A"}:
                    raise ValueError(
                        f"{path}:{line_number}: FTR must be H, D, or A; "
                        f"got {row['result']!r}"
                    )
                for column in MATCH_STAT_COLUMNS:
                    key = column.lower()
                    row[key] = (raw.get(column) or "").strip()
                row["_sort_time"] = (raw.get("Time") or "").strip()
                row["_source"] = path.name
                matches.append(row)

    # A stable sort allows multiple seasons and league files to be combined.
    matches.sort(key=lambda row: (row["date"], row["_sort_time"], row["_source"], row["home_team"]))
    return matches


def rolling_features(history: deque[dict[str, float | None]], prefix: str) -> dict[str, str]:
    features = {f"{prefix}_matches_last{FORM_WINDOW}": str(len(history))}
    for metric in HISTORY_METRICS:
        values = [match[metric] for match in history]
        observed = [value for value in values if value is not None]
        name = f"{prefix}_{metric}_avg_last{FORM_WINDOW}"
        features[name] = f"{sum(observed) / len(observed):.4f}" if observed else ""
    return features


def to_history_record(row: dict[str, str], team: str, is_home: bool) -> dict[str, float | None]:
    home_goals = parse_number(row["home_goals"])
    away_goals = parse_number(row["away_goals"])
    if home_goals is None or away_goals is None:
        raise ValueError(
            f"{row['_source']}: match {row['home_team']} vs {row['away_team']} "
            "has a missing or invalid full-time score"
        )

    if home_goals > away_goals:
        home_points, away_points = 3.0, 0.0
    elif home_goals < away_goals:
        home_points, away_points = 0.0, 3.0
    else:
        home_points = away_points = 1.0

    if is_home:
        goals_for, goals_against = home_goals, away_goals
        points = home_points
        shots_for, shots_against = parse_number(row["hs"]), parse_number(row["as"])
        sot_for, sot_against = parse_number(row["hst"]), parse_number(row["ast"])
    else:
        goals_for, goals_against = away_goals, home_goals
        points = away_points
        shots_for, shots_against = parse_number(row["as"]), parse_number(row["hs"])
        sot_for, sot_against = parse_number(row["ast"]), parse_number(row["hst"])

    return {
        "points": points,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_difference": goals_for - goals_against,
        "shots_for": shots_for,
        "shots_against": shots_against,
        "shots_on_target_for": sot_for,
        "shots_on_target_against": sot_against,
    }


def process(matches: list[dict[str, str]]) -> list[dict[str, str]]:
    form: dict[tuple[str, str], deque[dict[str, float | None]]] = defaultdict(
        lambda: deque(maxlen=FORM_WINDOW)
    )
    output: list[dict[str, str]] = []
    index = 0

    while index < len(matches):
        match_date = matches[index]["date"]
        end = index
        while end < len(matches) and matches[end]["date"] == match_date:
            end += 1

        # Snapshot form before processing any results on this date. Thus fixtures
        # on the same date cannot see each other's outcomes, even if kickoff times
        # are present or ordered in the source file.
        date_matches = matches[index:end]
        for row in date_matches:
            home_key = (row["league"], row["home_team"])
            away_key = (row["league"], row["away_team"])
            processed = {
                key: value
                for key, value in row.items()
                if not key.startswith("_")
            }
            processed.update(rolling_features(form[home_key], "home"))
            processed.update(rolling_features(form[away_key], "away"))
            output.append(processed)

        # Only after all fixtures on the date have their features do we add
        # their results to each club's history for future dates.
        for row in date_matches:
            home_key = (row["league"], row["home_team"])
            away_key = (row["league"], row["away_team"])
            form[home_key].append(to_history_record(row, row["home_team"], True))
            form[away_key].append(to_history_record(row, row["away_team"], False))
        index = end

    return output


def main() -> int:
    if not RAW_DIR.is_dir():
        print(f"Raw data directory not found: {RAW_DIR}", file=sys.stderr)
        return 1

    input_files = sorted(path for path in RAW_DIR.glob("*.csv") if path.is_file())
    if not input_files:
        print(f"No CSV files found in {RAW_DIR}", file=sys.stderr)
        return 1

    try:
        matches = load_matches(input_files)
        processed = process(matches)
    except (OSError, ValueError) as error:
        print(f"Preprocessing failed: {error}", file=sys.stderr)
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(processed[0].keys()) if processed else []
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed)

    print(f"Processed {len(matches)} matches from {len(input_files)} CSV file(s).")
    print(f"Saved processed data to: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
