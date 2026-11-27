"""Loading and validating the processed match data used by the app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_FILE = ROOT / "data" / "processed" / "matches_processed.csv"

REQUIRED_COLUMNS = {
    "date", "home_team", "away_team", "home_goals", "away_goals", "result",
    "home_matches_last5", "away_matches_last5",
}


def load_matches(path: Path = PROCESSED_FILE) -> pd.DataFrame:
    """Load processed matches and fail clearly if preparation is incomplete."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run `python3 src/preprocess_matches.py` first."
        )

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Processed data is missing columns: {', '.join(sorted(missing))}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("Processed data contains invalid dates.")
    for column in frame.columns:
        if column.startswith(("home_", "away_")) and column.endswith(("last5", "avg_last5")):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
    frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")
    frame["result"] = frame["result"].astype(str).str.upper()
    frame = frame.sort_values(["date", "time", "home_team"], kind="stable").reset_index(drop=True)
    return frame
