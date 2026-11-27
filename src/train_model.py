#!/usr/bin/env python3
"""Evaluate and persist the football outcome model and its metrics."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from football_analysis.data import PROCESSED_FILE, load_matches  # noqa: E402
from football_analysis.modeling import FEATURES, temporal_evaluation  # noqa: E402


MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "metrics"
MODEL_FILE = MODEL_DIR / "outcome_model.joblib"
METRICS_FILE = REPORT_DIR / "evaluation.json"


def main() -> int:
    try:
        matches = load_matches(PROCESSED_FILE)
        metrics, model, training_data = temporal_evaluation(matches)
    except (OSError, ValueError) as error:
        print(f"Model preparation failed: {error}", file=sys.stderr)
        return 1

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "features": FEATURES,
        "labels": ["H", "D", "A"],
        "league": sorted(matches["league"].dropna().astype(str).unique().tolist()),
        "trained_through": matches["date"].max().strftime("%Y-%m-%d"),
        "training_matches": int(len(training_data)),
    }
    joblib.dump(bundle, MODEL_FILE)
    metrics["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    metrics["source_file"] = str(PROCESSED_FILE.relative_to(ROOT))
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved trained model: {MODEL_FILE}")
    print(f"Saved temporal evaluation: {METRICS_FILE}")
    print(f"Holdout accuracy: {metrics['accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
