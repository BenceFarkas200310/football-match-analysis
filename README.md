# Football Match Analysis

Thesis project for preparing and analyzing football match data.

## Prepare the match data

Put one or more compatible football-data.co.uk CSV files in `data/raw/`, then run:

```bash
python3 src/preprocess_matches.py
```

The script leaves the source CSVs unchanged and writes the combined result to
`data/processed/matches_processed.csv`. It standardizes the main match fields,
keeps selected full-time statistics for descriptive analysis, and creates
home/away rolling form averages from each team's previous five matches. A
team's form for a match uses only earlier calendar dates; matches on the same
date do not contribute to one another's pre-match features.

The output's `result` column is the prediction target (`H`, `D`, or `A`).
Columns such as `hs`, `as`, `hst`, and `ast` describe the current completed
match and should not be used as pre-match predictors. Their rolling averages
are included as historical features instead.

## Run the model workflow and web app

Install the dependencies in a virtual environment, then prepare the dataset and
train/evaluate the model once:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python3 src/preprocess_matches.py
python3 src/train_model.py
streamlit run app.py
```

The training command evaluates a logistic regression model with a chronological
80/20 date split, then saves a final model trained on all eligible matches to
`models/outcome_model.joblib`. It also saves evaluation metrics to
`reports/metrics/evaluation.json`. The web app reads these saved artifacts on
startup; it does not retrain the model when users open a dashboard or request a
prediction.

The app contains four sections: a filtered match dashboard, team comparison,
hypothetical outcome prediction based on the latest five recorded matches, and
temporal model evaluation. Predictions use team form through the latest date in
the dataset. They are analysis estimates, not guarantees.
