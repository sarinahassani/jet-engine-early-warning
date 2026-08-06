# Jet Engine Early Warning System

Streamlit dashboard for multi-stage jet engine prognostics (CMAPSS / NASA turbofan).

This app **only consumes artifacts** exported by the training notebooks. It never retrains models and never re-runs feature engineering. Numbers shown for a given engine and cycle should match what the corresponding notebook printed for the same inputs (**app parity**).

---

## What it does

For each stage (dataset), after you upload the exported files you can:

1. Select a **test engine** and **cycle**
2. Choose a **risk horizon** (e.g. 10 / 20 / 30 cycles) used by the action policy
3. See:
   - **RUL** point estimate + 90% conformal interval
   - **Failure risk** probabilities at each horizon
   - **Anomaly** score (PCA reconstruction error) + percentile vs healthy train reference
   - **ACTION** recommendation: `CONTINUE` / `INSPECT` / `STOP`
   - **Timeline** of a representative sensor for that engine

Stages are independent (separate tabs and session state).

| Tab | Dataset |
|-----|---------|
| Stage 1 (FD001) | FD001 |
| Stage 2 (FD003) | FD003 |

---

## Required artifacts (per stage)

Upload via the UI (multi-select). Filenames are matched by convention:

| Role | Filename pattern | Source |
|------|------------------|--------|
| **Required** | `prognostics_system_<DATASET>.joblib` | Task 7 export |
| **Required** | `<DATASET>_test_features.joblib` | Task 2 export (already-engineered test features) |
| Optional | `task3_metadata_<DATASET>.json` | Task 3 export (shows best model name) |

Example for FD001:

```text
prognostics_system_FD001.joblib
FD001_test_features.joblib
task3_metadata_FD001.json   # optional
```

Nothing is read from disk by path — only files you upload at runtime.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

### `requirements.txt`

```text
streamlit
pandas
numpy
joblib
altair
scikit-learn
```

If your pickled models need extra libraries (e.g. `xgboost`, `lightgbm`, `catboost`), install those as well so `joblib.load` can unpickle successfully.

---

## Usage

1. Open a stage tab (e.g. **Stage 1 (FD001)**).
2. Upload that stage’s `.joblib` / `.json` files.
3. Click **Load files**.
4. Pick **Engine**, **Cycle**, and **Risk horizon**.
5. Read the evidence cards and the sensor timeline.

The cycle slider is limited to the selected engine’s available cycles (test trajectories are truncated). The risk horizon controls which probability the **ACTION** policy uses; all horizons still appear in the Failure risk card.

### Action policy (default)

| Condition | Action |
|-----------|--------|
| Lower RUL ≤ 5 **or** anomaly ≥ stop threshold **or** P(fail) ≥ 0.7 | **STOP** |
| Lower RUL ≤ 20 **or** anomaly ≥ warn threshold **or** P(fail) ≥ 0.3 | **INSPECT** |
| Otherwise | **CONTINUE** |

`P(fail)` is taken at the **selected risk horizon**.

---

## Deploy (Hugging Face Spaces)

1. Create a Space with SDK **Streamlit**.
2. Upload `app.py`, `requirements.txt`, and this `README.md`.
3. Users upload artifacts in the UI at runtime — no need to commit large `.joblib` files into the Space repo (unless you want demos preloaded).

---

## Design notes

- **No feature engineering in the app.** Task 2 already exports `*_test_features.joblib` (`window_test_df`: `engine_id`, `cycle`, active sensors, lags / rolling / ewm / slope / residual). Scoring is a **row lookup**, not a live transform.
- **`PrognosticsSystem`** is defined in `app.py` only so `joblib` can unpickle the class reference saved by the notebook.
- Stage tabs do not share state; Stage 1 and Stage 2 cannot cross-contaminate.

---

## Project layout

```text
.
├── app.py              # Streamlit dashboard
├── requirements.txt
└── README.md
```

Training notebooks and raw CMAPSS files live elsewhere; this repo is the inference UI only.



## Dashboard: jet-engine-early-warning-q6szn5fbxqd8c7o9uqumdo.streamlit.app
