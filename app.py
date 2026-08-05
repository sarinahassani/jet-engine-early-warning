"""
Jet Engine Early Warning System dashboard -- multi-stage edition.
Deliberately kept OUT of the training notebook(s) (see Task 0 notes there):
this file is what runs on Hugging Face Spaces. It only ever loads artifacts
that a notebook already exported -- it never retrains anything, and it
never re-runs feature engineering -- so "app parity" (Section 5, Task 0 of
the brief) means the numbers you see here should match what the
corresponding notebook printed for the same engine/cycle.
WHAT THIS VERSION DOES DIFFERENTLY FROM THE FIRST DRAFT
----------------------------------------------------------
Task 2 does NOT export a raw test_<DATASET>.txt/csv file -- it exports
<DATASET>_test_features.joblib, which is already the fully engineered
feature table (`window_test_df` from the notebook: engine_id, cycle, the
active sensor columns, and every lag/rolling/ewm/slope/residual feature).
So this app:
  1. Loads that table directly instead of a raw text file, and scores a
     cycle with a row lookup instead of re-running feature engineering.
  2. Drops FeaturePipeline / Task2Config / the transform helpers entirely
     -- they're only needed to PRODUCE test_features.joblib in the
     notebook, never to consume it, so there is nothing left in this file
     that could duplicate (or diverge from) the notebook's copy.
  3. Requires only two files per stage: prognostics_system_<DATASET>.joblib
     and <DATASET>_test_features.joblib. task3_metadata_<DATASET>.json is
     optional -- it only adds a "best model" line to the display.
Multi-stage support and the upload-only (no hardcoded path) design are
unchanged from the first draft: one independent tab per stage, nothing
read from local disk, everything supplied through the UI.
Run locally with: streamlit run app.py
Deploy on HF Spaces by uploading this file + requirements.txt. Artifacts
are supplied by the user at runtime through the upload widgets -- nothing
needs to be pre-populated in the Space's file system.
"""
import re
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import altair as alt

# ---------------------------------------------------------------------------
# The two stages this app supports, and the artifact-filename prefix each
# notebook uses (matches the {DATASET_NAME} convention throughout both
# notebooks: prognostics_system_{DATASET_NAME}.joblib, etc.)
# ---------------------------------------------------------------------------
STAGES = [
    {"label": "Stage 1 (FD001)", "dataset_name": "FD001"},
    {"label": "Stage 2 (FD003)", "dataset_name": "FD003"},
]

# ---------------------------------------------------------------------------
# PrognosticsSystem -- SINGLE definition, needed only because joblib
# unpickles custom classes by reference: whatever pickled the object (Task 7
# of the notebook) needs a structurally-matching class of the same name
# available wherever it's loaded. This file no longer needs FeaturePipeline
# at all, since it never transforms raw data -- it only reads a feature
# table the notebook already produced.
# ---------------------------------------------------------------------------
def pca_reconstruction_error(X, pca_model):
    Xr = pca_model.inverse_transform(pca_model.transform(X))
    return np.mean((X - Xr) ** 2, axis=1)


class PrognosticsSystem:
    def __init__(self, rul_model, rul_cols, conformal_q,
                 clf_models_by_h, clf_cols, calibrators_by_h, clf_thresholds_by_h,
                 anomaly_imputer, anomaly_scaler, anomaly_cols, anomaly_detector, anomaly_thresholds,
                 detector_healthy_ref, detector_name="PCA"):
        self.rul_model, self.rul_cols, self.conformal_q = rul_model, rul_cols, conformal_q
        self.clf_models_by_h, self.clf_cols = clf_models_by_h, clf_cols
        self.calibrators_by_h, self.clf_thresholds_by_h = calibrators_by_h, clf_thresholds_by_h
        self.anomaly_imputer, self.anomaly_scaler, self.anomaly_cols = anomaly_imputer, anomaly_scaler, anomaly_cols
        self.anomaly_detector, self.anomaly_thresholds, self.detector_name = anomaly_detector, anomaly_thresholds, detector_name
        self._ref_sorted = np.sort(np.asarray(detector_healthy_ref))

    def predict_rul(self, features: pd.DataFrame, interval: bool = True):
        point = self.rul_model.predict(features[self.rul_cols])
        if not interval:
            return point
        return {"point": point, "lower": point - self.conformal_q, "upper": point + self.conformal_q}

    def failure_risk(self, features: pd.DataFrame, horizons=(10, 20, 30), model_name="RandomForest"):
        return {h: self.calibrators_by_h[h][model_name].predict_proba(features[self.clf_cols])[:, 1] for h in horizons}

    def anomaly_score(self, features: pd.DataFrame):
        X = self.anomaly_scaler.transform(self.anomaly_imputer.transform(features[self.anomaly_cols]))
        raw = pca_reconstruction_error(X, self.anomaly_detector)
        pct = np.array([float(np.clip(np.searchsorted(self._ref_sorted, r) / len(self._ref_sorted) * 100, 0, 100)) for r in raw])
        return {"raw": raw, "percentile": pct}

    def recommend(self, outputs: dict, policy: dict = None):
        policy = policy or {"risk_horizon": 10, "proba_warn": 0.3}
        n = len(outputs["rul"]["point"])
        actions = []
        for i in range(n):
            rul_lower = outputs["rul"]["lower"][i]
            proba = outputs["risk"][policy["risk_horizon"]][i]
            anom_raw = outputs["anomaly"]["raw"][i]
            thr = self.anomaly_thresholds[self.detector_name]
            if rul_lower <= 5 or anom_raw >= thr["stop"] or proba >= 0.7:
                action, why = "STOP", "critical lower RUL bound or high validated near-term failure risk"
            elif rul_lower <= 20 or anom_raw >= thr["warn"] or proba >= policy["proba_warn"]:
                action, why = "INSPECT", "elevated risk, persistent anomaly, or wide uncertainty"
            else:
                action, why = "CONTINUE", "comfortable RUL, low risk, stable anomaly score"
            actions.append({"action": action, "trigger": why})
        return actions


# ---------------------------------------------------------------------------
# Upload handling -- replaces every hardcoded file path. Only two files are
# actually required per stage; a third (metadata) is optional.
# ---------------------------------------------------------------------------
REQUIRED_FILES = {
    "system": {
        "match": lambda name: name.endswith(".joblib") and "prognostics_system" in name,
        "description": "prognostics_system_<DATASET>.joblib (Task 7 export)",
    },
    "test_features": {
        "match": lambda name: name.endswith(".joblib") and "test_features" in name,
        "description": "<DATASET>_test_features.joblib (Task 2 export -- already-engineered test features)",
    },
}

OPTIONAL_FILES = {
    "metadata": {
        "match": lambda name: name.endswith(".json") and "metadata" in name,
        "description": "task3_metadata_<DATASET>.json (Task 3 export, optional -- adds model name to display)",
    },
}


def classify_uploaded_file(name: str) -> Optional[str]:
    """Identify which role an uploaded file fills, by filename -- the
    notebooks' own {name}_{DATASET_NAME} convention makes this reliable
    without asking the user to label anything by hand."""
    name_lower = name.lower()
    for role, spec in {**REQUIRED_FILES, **OPTIONAL_FILES}.items():
        if spec["match"](name_lower):
            return role
    return None


def load_stage_files(
    uploaded_files: Optional[List[Any]],
    fallback_dataset_name: str,
) -> Tuple[str, Optional[Any], Optional[pd.DataFrame], dict, str, str, List[int], int]:
    """Validate and load everything one stage tab needs from its uploaded
    files. Returns (status_message, system, test_features_df, metadata,
    dataset_name, metadata_markdown, engine_choices, max_cycle_overall).
    Clean file handling: uploaded files are read directly from the paths
    Streamlit already staged for us -- nothing is copied elsewhere, and no
    extra temp files are created by this app.
    """
    empty_engines: List[int] = []
    if not uploaded_files:
        msg = "⚠️ No files uploaded yet. Upload the required files listed above, then click **Load files**."
        return (
            msg,
            None,
            None,
            {},
            fallback_dataset_name,
            "Model metadata will appear here once files are loaded.",
            empty_engines,
            400,
        )

    found: Dict[str, Any] = {}
    detected_dataset = None
    for f in uploaded_files:
        role = classify_uploaded_file(f.name)
        if role and role not in found:  # first match per role wins
            found[role] = f
        m = re.search(r"(FD00[1-4])", f.name, re.IGNORECASE)
        if m and detected_dataset is None:
            detected_dataset = m.group(1).upper()

    missing = [spec["description"] for role, spec in REQUIRED_FILES.items() if role not in found]
    if missing:
        msg = "❌ Missing required file(s):\n- " + "\n- ".join(missing)
        return (
            msg,
            None,
            None,
            {},
            fallback_dataset_name,
            "Model metadata will appear here once files are loaded.",
            empty_engines,
            400,
        )

    dataset_name = detected_dataset or fallback_dataset_name
    if detected_dataset and detected_dataset != fallback_dataset_name:
        dataset_note = (
            f" (note: filenames indicate {detected_dataset}, "
            f"not this tab's expected {fallback_dataset_name} -- "
            f"double-check you uploaded the right stage's files)"
        )
    else:
        dataset_note = ""

    try:
        system = joblib.load(found["system"])
        test_features_df = joblib.load(found["test_features"])
        if "metadata" in found:
            metadata = json.loads(found["metadata"].read())
        else:
            metadata = {}
    except Exception as e:  # surfaced to the user, not just the server log
        msg = f"❌ Failed to load uploaded files: {e}"
        return (
            msg,
            None,
            None,
            {},
            fallback_dataset_name,
            "Model metadata will appear here once files are loaded.",
            empty_engines,
            400,
        )

    if "engine_id" not in test_features_df.columns or "cycle" not in test_features_df.columns:
        msg = "❌ The uploaded test_features file doesn't look like a Task 2 export (missing engine_id/cycle columns)."
        return (
            msg,
            None,
            None,
            {},
            fallback_dataset_name,
            "Model metadata will appear here once files are loaded.",
            empty_engines,
            400,
        )

    engine_choices = sorted(test_features_df["engine_id"].unique().tolist())
    max_cycle_overall = int(test_features_df.groupby("engine_id")["cycle"].max().max())
    metadata_note = "" if "metadata" in found else " (no metadata file uploaded -- model name unavailable)"
    status = (
        f"✅ Loaded {dataset_name} successfully{dataset_note} -- "
        f"{len(engine_choices)} test engines available.{metadata_note}"
    )
    meta_text = (
        f"**Model metadata** -- dataset: {dataset_name}, "
        f"model: {metadata.get('best_model', 'n/a')}, "
        f"feature window: causal lag/rolling/ewm/slope/residual (Task 2)"
    )
    return (
        status,
        system,
        test_features_df,
        metadata,
        dataset_name,
        meta_text,
        engine_choices,
        max_cycle_overall,
    )


# ---------------------------------------------------------------------------
# Scoring -- a row lookup into the precomputed feature table, not a live
# feature transform. Everything needed is passed in explicitly (from this
# stage tab's session state), so Stage 1 and Stage 2 tabs can never
# cross-contaminate each other.
# ---------------------------------------------------------------------------
def _pick_timeline_sensor(columns) -> Optional[str]:
    """Prefer sensor_2 (matches the original app); otherwise the first raw
    sensor_* column; finally fall back to any column that looks like a
    sensor measurement (e.g. sensor_2_lag_0 or similar) so the timeline
    still renders even when the notebook dropped pure raw columns."""
    cols = list(columns)
    if "sensor_2" in cols:
        return "sensor_2"
    exact = sorted(c for c in cols if re.fullmatch(r"sensor_\d+", str(c)))
    if exact:
        return exact[0]
    # engineered variants that still carry the live sensor reading
    soft = sorted(
        c for c in cols
        if re.match(r"sensor_\d+", str(c)) and not any(
            tok in str(c).lower() for tok in ("lag_", "roll", "ewm", "slope", "resid")
        )
    )
    if soft:
        return soft[0]
    soft2 = sorted(c for c in cols if re.match(r"sensor_\d+", str(c)))
    return soft2[0] if soft2 else None


def score_engine_at_cycle(
    system,
    test_features_df,
    engine_id,
    cycle,
    risk_horizon: int = 10,
    horizons=(10, 20, 30),
):
    if system is None or test_features_df is None:
        return "Upload and load this stage's files first.", None, None, None, None, 1, 1
    if engine_id is None:
        return "Select an engine.", None, None, None, None, 1, 1
    history = test_features_df[test_features_df.engine_id == engine_id].sort_values("cycle")
    if history.empty:
        return f"No data for engine {engine_id}.", None, None, None, None, 1, 1
    max_cycle = int(history["cycle"].max())
    cycle = int(min(max(cycle, 1), max_cycle))
    feats = history[history["cycle"] == cycle]
    if feats.empty:
        return "No feature row for this cycle.", None, None, None, None, cycle, max_cycle

    # Ensure the selected horizon is included in the scored set
    horizons = tuple(sorted(set(list(horizons) + [int(risk_horizon)])))
    rul_out = system.predict_rul(feats)
    risk_out = system.failure_risk(feats, horizons=horizons)
    anom_out = system.anomaly_score(feats)
    action = system.recommend(
        {"rul": rul_out, "risk": risk_out, "anomaly": anom_out},
        policy={"risk_horizon": int(risk_horizon), "proba_warn": 0.3},
    )[0]
    rul_card = (
        f"Point estimate: {rul_out['point'][0]:.1f} cycles\n"
        f"90% interval: [{rul_out['lower'][0]:.1f}, {rul_out['upper'][0]:.1f}] cycles"
    )
    risk_lines = []
    for h in risk_out:
        marker = "  ← selected" if h == int(risk_horizon) else ""
        risk_lines.append(f"P(failure within {h} cycles) = {risk_out[h][0]:.2f}{marker}")
    risk_card = "\n".join(risk_lines)
    anomaly_card = (
        f"Normalized score (raw): {anom_out['raw'][0]:.3f}\n"
        f"Percentile vs. healthy-train reference: {anom_out['percentile'][0]:.1f}%"
    )
    action_card = (
        f"ACTION: {action['action']}\n"
        f"Trigger: {action['trigger']}\n"
        f"Policy horizon: {int(risk_horizon)} cycles"
    )
    sensor_col = _pick_timeline_sensor(history.columns)
    if sensor_col is not None:
        timeline_df = history[["cycle", sensor_col]].copy()
        timeline_df = timeline_df.rename(columns={sensor_col: "value"})
        timeline_df["sensor"] = sensor_col  # keep name for the chart title
    else:
        timeline_df = pd.DataFrame({"cycle": history["cycle"], "value": np.nan, "sensor": None})
    return rul_card, risk_card, anomaly_card, action_card, timeline_df, cycle, max_cycle


# ---------------------------------------------------------------------------
# Streamlit UI -- same visual structure as the original Gradio app (Markdown
# header, engine/cycle Row, timeline Row, two Rows of evidence cards,
# metadata footer), one instance per stage tab.
# ---------------------------------------------------------------------------
def render_stage_tab(stage_label: str, default_dataset_name: str, stage_key: str):
    """Render one stage tab. All state is keyed by stage_key so Stage 1 and
    Stage 2 remain fully independent."""
    st.markdown(
        f"### {stage_label} -- upload the files this stage's notebook exported, "
        f"then click **Load files**:\n"
        + "\n".join(f"- `{spec['description']}` (required)" for spec in REQUIRED_FILES.values())
        + "\n"
        + "\n".join(f"- `{spec['description']}`" for spec in OPTIONAL_FILES.values())
    )

    uploaded_files = st.file_uploader(
        "Upload files (multi-select)",
        type=["joblib", "json"],
        accept_multiple_files=True,
        key=f"upload_{stage_key}",
    )

    # Initialize per-stage session state
    for key, default in [
        (f"system_{stage_key}", None),
        (f"test_features_{stage_key}", None),
        (f"metadata_{stage_key}", {}),
        (f"dataset_name_{stage_key}", default_dataset_name),
        (f"status_{stage_key}", "No files loaded yet."),
        (f"meta_text_{stage_key}", "Model metadata will appear here once files are loaded."),
        (f"engines_{stage_key}", []),
        (f"max_cycle_{stage_key}", 400),
        (f"loaded_{stage_key}", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if st.button("Load files", type="primary", key=f"load_btn_{stage_key}"):
        (
            status,
            system,
            test_features_df,
            metadata,
            dataset_name,
            meta_text,
            engine_choices,
            max_cycle_overall,
        ) = load_stage_files(uploaded_files, st.session_state[f"dataset_name_{stage_key}"])
        st.session_state[f"status_{stage_key}"] = status
        st.session_state[f"system_{stage_key}"] = system
        st.session_state[f"test_features_{stage_key}"] = test_features_df
        st.session_state[f"metadata_{stage_key}"] = metadata
        st.session_state[f"dataset_name_{stage_key}"] = dataset_name
        st.session_state[f"meta_text_{stage_key}"] = meta_text
        st.session_state[f"engines_{stage_key}"] = engine_choices
        st.session_state[f"max_cycle_{stage_key}"] = max_cycle_overall
        st.session_state[f"loaded_{stage_key}"] = system is not None

    st.markdown(st.session_state[f"status_{stage_key}"])
    st.markdown("---")

    engines = st.session_state[f"engines_{stage_key}"]
    max_cycle_overall = st.session_state[f"max_cycle_{stage_key}"]
    system = st.session_state[f"system_{stage_key}"]
    test_features_df = st.session_state[f"test_features_{stage_key}"]

    # Horizons available from the trained calibrators (fallback to the
    # standard set if the system object isn't loaded yet).
    default_horizons = [10, 20, 30]
    available_horizons = default_horizons
    if system is not None and getattr(system, "calibrators_by_h", None):
        available_horizons = sorted(int(h) for h in system.calibrators_by_h.keys())
        if not available_horizons:
            available_horizons = default_horizons

    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        engine_id = st.selectbox(
            "Engine",
            options=engines if engines else [None],
            format_func=lambda x: str(x) if x is not None else "—",
            key=f"engine_{stage_key}",
            disabled=not engines,
        )

    # Limit the cycle slider to this engine's actual trajectory length
    # (test engines are truncated; global max is only for initial load).
    engine_max_cycle = max_cycle_overall
    if test_features_df is not None and engine_id is not None:
        eng_cycles = test_features_df.loc[
            test_features_df["engine_id"] == engine_id, "cycle"
        ]
        if not eng_cycles.empty:
            engine_max_cycle = int(eng_cycles.max())

    cycle_key = f"cycle_{stage_key}"
    # Clamp stored value if the user switched to a shorter engine
    if cycle_key in st.session_state:
        try:
            st.session_state[cycle_key] = min(
                int(st.session_state[cycle_key]), max(1, engine_max_cycle)
            )
        except (TypeError, ValueError):
            st.session_state[cycle_key] = 1

    with col2:
        cycle = st.slider(
            "Cycle",
            min_value=1,
            max_value=max(1, engine_max_cycle),
            value=1,
            step=1,
            key=cycle_key,
            disabled=not engines,
        )

    with col3:
        risk_horizon = st.selectbox(
            "Risk horizon (cycles)",
            options=available_horizons,
            index=available_horizons.index(10) if 10 in available_horizons else 0,
            key=f"horizon_{stage_key}",
            disabled=not engines,
            help="Horizon used by the ACTION policy (recommend). All horizons still appear in the Failure risk card.",
        )

    if system is not None and test_features_df is not None and engine_id is not None:
        (
            rul_card,
            risk_card,
            anomaly_card,
            action_card,
            timeline_df,
            _,
            _,
        ) = score_engine_at_cycle(
            system,
            test_features_df,
            engine_id,
            cycle,
            risk_horizon=int(risk_horizon),
            horizons=tuple(available_horizons),
        )

        if timeline_df is not None and not timeline_df.empty and timeline_df["value"].notna().any():
            plot_df = timeline_df.copy()
            plot_df["cycle"] = pd.to_numeric(plot_df["cycle"], errors="coerce")
            plot_df["value"] = pd.to_numeric(plot_df["value"], errors="coerce")
            plot_df = plot_df.dropna(subset=["cycle", "value"]).sort_values("cycle")

            sensor_name = (
                str(plot_df["sensor"].iloc[0])
                if "sensor" in plot_df.columns and plot_df["sensor"].notna().any()
                else "sensor"
            )

            line = (
                alt.Chart(plot_df)
                .mark_line(color="#4C9BE8", strokeWidth=2)
                .encode(
                    x=alt.X("cycle:Q", title="Cycle"),
                    y=alt.Y("value:Q", title=sensor_name, scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip("cycle:Q", title="Cycle"),
                        alt.Tooltip("value:Q", title=sensor_name, format=".3f"),
                    ],
                )
            )
            # Vertical rule + red marker at the currently selected cycle
            rule = (
                alt.Chart(pd.DataFrame({"cycle": [cycle]}))
                .mark_rule(color="#FF4B4B", strokeDash=[4, 4], strokeWidth=1.5)
                .encode(x="cycle:Q")
            )
            current = plot_df[plot_df["cycle"] == cycle]
            layers = [line, rule]
            if not current.empty:
                point = (
                    alt.Chart(current)
                    .mark_circle(size=100, color="#FF4B4B", opacity=0.9)
                    .encode(
                        x="cycle:Q",
                        y="value:Q",
                        tooltip=[
                            alt.Tooltip("cycle:Q", title="Cycle"),
                            alt.Tooltip("value:Q", title=sensor_name, format=".3f"),
                        ],
                    )
                )
                layers.append(point)

            chart = alt.layer(*layers).properties(height=300).interactive()
            st.altair_chart(chart, use_container_width=True)
            st.caption(f"Engine timeline — {sensor_name}")
        else:
            st.info("No timeline sensor data available for this engine.")

        # No fixed key on these widgets: a fixed key would pin the first
        # value in session_state and ignore later value= updates when the
        # user changes engine/cycle (classic Streamlit gotcha).
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**RUL card**")
            st.code(rul_card or "", language=None)
        with c2:
            st.markdown("**Failure risk card**")
            st.code(risk_card or "", language=None)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Anomaly card**")
            st.code(anomaly_card or "", language=None)
        with c4:
            st.markdown("**Action**")
            st.code(action_card or "", language=None)
    else:
        st.info("Upload and load this stage's files, then select an engine and cycle.")

    st.markdown(st.session_state[f"meta_text_{stage_key}"])


def main():
    st.set_page_config(
        page_title="Jet Engine Hospital",
        page_icon="✈️",
        layout="wide",
    )
    st.markdown(
        "# Jet Engine Early Warning System\n"
        "Pick a stage tab below, upload that stage's exported files, then select an engine "
        "and cycle to see the full evidence chain."
    )

    tab_labels = [s["label"] for s in STAGES]
    tabs = st.tabs(tab_labels)
    for tab, stage in zip(tabs, STAGES):
        with tab:
            # Use dataset_name as a stable key for session state isolation
            render_stage_tab(stage["label"], stage["dataset_name"], stage["dataset_name"])


if __name__ == "__main__":
    main()
