import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import requests
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="SensorLens", layout="wide")


def _api_error(e: Exception) -> str:
    """Extract a readable message from a requests exception.

    For HTTPError responses from FastAPI, returns the 'detail' field from the
    JSON body rather than the generic HTTP status line.
    """
    if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
        try:
            return e.response.json().get("detail", str(e))
        except Exception:
            return str(e)
    return str(e)


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
FEATURE_COLUMNS = [
    "air_temp",
    "process_temp",
    "rotational_speed",
    "torque",
    "tool_wear",
]
FEATURE_LABELS = {
    "air_temp": "Air Temp [K]",
    "process_temp": "Process Temp [K]",
    "rotational_speed": "Rotational Speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool Wear [min]",
}


# --- Load dataset on startup ---
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Fetch the first 100 dataset rows from the backend.

    Returns:
        DataFrame with the preview rows. Retries once after 30s on failure.
    """
    try:
        resp = requests.get(f"{BACKEND_URL}/dataset", timeout=10)
        resp.raise_for_status()
        return pd.DataFrame(resp.json())
    except Exception:
        st.warning("Backend is waking up, please wait 30 seconds...")
        time.sleep(30)
        try:
            resp = requests.get(f"{BACKEND_URL}/dataset", timeout=10)
            resp.raise_for_status()
            return pd.DataFrame(resp.json())
        except Exception as e:
            st.error(f"Backend unavailable after retry: {_api_error(e)}")
            st.stop()


dataset_df = load_dataset()


@st.cache_data(show_spinner=False)
def load_dataset_stats() -> dict:
    """Fetch full-dataset statistics (means, failure rate) from the backend.

    Returns:
        Dict with total_rows, feature_means, and failure_rate.
    """
    resp = requests.get(f"{BACKEND_URL}/dataset/stats", timeout=10)
    resp.raise_for_status()
    return resp.json()


try:
    dataset_stats = load_dataset_stats()
except Exception as e:
    st.error(f"Failed to load dataset statistics: {_api_error(e)}")
    st.stop()


@st.cache_data(show_spinner=False)
def load_full_dataset() -> pd.DataFrame:
    """Fetch all 10,000 rows from the backend for visualizations.

    Returns:
        DataFrame with the complete dataset.
    """
    resp = requests.get(f"{BACKEND_URL}/dataset/full", timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


try:
    full_df = load_full_dataset()
except Exception as e:
    st.error(f"Failed to load full dataset: {_api_error(e)}")
    st.stop()

# --- Session state defaults ---
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "explanations" not in st.session_state:
    st.session_state["explanations"] = None
if "query_history" not in st.session_state:
    st.session_state["query_history"] = []

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Dataset", "Controls & Analysis", "Visualizations", "Anomaly Results", "Query Data"]
)

# =========================================================================
# TAB 1 — Dataset
# =========================================================================
with tab1:
    st.title("SensorLens")
    st.caption(
        "Manufacturing Anomaly Detection & LLM Explanation — GKN Aerospace Portfolio Project"
    )
    st.dataframe(dataset_df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", f"{dataset_stats['total_rows']:,}")
    c2.metric("Features Available", len(FEATURE_COLUMNS))
    c3.metric("Failure Rate", f"{dataset_stats['failure_rate']:.1f}%")

# =========================================================================
# TAB 2 — Controls & Analysis
# =========================================================================
with tab2:
    st.subheader("Analysis Controls")

    features = st.multiselect(
        "Select Features",
        options=FEATURE_COLUMNS,
        default=FEATURE_COLUMNS,
    )
    contamination = st.slider(
        "Contamination Rate",
        0.01,
        0.20,
        0.05,
        0.01,
        help="Expected proportion of anomalies",
    )
    n_estimators = st.number_input(
        "Number of Estimators (Trees)", 50, 300, 100, 10
    )
    max_samples = st.selectbox("Max Samples", ["auto", 256, 512, 1024])

    if st.button("Run Analysis", type="primary"):
        if len(features) < 2:
            st.error("Select at least 2 features")
        else:
            # Clear stale explanations so a failed explain doesn't leave
            # explanations from a previous run alongside new anomaly data.
            st.session_state["explanations"] = None
            try:
                with st.spinner("Running Isolation Forest..."):
                    analyze_resp = requests.post(
                        f"{BACKEND_URL}/analyze",
                        json={
                            "features": features,
                            "contamination": contamination,
                            "n_estimators": n_estimators,
                            "max_samples": max_samples,
                        },
                        timeout=60,
                    )
                    analyze_resp.raise_for_status()
                    result = analyze_resp.json()
                    st.session_state["analysis_result"] = result
            except Exception as e:
                st.error(f"Analysis failed: {_api_error(e)}")
                st.stop()

            anomaly_count = result["anomaly_count"]
            pct = anomaly_count / result["total_rows"] * 100
            st.success(
                f"Analysis complete. {anomaly_count} anomalies detected "
                f"({pct:.1f}% of 10,000 rows)."
            )

            try:
                with st.spinner("Generating LLM explanations..."):
                    explain_resp = requests.post(
                        f"{BACKEND_URL}/explain",
                        json={"anomalies": result["anomalies"]},
                        timeout=300,
                    )
                    explain_resp.raise_for_status()
                    st.session_state["explanations"] = explain_resp.json()["explanations"]
            except Exception as e:
                st.warning(f"LLM explanations unavailable: {_api_error(e)}")

# =========================================================================
# TAB 3 — Visualizations
# =========================================================================
with tab3:
    if st.session_state.get("analysis_result") is None:
        st.info("Run analysis in the Controls tab to see visualizations.")
    else:
        result = st.session_state["analysis_result"]
        all_scores = result["all_scores"]
        anomaly_ids = {a["row_id"] for a in result["anomalies"]}

        is_anomaly = [i in anomaly_ids for i in range(len(all_scores))]
        normal_idx = [i for i, a in enumerate(is_anomaly) if not a]
        anomaly_idx = [i for i, a in enumerate(is_anomaly) if a]
        normal_scores = [all_scores[i] for i in normal_idx]
        anomaly_scores = [all_scores[i] for i in anomaly_idx]

        col1, col2 = st.columns(2)

        # Chart 1 — Anomaly Score Scatter
        with col1:
            threshold = min(normal_scores) if normal_scores else 0
            fig1 = go.Figure()
            fig1.add_trace(
                go.Scatter(
                    x=normal_idx,
                    y=normal_scores,
                    mode="markers",
                    marker=dict(color="grey", size=3, opacity=0.5),
                    name="Normal",
                )
            )
            fig1.add_trace(
                go.Scatter(
                    x=anomaly_idx,
                    y=anomaly_scores,
                    mode="markers",
                    marker=dict(color="red", size=5, opacity=0.8),
                    name="Anomaly",
                )
            )
            fig1.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="yellow",
                annotation_text="Threshold",
            )
            fig1.update_layout(
                title="Anomaly Scores — All 10,000 Rows",
                xaxis_title="Row Index",
                yaxis_title="Anomaly Score",
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig1, use_container_width=True)

        # Chart 2 — Feature Distributions
        with col2:
            selected_features = result.get("features", FEATURE_COLUMNS)
            n_feats = len(selected_features)
            rows_sp = (n_feats + 1) // 2
            fig2 = make_subplots(
                rows=rows_sp,
                cols=2,
                subplot_titles=[FEATURE_LABELS.get(f, f) for f in selected_features],
            )

            anomalies_df = pd.DataFrame(result["anomalies"])
            for i, feat in enumerate(selected_features):
                r = i // 2 + 1
                c = i % 2 + 1
                if feat in full_df.columns:
                    fig2.add_trace(
                        go.Histogram(
                            x=full_df.loc[
                                ~full_df.index.isin(anomaly_ids), feat
                            ],
                            name="Normal",
                            marker_color="blue",
                            opacity=0.6,
                            showlegend=(i == 0),
                        ),
                        row=r,
                        col=c,
                    )
                if feat in anomalies_df.columns:
                    fig2.add_trace(
                        go.Histogram(
                            x=anomalies_df[feat],
                            name="Anomaly",
                            marker_color="red",
                            opacity=0.6,
                            showlegend=(i == 0),
                        ),
                        row=r,
                        col=c,
                    )

            fig2.update_layout(
                title="Feature Distributions: Normal vs Anomaly",
                barmode="overlay",
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        # Chart 3 — Correlation Heatmap
        with col3:
            st.markdown("**Feature Correlation (Anomalies Only)**")
            selected_features = result.get("features", FEATURE_COLUMNS)
            corr_cols = [f for f in selected_features if f in anomalies_df.columns]
            if len(corr_cols) >= 2:
                corr = anomalies_df[corr_cols].corr()
                fig3, ax = plt.subplots(figsize=(6, 5))
                fig3.patch.set_facecolor("#0D1117")
                ax.set_facecolor("#0D1117")
                sns.heatmap(
                    corr,
                    annot=True,
                    cmap="coolwarm",
                    vmin=-1,
                    vmax=1,
                    fmt=".2f",
                    ax=ax,
                    xticklabels=[FEATURE_LABELS.get(c, c) for c in corr_cols],
                    yticklabels=[FEATURE_LABELS.get(c, c) for c in corr_cols],
                )
                ax.tick_params(colors="#E2E8F0")
                for text in ax.texts:
                    text.set_color("#E2E8F0")
                plt.tight_layout()
                st.pyplot(fig3)
            else:
                st.warning("Need at least 2 features for correlation heatmap.")

        # Chart 4 — Time Series (Torque)
        with col4:
            fig4 = go.Figure()
            if "torque" in full_df.columns:
                normal_torque_idx = [i for i in range(len(full_df)) if i not in anomaly_ids]
                fig4.add_trace(
                    go.Scatter(
                        x=normal_torque_idx,
                        y=full_df.loc[normal_torque_idx, "torque"],
                        mode="lines",
                        line=dict(color="grey", width=1),
                        opacity=0.5,
                        name="Normal",
                    )
                )
                if "torque" in anomalies_df.columns:
                    fig4.add_trace(
                        go.Scatter(
                            x=anomalies_df["row_id"].tolist(),
                            y=anomalies_df["torque"].tolist(),
                            mode="markers",
                            marker=dict(color="red", size=6),
                            name="Anomaly",
                        )
                    )
            fig4.update_layout(
                title="Torque Over Time — Anomalies Highlighted",
                xaxis_title="Row Index",
                yaxis_title="Torque [Nm]",
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig4, use_container_width=True)

# =========================================================================
# TAB 4 — Anomaly Results
# =========================================================================
with tab4:
    if st.session_state.get("analysis_result") is None:
        st.info("Run analysis first.")
    else:
        result = st.session_state["analysis_result"]
        anomalies = result["anomalies"]
        anomaly_count = result["anomaly_count"]
        pct = anomaly_count / result["total_rows"] * 100

        st.metric("Anomalies Detected", anomaly_count, f"{pct:.1f}% of total")

        anomalies_df = pd.DataFrame(anomalies)
        st.download_button(
            "Download Anomalies CSV",
            data=anomalies_df.to_csv(index=False),
            file_name="sensorlens_anomalies.csv",
            mime="text/csv",
        )

        # Ground Truth Comparison
        st.subheader("Ground Truth Comparison")
        gt_df = pd.DataFrame(
            {
                "row_id": [a["row_id"] for a in anomalies],
                "model_flagged": [True] * len(anomalies),
                "actual_machine_failure": [a["ground_truth_failure"] for a in anomalies],
                "failure_types": [a["failure_types"] for a in anomalies],
            }
        )
        st.dataframe(gt_df, use_container_width=True)

        # Anomaly Details & LLM Explanations
        st.subheader("Anomaly Details & LLM Explanations")

        explanations_list = st.session_state.get("explanations") or []
        explanations_lookup = {e["row_id"]: e["explanation"] for e in explanations_list}

        # Use full-dataset means from /dataset/stats
        dataset_means = dataset_stats["feature_means"]

        for row in anomalies:
            with st.expander(f"Row {row['row_id']} — Score: {row['anomaly_score']:.3f}"):
                left, right = st.columns([1, 2])
                with left:
                    for feat in FEATURE_COLUMNS:
                        val = row.get(feat, 0)
                        mean_val = dataset_means.get(feat, 0)
                        delta = val - mean_val
                        st.metric(
                            label=FEATURE_LABELS.get(feat, feat),
                            value=f"{val:.1f}",
                            delta=f"{delta:.2f} vs mean",
                        )
                with right:
                    explanation = explanations_lookup.get(
                        row["row_id"], "Explanation unavailable"
                    )
                    st.info(explanation)

                if row["failure_types"] == "None":
                    st.success("No failure recorded")
                else:
                    st.error(f"Failure: {row['failure_types']}")

# =========================================================================
# TAB 5 — Query Data
# =========================================================================
with tab5:
    st.subheader("Ask Questions About Your Anomalies")

    if st.session_state.get("analysis_result") is None:
        st.warning("Run analysis in the Controls tab first to enable querying.")
        st.stop()

    question = st.text_input(
        "Your question",
        placeholder="e.g. Which anomalies have the highest torque? What is the average tool wear among anomalies?",
    )

    if st.button("Ask"):
        if not question:
            st.warning("Enter a question first")
        else:
            try:
                with st.spinner("Thinking..."):
                    resp = requests.post(
                        f"{BACKEND_URL}/query",
                        json={"question": question, "context_rows": 20},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    answer = resp.json()["answer"]

                st.markdown(answer)
                st.session_state["query_history"].append(
                    {"question": question, "answer": answer}
                )
            except Exception as e:
                st.error(f"Query failed: {_api_error(e)}")

    if st.session_state["query_history"]:
        with st.expander("Query History (last 5)"):
            for item in reversed(st.session_state["query_history"][-5:]):
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer']}")
                st.divider()
#   T e m p o r a r y   c h a n g e  
 #   T e m p o r a r y   c h a n g e  
 #   T e m p o r a r y   c h a n g e  
 #   T e m p o r a r y   c h a n g e  
 #   T e m p o r a r y   c h a n g e  
 #   T e m p o r a r y   c h a n g e  
 #   T e m p o r a r y   c h a n g e  
 #   T e m p o r a r y   c h a n g e  
 #   T e m p o r a r y   c h a n g e  
 