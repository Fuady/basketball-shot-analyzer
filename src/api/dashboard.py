"""
dashboard.py
────────────
Streamlit dashboard for the Basketball Shot Analyzer.

Run:
    streamlit run src/api/dashboard.py
"""
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from pathlib import Path

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Basketball Shot Analyzer",
    page_icon="🏀",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏀 Shot Analyzer")
    st.markdown("Upload a basketball shot clip to analyze mechanics and predict outcome.")
    api_url = st.text_input("API URL", value=API_URL)
    st.divider()
    try:
        r = requests.get(f"{api_url}/health", timeout=2)
        h = r.json()
        icon = "🟢" if h["status"] == "ok" else "🟡"
        st.markdown(f"{icon} **API:** {h['status'].upper()}")
        st.markdown(f"{'✅' if h['bilstm_loaded'] else '❌'} BiLSTM Predictor")
        st.markdown(f"{'✅' if h['form_model_loaded'] else '❌'} Form Scorer")
    except Exception:
        st.warning("API not reachable.\nStart: `uvicorn src.api.main:app`")
    st.divider()
    st.markdown("**Ideal Benchmarks (Free Throw)**")
    for k, v in {
        "Elbow angle at release": "80–105°",
        "Knee bend (pre-release)": "120–155°",
        "Wrist height": "> 0.8 torso",
        "Shoulder tilt": "< 8°",
        "Follow-through elbow": "> 150°",
    }.items():
        st.markdown(f"- **{k}**: {v}")

# ── Main content ──────────────────────────────────────────────────────────────
st.title("🏀 Basketball Shot Prediction & Form Analyzer")
st.markdown("*Pose estimation → BiLSTM outcome prediction → biomechanical feedback*")

tab1, tab2, tab3 = st.tabs(["🎥 Analyze Shot", "📊 Feature Explorer", "ℹ️ About"])

# ── Tab 1: Video Analysis ─────────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader(
        "Upload a shot video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Best results: single shot, full body visible, 30fps+",
    )

    if uploaded:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("📹 Video Preview")
            st.video(uploaded)
            size_mb = len(uploaded.getvalue()) / 1024 / 1024
            st.caption(f"File: {uploaded.name} | Size: {size_mb:.1f} MB")

        with c2:
            st.subheader("🔬 Analysis")
            if st.button("▶ Analyze Shot", type="primary", use_container_width=True):
                with st.spinner("Extracting pose → features → predicting..."):
                    try:
                        resp = requests.post(
                            f"{api_url}/analyze",
                            files={"video": (uploaded.name, uploaded.getvalue(), "video/mp4")},
                            timeout=180,
                        )
                        if resp.status_code == 200:
                            st.session_state["result"] = resp.json()
                        else:
                            st.error(f"API error {resp.status_code}: {resp.text[:300]}")
                    except requests.ConnectionError:
                        st.error("Cannot connect to API. Is it running?")
                    except Exception as exc:
                        st.error(f"Error: {exc}")

        # Show results
        if "result" in st.session_state:
            result = st.session_state["result"]
            st.divider()

            pred = result.get("shot_prediction", {})
            form = result.get("form_analysis", {})
            make_prob = pred.get("make_probability")
            form_score = form.get("form_score")

            # Metric row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Frames Processed", result.get("total_frames", "—"))
            m2.metric("Release Frame", result.get("release_frame", "—"))
            if make_prob is not None:
                pred_label = "🟢 MAKE" if pred["prediction"] == "make" else "🔴 MISS"
                m3.metric("Prediction", pred_label, f"{make_prob*100:.1f}% confidence")
            if form_score is not None:
                color = "normal" if form_score >= 70 else "inverse"
                m4.metric("Form Score", f"{form_score:.0f}/100")

            st.subheader("Shot Outcome Probability")
            if make_prob is not None:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=make_prob * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Make Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2ecc71" if make_prob >= 0.5 else "#e74c3c"},
                        "steps": [
                            {"range": [0, 40],  "color": "#fde9e9"},
                            {"range": [40, 60], "color": "#fef9e7"},
                            {"range": [60, 100],"color": "#e9f7ef"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Attention weights
            attn = pred.get("attention_weights")
            if attn:
                st.subheader("Attention Weights (which frames matter most)")
                fig_attn = px.line(
                    x=list(range(len(attn))), y=attn,
                    labels={"x": "Frame (relative to release)", "y": "Attention Weight"},
                    color_discrete_sequence=["#4C72B0"],
                )
                fig_attn.update_layout(height=200)
                st.plotly_chart(fig_attn, use_container_width=True)

            # Biomechanical readings
            st.subheader("Biomechanical Readings at Release")
            rel_feats = form.get("release_features", {})
            if rel_feats:
                angle_items = [(k, v) for k, v in rel_feats.items()
                               if "angle" in k or "height" in k or "lean" in k or "tilt" in k
                               and isinstance(v, (int, float))]
                if angle_items:
                    feat_df = pd.DataFrame(angle_items, columns=["Feature", "Value"])
                    feat_df["Feature"] = feat_df["Feature"].str.replace("_", " ").str.title()
                    st.dataframe(feat_df, use_container_width=True, hide_index=True)

            # Feedback items
            feedback = form.get("feedback", [])
            if feedback:
                st.subheader(f"💡 Corrective Feedback ({len(feedback)} item(s))")
                for item in feedback:
                    severity = item.get("severity", "low")
                    icon = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                    st.markdown(f"{icon} {item['message']}")
            else:
                st.success("✅ No major form issues detected!")

# ── Tab 2: Feature Explorer ───────────────────────────────────────────────────
with tab2:
    st.subheader("📊 Release Feature Explorer")
    st.markdown("Upload `release_features.csv` to explore make/miss differences.")

    feat_file = st.file_uploader("Upload release_features.csv", type=["csv"], key="feats")
    if feat_file:
        df = pd.read_csv(feat_file)
        if "outcome" in df.columns:
            df["Outcome"] = df["outcome"].map({1: "Make", 0: "Miss"})
            st.markdown(f"**{len(df)} shots** | Makes: {df['outcome'].sum()} | Misses: {(1-df['outcome']).sum()}")

            numeric = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c not in ["outcome", "release_frame"]]
            feat_x = st.selectbox("X feature", numeric, index=0)
            feat_y = st.selectbox("Y feature", numeric, index=min(1, len(numeric)-1))

            fig = px.scatter(
                df, x=feat_x, y=feat_y, color="Outcome",
                color_discrete_map={"Make": "#2ecc71", "Miss": "#e74c3c"},
                title=f"{feat_x} vs {feat_y}",
                opacity=0.7,
            )
            st.plotly_chart(fig, use_container_width=True)

            feat_box = st.selectbox("Feature for boxplot", numeric)
            fig2 = px.box(df, x="Outcome", y=feat_box, color="Outcome",
                          color_discrete_map={"Make": "#2ecc71", "Miss": "#e74c3c"},
                          title=f"{feat_box} distribution")
            st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: About ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("About This Project")
    st.markdown("""
    **Basketball Shot Prediction & Release Point Analyzer**

    | Stage | Description |
    |---|---|
    | Data Engineering | yt-dlp video download, MediaPipe pose extraction, normalization |
    | Analytics | Joint angle computation, wrist speed profiles, release detection |
    | Modeling | BiLSTM (shot prediction) + XGBoost (form scoring) |
    | API | FastAPI REST endpoint with Pydantic validation |
    | Dashboard | This Streamlit app |
    | MLOps | MLflow tracking, Prometheus metrics, Docker Compose |

    **GitHub**: [github.com/yourusername/bball-shot-analyzer](https://github.com)
    """)
