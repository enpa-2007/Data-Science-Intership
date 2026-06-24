"""
pages/ModelTraining.py  –  Model Training & Comparison
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train_model import get_trained_models


def _load():
    if "model_data" not in st.session_state:
        st.session_state["model_data"] = get_trained_models()
    return st.session_state["model_data"]


def show():
    st.markdown("""
    <div class="iris-header">
        <h1>🤖 Model Training & Comparison</h1>
        <p>Five classifiers trained, evaluated, and compared — select the best for prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading models…"):
        data = _load()

    results = data["results"]
    best    = data["best_model"]

    # ── Pipeline overview ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">⚙️ Training Pipeline</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    steps = [
        ("1️⃣", "Data Loading",      "150 samples, 4 features"),
        ("2️⃣", "Train/Test Split",  "80% train · 20% test (stratified)"),
        ("3️⃣", "Feature Scaling",   "StandardScaler (zero mean, unit var)"),
        ("4️⃣", "Model Training",    "5 classifiers + 5-fold CV"),
    ]
    for col, (num, title, desc) in zip([p1, p2, p3, p4], steps):
        col.markdown(f"""
        <div style='background:white; border-radius:10px; padding:1rem;
                    box-shadow:0 2px 8px rgba(0,0,0,.07); text-align:center'>
            <div style='font-size:1.8rem'>{num}</div>
            <div style='font-weight:700; color:#1a6b3c; margin:.3rem 0 .2rem'>{title}</div>
            <div style='font-size:.8rem; color:#64748b'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Best model highlight ───────────────────────────────────────────────
    br = results[best]
    st.success(f"🏆 **Best Model: {best}** — Accuracy {br['accuracy']}% | "
               f"F1 {br['f1_score']}% | CV {br['cv_score']}%")

    # ── Comparison table ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Model Comparison</div>', unsafe_allow_html=True)

    rows = []
    for name, r in results.items():
        rows.append({
            "Model":        name,
            "Accuracy":     r["accuracy"],
            "Precision":    r["precision"],
            "Recall":       r["recall"],
            "F1 Score":     r["f1_score"],
            "CV Score":     r["cv_score"],
            "Best":         "🏆" if name == best else "",
        })
    df_cmp = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    st.dataframe(
        df_cmp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy":  st.column_config.ProgressColumn("Accuracy (%)",  min_value=0, max_value=100, format="%.2f%%"),
            "Precision": st.column_config.ProgressColumn("Precision (%)", min_value=0, max_value=100, format="%.2f%%"),
            "Recall":    st.column_config.ProgressColumn("Recall (%)",    min_value=0, max_value=100, format="%.2f%%"),
            "F1 Score":  st.column_config.ProgressColumn("F1 Score (%)",  min_value=0, max_value=100, format="%.2f%%"),
            "CV Score":  st.column_config.ProgressColumn("CV Score (%)",  min_value=0, max_value=100, format="%.2f%%"),
        }
    )

    # ── Bar chart comparison ───────────────────────────────────────────────
    st.markdown('<div class="section-title">📉 Metric Comparison Chart</div>', unsafe_allow_html=True)
    metric = st.selectbox("Compare by", ["Accuracy", "F1 Score", "CV Score", "Precision", "Recall"])
    fig_bar = px.bar(
        df_cmp.sort_values(metric, ascending=True),
        x=metric, y="Model", orientation="h",
        color=metric, color_continuous_scale=["#1a6bbd", "#2d9e5f"],
        title=f"{metric} — All Models",
        text=df_cmp.sort_values(metric, ascending=True)[metric].apply(lambda v: f"{v:.2f}%"),
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          coloraxis_showscale=False, height=350)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Radar chart ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🕸️ Radar Comparison</div>', unsafe_allow_html=True)
    model_names = list(results.keys())
    selected_models = st.multiselect("Select models", model_names, default=model_names)

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "CV Score"]
    fig_radar = go.Figure()
    PALETTE = ["#2d9e5f", "#1a6bbd", "#e67e22", "#8b2fc9", "#e74c3c"]

    for i, name in enumerate(selected_models):
        r = results[name]
        vals = [r["accuracy"], r["precision"], r["recall"], r["f1_score"], r["cv_score"]]
        vals += [vals[0]]  # close polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=metrics + [metrics[0]],
            fill="toself", name=name,
            line_color=PALETTE[i % len(PALETTE)],
            opacity=.7
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[80, 100])),
        showlegend=True, height=450,
        paper_bgcolor="white"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Model descriptions ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">📖 Model Descriptions</div>', unsafe_allow_html=True)
    descs = {
        "Logistic Regression":
            "Linear model that estimates class probabilities via the logistic function. Fast, interpretable, "
            "and performs well when classes are linearly separable.",
        "Decision Tree":
            "Hierarchical rule-based model that splits data recursively by the most informative feature. "
            "Highly interpretable but prone to overfitting.",
        "Random Forest":
            "Ensemble of decision trees trained on random data subsets (bagging). Reduces overfitting "
            "and provides reliable feature importances.",
        "KNN":
            "Classifies a sample by majority vote among its k nearest neighbours in feature space. "
            "Non-parametric and sensitive to feature scale — hence the StandardScaler.",
        "SVM":
            "Finds the maximum-margin hyperplane separating classes in a high-dimensional space (RBF kernel). "
            "Excellent on small-to-medium datasets.",
    }
    for name, desc in descs.items():
        badge = "🏆 " if name == best else ""
        with st.expander(f"{badge}{name}"):
            st.write(desc)
            r = results[name]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",  f"{r['accuracy']}%")
            c2.metric("Precision", f"{r['precision']}%")
            c3.metric("Recall",    f"{r['recall']}%")
            c4.metric("F1 Score",  f"{r['f1_score']}%")
