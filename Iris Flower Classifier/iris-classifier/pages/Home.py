"""
pages/Home.py  –  Dashboard / Landing Page
"""

import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train_model import load_data, get_trained_models


def _load(force=False):
    if "model_data" not in st.session_state or force:
        st.session_state["model_data"] = get_trained_models()
    return st.session_state["model_data"]


def show():
    # ── Gradient header ───────────────────────────────────────────────────
    st.markdown("""
    <div class="iris-header">
        <h1>🌸 Iris Flower Classifier</h1>
        <p>A production-ready Machine Learning application that classifies Iris flowers
        into <strong>Setosa</strong>, <strong>Versicolor</strong>, and <strong>Virginica</strong>
        using sepal and petal measurements. Built with Scikit-Learn · Streamlit · Plotly.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────
    with st.spinner("Loading models…"):
        data = _load()
        df   = load_data()

    # ── Top metrics ───────────────────────────────────────────────────────
    best   = data["best_model"]
    best_r = data["results"][best]

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Total Samples",  "150",        "Balanced dataset"),
        ("Features",       "4",          "Sepal & petal dims"),
        ("Species Classes","3",          "Setosa · Versicolor · Virginica"),
        ("Best Accuracy",  f"{best_r['accuracy']}%", f"{best} model"),
    ]
    for col, (label, value, sub) in zip([c1, c2, c3, c4], cards):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model performance overview ────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Model Performance Overview</div>', unsafe_allow_html=True)

    results = data["results"]
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Accuracy (%)":  r["accuracy"],
            "Precision (%)": r["precision"],
            "Recall (%)":    r["recall"],
            "F1 Score (%)":  r["f1_score"],
            "CV Score (%)":  r["cv_score"],
            "🏆 Best": "✅" if name == best else "",
        })
    df_res = pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False)
    st.dataframe(
        df_res,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy (%)":  st.column_config.ProgressColumn("Accuracy (%)",  min_value=0, max_value=100),
            "F1 Score (%)":  st.column_config.ProgressColumn("F1 Score (%)",  min_value=0, max_value=100),
            "CV Score (%)":  st.column_config.ProgressColumn("CV Score (%)",  min_value=0, max_value=100),
        }
    )

    # ── About / species cards ──────────────────────────────────────────────
    st.markdown('<div class="section-title">🌿 About the Iris Species</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)

    species_info = [
        ("🌸", "Iris Setosa",     "badge-setosa",
         "Easily distinguishable by its very small petal size. "
         "Petal length typically 1–2 cm. Native to Arctic regions."),
        ("💜", "Iris Versicolor", "badge-versicolor",
         "Medium-sized petals and sepals. Commonly found in "
         "North America. Often called the 'Blue Flag' iris."),
        ("🌺", "Iris Virginica",  "badge-virginica",
         "The largest of the three species with the longest petals. "
         "Native to the eastern United States."),
    ]
    for col, (icon, name, badge, desc) in zip([s1, s2, s3], species_info):
        col.markdown(f"""
        <div style='background:white; border-radius:12px; padding:1.4rem;
                    box-shadow:0 4px 16px rgba(0,0,0,.08); height:100%'>
            <div style='font-size:2.5rem; text-align:center; margin-bottom:.8rem'>{icon}</div>
            <div style='text-align:center; margin-bottom:.6rem'>
                <span class='species-badge {badge}'>{name}</span>
            </div>
            <p style='font-size:.88rem; color:#475569; text-align:center; margin:0'>{desc}</p>
        </div>""", unsafe_allow_html=True)

    # ── Feature descriptions ───────────────────────────────────────────────
    st.markdown('<div class="section-title">📐 Feature Descriptions</div>', unsafe_allow_html=True)
    feats = pd.DataFrame({
        "Feature":     ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
        "Unit":        ["cm", "cm", "cm", "cm"],
        "Min":         [df.sepal_length.min(), df.sepal_width.min(),
                        df.petal_length.min(), df.petal_width.min()],
        "Max":         [df.sepal_length.max(), df.sepal_width.max(),
                        df.petal_length.max(), df.petal_width.max()],
        "Mean":        [round(df.sepal_length.mean(),2), round(df.sepal_width.mean(),2),
                        round(df.petal_length.mean(),2), round(df.petal_width.mean(),2)],
        "Description": [
            "Length of the sepal (outermost leaf-like structure)",
            "Width of the sepal",
            "Length of the petal (inner coloured leaf)",
            "Width of the petal",
        ]
    })
    st.dataframe(feats, use_container_width=True, hide_index=True)

    # ── Quick-start tip ────────────────────────────────────────────────────
    st.info("💡 **Quick Start:** Head to **🔮 Predict Species** in the sidebar to classify a flower instantly, "
            "or explore the **📊 Dataset Explorer** to understand the data.")
