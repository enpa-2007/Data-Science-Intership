"""
pages/Evaluation.py  –  Model Evaluation & Metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
        <h1>📋 Model Evaluation</h1>
        <p>Accuracy, Precision, Recall, F1, and Confusion Matrices for all trained classifiers.</p>
    </div>
    """, unsafe_allow_html=True)

    data = _load()
    results = data["results"]
    best    = data["best_model"]
    le      = data["label_encoder"]
    classes = le.classes_

    # ── Model selector ─────────────────────────────────────────────────────
    selected = st.selectbox(
        "Select model to evaluate",
        list(results.keys()),
        index=list(results.keys()).index(best)
    )
    r = results[selected]

    # ── Metric cards ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Performance Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        ("Accuracy",  r["accuracy"],  "%"),
        ("Precision", r["precision"], "%"),
        ("Recall",    r["recall"],    "%"),
        ("F1 Score",  r["f1_score"],  "%"),
        ("CV Score",  r["cv_score"],  "%"),
    ]
    colors = ["#2d9e5f", "#1a6bbd", "#e67e22", "#8b2fc9", "#e74c3c"]
    for col, (label, val, unit), clr in zip([m1,m2,m3,m4,m5], metrics, colors):
        col.markdown(f"""
        <div class="metric-card" style="border-left-color:{clr}">
            <div class="label">{label}</div>
            <div class="value" style="color:{clr}">{val}{unit}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_left, c_right = st.columns([1, 1])

    # ── Confusion Matrix ───────────────────────────────────────────────────
    with c_left:
        st.markdown("#### 🧩 Confusion Matrix")
        cm = r["confusion_matrix"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=list(classes),
            y=list(classes),
            colorscale=[[0,"#ffffff"], [0.5,"#4eca85"], [1.0,"#1a6b3c"]],
            text=cm.astype(str),
            texttemplate="<b>%{text}</b>",
            textfont={"size": 22},
            showscale=True,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_cm.update_layout(
            title=f"Confusion Matrix — {selected}",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=380,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Metric gauge charts ────────────────────────────────────────────────
    with c_right:
        st.markdown("#### 📊 Metric Gauges")
        for (label, val, _), clr in zip(metrics, colors):
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                title={"text": label, "font": {"size": 13}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": clr},
                    "steps": [
                        {"range": [0,  60], "color": "#fee2e2"},
                        {"range": [60, 80], "color": "#fef9c3"},
                        {"range": [80,100], "color": "#dcfce7"},
                    ],
                    "threshold": {"line":{"color":"#0f172a","width":3},
                                  "thickness":.75, "value":95}
                },
                number={"suffix": "%", "font": {"size": 20}},
                domain={"x": [0,1], "y": [0,1]}
            ))
            fig_g.update_layout(height=160, margin=dict(l=10,r=10,t=30,b=10),
                                 paper_bgcolor="white")
            st.plotly_chart(fig_g, use_container_width=True)

    # ── Classification Report ──────────────────────────────────────────────
    st.markdown('<div class="section-title">📄 Classification Report</div>', unsafe_allow_html=True)
    with st.expander("View full classification report", expanded=False):
        st.code(r["classification_report"], language="text")

    # ── All-models confusion matrices ──────────────────────────────────────
    st.markdown('<div class="section-title">🔁 All Models — Confusion Matrices</div>', unsafe_allow_html=True)
    model_names = list(results.keys())
    cols = st.columns(len(model_names))
    for col, name in zip(cols, model_names):
        cm = results[name]["confusion_matrix"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=list(classes), y=list(classes),
            colorscale=[[0,"#ffffff"],[1,"#1a6b3c"]],
            text=cm.astype(str), texttemplate="%{text}",
            showscale=False
        ))
        fig.update_layout(
            title={"text": name, "font": {"size": 11}},
            height=220, margin=dict(l=5,r=5,t=35,b=5),
            xaxis=dict(tickfont={"size":8}), yaxis=dict(tickfont={"size":8}),
            plot_bgcolor="white", paper_bgcolor="white"
        )
        col.plotly_chart(fig, use_container_width=True)
        acc = results[name]["accuracy"]
        col.markdown(f"<div style='text-align:center; font-size:.82rem; font-weight:600; color:#1a6b3c'>{acc}%</div>",
                     unsafe_allow_html=True)

    # ── Download model ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">💾 Export Model</div>', unsafe_allow_html=True)
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "iris_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_bytes = f.read()
        st.download_button(
            "⬇️ Download Trained Models (.pkl)",
            model_bytes, "iris_model.pkl", "application/octet-stream"
        )
        st.caption("The .pkl file contains all 5 trained models, scaler, label encoder, and evaluation results.")
