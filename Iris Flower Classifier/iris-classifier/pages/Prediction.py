"""
pages/Prediction.py  –  Real-time Species Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train_model import get_trained_models

SPECIES_ICONS  = {"setosa": "🌸", "versicolor": "💜", "virginica": "🌺"}
SPECIES_COLORS = {"setosa": "#2d9e5f", "versicolor": "#1a6bbd", "virginica": "#8b2fc9"}


def _load():
    if "model_data" not in st.session_state:
        st.session_state["model_data"] = get_trained_models()
    return st.session_state["model_data"]


def show():
    st.markdown("""
    <div class="iris-header">
        <h1>🔮 Predict Species</h1>
        <p>Enter flower measurements and get an instant AI-powered species prediction with confidence scores.</p>
    </div>
    """, unsafe_allow_html=True)

    data = _load()
    le       = data["label_encoder"]
    scaler   = data["scaler"]
    models   = data["models"]
    best     = data["best_model"]

    # ── Model selector ─────────────────────────────────────────────────────
    selected_model = st.selectbox(
        "🤖 Choose classifier",
        list(models.keys()),
        index=list(models.keys()).index(best),
        help="Select which trained model to use for prediction"
    )
    clf = models[selected_model]

    st.markdown("---")

    # ── Input sliders ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📐 Enter Flower Measurements</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    sepal_length = c1.slider("🌿 Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
    sepal_width  = c1.slider("🌿 Sepal Width (cm)",  2.0, 4.5, 3.4, 0.1)
    petal_length = c2.slider("🌸 Petal Length (cm)", 1.0, 7.0, 4.7, 0.1)
    petal_width  = c2.slider("🌸 Petal Width (cm)",  0.1, 2.5, 1.4, 0.1)

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # ── Predict ────────────────────────────────────────────────────────────
    predict_col, _ = st.columns([1, 3])
    predict_btn = predict_col.button("🔮  Predict Species", use_container_width=True)

    if predict_btn or True:   # real-time prediction
        X_sc = scaler.transform(features)
        pred_idx = clf.predict(X_sc)[0]
        pred_name = le.inverse_transform([pred_idx])[0]
        icon  = SPECIES_ICONS[pred_name]
        color = SPECIES_COLORS[pred_name]

        probas = clf.predict_proba(X_sc)[0] if hasattr(clf, "predict_proba") else None

        # ── Result box ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="prediction-box">
            <div style='font-size:3.5rem'>{icon}</div>
            <div class="species-name">Iris {pred_name.title()}</div>
            <div class="confidence">
                Predicted by <strong>{selected_model}</strong>
                {f'· Confidence: <strong>{round(max(probas)*100,1)}%</strong>' if probas is not None else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability bars ───────────────────────────────────────────────
        if probas is not None:
            st.markdown("#### 📊 Prediction Probabilities")
            for cls, prob in zip(le.classes_, probas):
                pct = round(prob * 100, 1)
                clr = SPECIES_COLORS[cls]
                icn = SPECIES_ICONS[cls]
                c_l, c_bar, c_r = st.columns([2, 6, 1])
                c_l.markdown(f"{icn} **Iris {cls.title()}**")
                c_bar.progress(float(prob))
                c_r.markdown(f"**{pct}%**")

            # Gauge / donut chart
            fig_prob = go.Figure(go.Bar(
                x=le.classes_,
                y=[round(p*100,2) for p in probas],
                marker_color=[SPECIES_COLORS[c] for c in le.classes_],
                text=[f"{round(p*100,1)}%" for p in probas],
                textposition="outside"
            ))
            fig_prob.update_layout(
                title="Probability Distribution",
                yaxis_title="Probability (%)",
                yaxis_range=[0, 110],
                plot_bgcolor="white", paper_bgcolor="white",
                height=300, margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        # ── Save to history ────────────────────────────────────────────────
        entry = {
            "Model":         selected_model,
            "Sepal L":       sepal_length,
            "Sepal W":       sepal_width,
            "Petal L":       petal_length,
            "Petal W":       petal_width,
            "Predicted":     f"{icon} Iris {pred_name.title()}",
            "Confidence (%)": round(max(probas)*100,1) if probas is not None else "N/A",
        }
        if "pred_history" not in st.session_state:
            st.session_state["pred_history"] = []
        # Avoid duplicate consecutive entries
        if not st.session_state["pred_history"] or \
           st.session_state["pred_history"][-1] != entry:
            st.session_state["pred_history"].append(entry)

    # ── Input summary ──────────────────────────────────────────────────────
    with st.expander("📋 Input Summary"):
        st.json({
            "sepal_length": sepal_length,
            "sepal_width":  sepal_width,
            "petal_length": petal_length,
            "petal_width":  petal_width,
        })

    # ── Prediction History ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">🕑 Prediction History</div>', unsafe_allow_html=True)

    if st.session_state.get("pred_history"):
        hist_df = pd.DataFrame(st.session_state["pred_history"])
        st.dataframe(hist_df.tail(20)[::-1], use_container_width=True, hide_index=True)

        csv = hist_df.to_csv(index=False).encode()
        col1, col2, _ = st.columns([2, 2, 4])
        col1.download_button(
            "⬇️ Download History (CSV)",
            csv, "iris_predictions.csv", "text/csv"
        )
        if col2.button("🗑️ Clear History"):
            st.session_state["pred_history"] = []
            st.rerun()
    else:
        st.info("No prediction history yet — adjust the sliders above to begin.")
