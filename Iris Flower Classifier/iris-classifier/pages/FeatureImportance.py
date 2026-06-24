"""
pages/FeatureImportance.py  –  Feature Importance Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train_model import get_trained_models, load_data


def _load():
    if "model_data" not in st.session_state:
        st.session_state["model_data"] = get_trained_models()
    return st.session_state["model_data"]


def show():
    st.markdown("""
    <div class="iris-header">
        <h1>⭐ Feature Importance</h1>
        <p>Discover which flower measurements influence predictions most — across all classifiers.</p>
    </div>
    """, unsafe_allow_html=True)

    data     = _load()
    df       = load_data()
    models   = data["models"]
    scaler   = data["scaler"]
    le       = data["label_encoder"]
    best     = data["best_model"]

    feat_names  = data["feature_cols"]
    feat_labels = {
        "sepal_length": "Sepal Length",
        "sepal_width":  "Sepal Width",
        "petal_length": "Petal Length",
        "petal_width":  "Petal Width",
    }
    labels = [feat_labels[f] for f in feat_names]

    # ── Tree-based models ──────────────────────────────────────────────────
    tree_models = {n: m for n, m in models.items()
                   if hasattr(m, "feature_importances_")}
    linear_models = {n: m for n, m in models.items()
                     if hasattr(m, "coef_")}

    st.markdown('<div class="section-title">🌲 Tree-based Feature Importances</div>', unsafe_allow_html=True)

    if tree_models:
        sel_tree = st.selectbox("Select tree-based model", list(tree_models.keys()),
                                index=0)
        clf = tree_models[sel_tree]
        imps = clf.feature_importances_

        fi_df = pd.DataFrame({"Feature": labels, "Importance": imps}) \
                  .sort_values("Importance", ascending=True)

        c1, c2 = st.columns(2)

        # Horizontal bar
        fig_bar = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=["#e8f3fc", "#1a6bbd", "#1a6b3c"],
            title=f"Feature Importances — {sel_tree}",
            text=fi_df["Importance"].apply(lambda v: f"{v:.4f}")
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               coloraxis_showscale=False, height=320)
        c1.plotly_chart(fig_bar, use_container_width=True)

        # Pie donut
        fig_pie = px.pie(
            fi_df, values="Importance", names="Feature",
            title="Relative Contribution (%)",
            color_discrete_sequence=["#2d9e5f","#1a6bbd","#e67e22","#8b2fc9"],
            hole=0.45
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        c2.plotly_chart(fig_pie, use_container_width=True)

        # All tree models comparison
        st.markdown("#### 🔁 All Tree Models Comparison")
        comp_rows = []
        for name, m in tree_models.items():
            for feat, imp in zip(labels, m.feature_importances_):
                comp_rows.append({"Model": name, "Feature": feat, "Importance": imp})
        comp_df = pd.DataFrame(comp_rows)
        fig_grp = px.bar(
            comp_df, x="Feature", y="Importance", color="Model", barmode="group",
            color_discrete_sequence=["#2d9e5f","#1a6bbd"],
            title="Feature Importance Comparison — Tree Models"
        )
        fig_grp.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_grp, use_container_width=True)

    # ── Linear models ──────────────────────────────────────────────────────
    if linear_models:
        st.markdown('<div class="section-title">📐 Logistic Regression Coefficients</div>', unsafe_allow_html=True)
        for name, clf in linear_models.items():
            coef_df = pd.DataFrame(
                clf.coef_,
                columns=labels,
                index=le.classes_
            ).T.reset_index()
            coef_df.columns = ["Feature"] + list(le.classes_)

            coef_melt = coef_df.melt(id_vars="Feature", var_name="Species", value_name="Coefficient")
            fig_coef = px.bar(
                coef_melt, x="Feature", y="Coefficient", color="Species",
                barmode="group",
                color_discrete_map={"setosa":"#2d9e5f","versicolor":"#1a6bbd","virginica":"#8b2fc9"},
                title=f"{name} — Coefficients per Class"
            )
            fig_coef.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_coef, use_container_width=True)

    # ── Permutation importance (model-agnostic) ────────────────────────────
    st.markdown('<div class="section-title">🎲 Permutation Importance (Model-Agnostic)</div>', unsafe_allow_html=True)
    st.caption("Measures how much the model score decreases when each feature is randomly shuffled.")

    from sklearn.inspection import permutation_importance

    sel_perm = st.selectbox("Select model for permutation analysis",
                             list(models.keys()),
                             index=list(models.keys()).index(best),
                             key="perm_sel")
    clf_p = models[sel_perm]
    X_test_sc = data["X_test_sc"]
    y_test    = data["y_test"]

    with st.spinner("Computing permutation importance…"):
        perm = permutation_importance(clf_p, X_test_sc, y_test, n_repeats=30,
                                      random_state=42, scoring="accuracy")

    perm_df = pd.DataFrame({
        "Feature":    labels,
        "Mean":       perm.importances_mean,
        "Std":        perm.importances_std,
    }).sort_values("Mean", ascending=True)

    fig_perm = go.Figure()
    fig_perm.add_trace(go.Bar(
        x=perm_df["Mean"], y=perm_df["Feature"],
        orientation="h",
        error_x=dict(type="data", array=perm_df["Std"], visible=True),
        marker_color=["#2d9e5f" if v > 0 else "#e74c3c" for v in perm_df["Mean"]],
        text=perm_df["Mean"].apply(lambda v: f"{v:.4f}"),
        textposition="outside"
    ))
    fig_perm.update_layout(
        title=f"Permutation Importance — {sel_perm}",
        xaxis_title="Mean Accuracy Decrease",
        plot_bgcolor="white", paper_bgcolor="white", height=320
    )
    st.plotly_chart(fig_perm, use_container_width=True)

    # ── Key takeaway ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">💡 Key Insights</div>', unsafe_allow_html=True)
    insights = [
        ("🌸 Petal measurements dominate", "Petal Length and Petal Width consistently rank as the most important features across all models. They alone can achieve ~96% accuracy."),
        ("📏 Sepal width least useful", "Sepal Width shows the weakest correlation with species and lowest importance in most models."),
        ("🤖 Setosa is the easiest", "Iris Setosa has distinctly small petals (length < 2cm), making it trivially separable from the other two species."),
        ("⚠️ Versicolor vs Virginica", "The main classification challenge lies between Versicolor and Virginica, which overlap slightly in petal dimensions."),
    ]
    c1, c2 = st.columns(2)
    for idx, (title, body) in enumerate(insights):
        col = c1 if idx % 2 == 0 else c2
        col.markdown(f"""
        <div style='background:white; border-radius:10px; padding:1rem 1.2rem;
                    box-shadow:0 2px 8px rgba(0,0,0,.07); margin-bottom:1rem'>
            <div style='font-weight:700; color:#1a6b3c; margin-bottom:.4rem'>{title}</div>
            <div style='font-size:.88rem; color:#475569'>{body}</div>
        </div>""", unsafe_allow_html=True)
