"""
pages/Visualization.py  –  Data Visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train_model import load_data

COLORS = {"setosa": "#2d9e5f", "versicolor": "#1a6bbd", "virginica": "#8b2fc9"}


def show():
    st.markdown("""
    <div class="iris-header">
        <h1>📈 Data Visualizations</h1>
        <p>Interactive charts revealing patterns, distributions, and correlations in the Iris dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    feat_labels = {
        "sepal_length": "Sepal Length (cm)",
        "sepal_width":  "Sepal Width (cm)",
        "petal_length": "Petal Length (cm)",
        "petal_width":  "Petal Width (cm)",
    }

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔗 Pair Plot", "📊 Histograms", "🌡️ Correlation Heatmap",
        "📦 Box Plots", "🫧 3D Scatter"
    ])

    # ── Pair Plot ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Feature Relationship Pair Plot")
        st.caption("Each cell shows the relationship between two features, coloured by species.")
        fig = px.scatter_matrix(
            df, dimensions=features, color="species",
            color_discrete_map=COLORS,
            labels={f: feat_labels[f] for f in features},
            title="Pair Plot — All Feature Combinations"
        )
        fig.update_traces(diagonal_visible=True, showupperhalf=False,
                          marker=dict(size=4, opacity=.8))
        fig.update_layout(height=650, plot_bgcolor="white",
                          paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # ── Histograms ────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Feature Distributions by Species")
        sel_feat = st.selectbox(
            "Select feature", features,
            format_func=lambda x: feat_labels[x]
        )
        c1, c2 = st.columns(2)

        # Overlapping histogram
        fig_hist = px.histogram(
            df, x=sel_feat, color="species", barmode="overlay",
            color_discrete_map=COLORS, nbins=25,
            labels={sel_feat: feat_labels[sel_feat]},
            title=f"Distribution of {feat_labels[sel_feat]}"
        )
        fig_hist.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        c1.plotly_chart(fig_hist, use_container_width=True)

        # KDE violin
        fig_vio = px.violin(
            df, y=sel_feat, x="species", color="species", box=True, points="all",
            color_discrete_map=COLORS,
            labels={sel_feat: feat_labels[sel_feat]},
            title=f"Violin Plot — {feat_labels[sel_feat]}"
        )
        fig_vio.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
        c2.plotly_chart(fig_vio, use_container_width=True)

        # All features grid
        st.markdown("#### All Features — Histogram Grid")
        cols = st.columns(2)
        for idx, feat in enumerate(features):
            fig_f = px.histogram(
                df, x=feat, color="species", barmode="overlay",
                color_discrete_map=COLORS, nbins=20,
                labels={feat: feat_labels[feat]},
                title=feat_labels[feat]
            )
            fig_f.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                 showlegend=(idx == 0), height=300, margin=dict(t=40, b=20))
            cols[idx % 2].plotly_chart(fig_f, use_container_width=True)

    # ── Correlation Heatmap ───────────────────────────────────────────────
    with tab3:
        st.markdown("### Feature Correlation Heatmap")
        corr = df[features].corr().round(3)
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values,
            x=[feat_labels[f] for f in features],
            y=[feat_labels[f] for f in features],
            colorscale="RdYlGn",
            zmin=-1, zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 14, "color": "black"},
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>"
        ))
        fig_heat.update_layout(
            title="Pearson Correlation Coefficients",
            height=480, plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("**Key Insight:**")
        st.info("🔗 Petal length and petal width have a very high positive correlation (≈0.96), "
                "making them the most powerful joint predictors for species classification.")

    # ── Box Plots ─────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Box Plot — Feature Spread per Species")
        c1, c2 = st.columns(2)
        for idx, feat in enumerate(features):
            fig_box = px.box(
                df, x="species", y=feat, color="species",
                color_discrete_map=COLORS, points="outliers",
                labels={feat: feat_labels[feat]},
                title=feat_labels[feat]
            )
            fig_box.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                   showlegend=False, height=320, margin=dict(t=40,b=20))
            [c1, c2][idx % 2].plotly_chart(fig_box, use_container_width=True)

    # ── 3D Scatter ────────────────────────────────────────────────────────
    with tab5:
        st.markdown("### 3D Feature Space")
        ax1 = st.selectbox("X axis", features, index=2, format_func=lambda x: feat_labels[x], key="3dx")
        ax2 = st.selectbox("Y axis", features, index=3, format_func=lambda x: feat_labels[x], key="3dy")
        ax3 = st.selectbox("Z axis", features, index=0, format_func=lambda x: feat_labels[x], key="3dz")

        fig_3d = px.scatter_3d(
            df, x=ax1, y=ax2, z=ax3, color="species",
            color_discrete_map=COLORS, opacity=.85, size_max=6,
            labels={ax1: feat_labels[ax1], ax2: feat_labels[ax2], ax3: feat_labels[ax3]},
            title="3D Feature Space — Iris Species"
        )
        fig_3d.update_layout(height=580, scene=dict(
            xaxis=dict(backgroundcolor="white", gridcolor="#e2e8f0"),
            yaxis=dict(backgroundcolor="white", gridcolor="#e2e8f0"),
            zaxis=dict(backgroundcolor="white", gridcolor="#e2e8f0"),
        ))
        st.plotly_chart(fig_3d, use_container_width=True)
