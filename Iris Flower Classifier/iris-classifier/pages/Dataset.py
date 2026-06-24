"""
pages/Dataset.py  –  Dataset Explorer
"""

import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train_model import load_data


def show():
    st.markdown("""
    <div class="iris-header">
        <h1>📊 Dataset Explorer</h1>
        <p>Browse, filter, and analyse the Iris dataset — 150 samples · 4 features · 3 species</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()

    # ── Sidebar filters ────────────────────────────────────────────────────
    with st.expander("🔍 Filter & Search", expanded=True):
        col_sp, col_sl, col_sw = st.columns(3)
        species_filter = col_sp.multiselect(
            "Species", df["species"].unique().tolist(),
            default=df["species"].unique().tolist()
        )
        sl_range = col_sl.slider(
            "Sepal Length (cm)",
            float(df.sepal_length.min()), float(df.sepal_length.max()),
            (float(df.sepal_length.min()), float(df.sepal_length.max()))
        )
        sw_range = col_sw.slider(
            "Sepal Width (cm)",
            float(df.sepal_width.min()), float(df.sepal_width.max()),
            (float(df.sepal_width.min()), float(df.sepal_width.max()))
        )
        col_pl, col_pw, _ = st.columns(3)
        pl_range = col_pl.slider(
            "Petal Length (cm)",
            float(df.petal_length.min()), float(df.petal_length.max()),
            (float(df.petal_length.min()), float(df.petal_length.max()))
        )
        pw_range = col_pw.slider(
            "Petal Width (cm)",
            float(df.petal_width.min()), float(df.petal_width.max()),
            (float(df.petal_width.min()), float(df.petal_width.max()))
        )

    mask = (
        df["species"].isin(species_filter) &
        df["sepal_length"].between(*sl_range) &
        df["sepal_width"].between(*sw_range) &
        df["petal_length"].between(*pl_range) &
        df["petal_width"].between(*pw_range)
    )
    filtered = df[mask].reset_index(drop=True)

    # ── Quick stats ────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows shown",   len(filtered))
    m2.metric("Total rows",   len(df))
    m3.metric("Columns",      len(df.columns))
    m4.metric("Missing vals", int(df.isnull().sum().sum()))

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab_data, tab_stats, tab_dist, tab_missing = st.tabs(
        ["📋 Data Table", "📈 Statistics", "🔵 Class Distribution", "❓ Missing Values"]
    )

    with tab_data:
        st.markdown(f"**{len(filtered)} rows** after filter")
        st.dataframe(
    filtered,
    use_container_width=True, height=420
)
        csv = filtered.to_csv(index=False).encode()
        st.download_button("⬇️ Download filtered CSV", csv, "iris_filtered.csv", "text/csv")

    with tab_stats:
        st.markdown("**Statistical Summary**")
        desc = df.describe().round(4)
        st.dataframe(desc, use_container_width=True)

        st.markdown("**Per-species Statistics**")
        sp_stats = df.groupby("species").agg(
            count=("sepal_length","count"),
            sl_mean=("sepal_length","mean"),
            sw_mean=("sepal_width","mean"),
            pl_mean=("petal_length","mean"),
            pw_mean=("petal_width","mean"),
        ).round(3).reset_index()
        sp_stats.columns = ["Species","Count","Sepal L (mean)","Sepal W (mean)",
                             "Petal L (mean)","Petal W (mean)"]
        st.dataframe(sp_stats, use_container_width=True, hide_index=True)

    with tab_dist:
        import plotly.express as px
        c1, c2 = st.columns(2)
        fig_bar = px.bar(
            df["species"].value_counts().reset_index(),
            x="species", y="count", color="species",
            color_discrete_map={"setosa":"#2d9e5f","versicolor":"#1a6bbd","virginica":"#8b2fc9"},
            title="Sample Count per Species",
            labels={"species":"Species","count":"Count"}
        )
        fig_bar.update_layout(showlegend=False, plot_bgcolor="white")
        c1.plotly_chart(fig_bar, use_container_width=True)

        fig_pie = px.pie(
            df, names="species",
            color="species",
            color_discrete_map={"setosa":"#2d9e5f","versicolor":"#1a6bbd","virginica":"#8b2fc9"},
            title="Species Distribution (%)"
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        c2.plotly_chart(fig_pie, use_container_width=True)

    with tab_missing:
        mv = df.isnull().sum().reset_index()
        mv.columns = ["Feature", "Missing Values"]
        mv["% Missing"] = (mv["Missing Values"] / len(df) * 100).round(2)
        st.dataframe(mv, use_container_width=True, hide_index=True)
        if mv["Missing Values"].sum() == 0:
            st.success("✅ No missing values found! Dataset is complete and clean.")
