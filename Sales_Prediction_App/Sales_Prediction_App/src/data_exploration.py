import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_data, get_data_info
from utils.visualization import (
    correlation_heatmap, sales_distribution, channel_vs_sales, pairplot_figure
)

def show():
    st.markdown("""
    <h2 style='background: linear-gradient(135deg,#00D4FF,#7B2FFF);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               margin-bottom:1.5rem;'>🔍 Data Exploration</h2>
    """, unsafe_allow_html=True)

    df = load_data()
    info = get_data_info(df)

    tab1, tab2, tab3 = st.tabs(["📋 Dataset Overview", "📊 Statistics", "📈 Visualizations"])

    with tab1:
        st.markdown("#### Dataset Preview")
        st.dataframe(df.head(15), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", info["shape"][0])
        c2.metric("Columns", info["shape"][1])
        c3.metric("Missing Values", sum(info["missing_values"].values()))
        c4.metric("Numeric Cols", len(df.select_dtypes(include="number").columns))

        st.markdown("#### Data Types")
        dtype_df = pd.DataFrame({"Column": list(info["dtypes"].keys()), "Type": list(info["dtypes"].values())})
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        mv = info["missing_values"]
        if sum(mv.values()) == 0:
            st.success("✅ No missing values detected!")
        else:
            st.warning(f"⚠️ Missing values: {mv}")

    with tab2:
        st.markdown("#### Statistical Summary")
        st.dataframe(df.describe().round(3), use_container_width=True)

        st.markdown("#### Correlation Matrix (Raw)")
        st.dataframe(df.corr().round(4), use_container_width=True)

        st.markdown("#### Sales Correlation by Channel")
        corr_cols = st.columns(3)
        channels = ["TV", "Radio", "Newspaper"]
        for col, ch in zip(corr_cols, channels):
            corr_val = df[ch].corr(df["Sales"])
            col.metric(f"{ch} ↔ Sales", f"{corr_val:.4f}")

    with tab3:
        st.plotly_chart(correlation_heatmap(df), use_container_width=True)
        st.plotly_chart(sales_distribution(df), use_container_width=True)

        st.markdown("#### Channel Budget vs Sales")
        cols = st.columns(3)
        for col, ch in zip(cols, ["TV", "Radio", "Newspaper"]):
            with col:
                st.plotly_chart(channel_vs_sales(df, ch), use_container_width=True)

        st.plotly_chart(pairplot_figure(df), use_container_width=True)
