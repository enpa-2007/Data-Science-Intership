import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_data

def show():
    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem;'>
        <div style='font-size:3.5rem; font-weight:900; letter-spacing:-2px;
                    background: linear-gradient(135deg, #00D4FF, #7B2FFF, #FF6B35);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            SalesPulse AI
        </div>
        <div style='font-size:1.1rem; color:#94A3B8; margin-top:0.5rem; letter-spacing:3px; text-transform:uppercase;'>
            Advertising Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(123,47,255,0.08));
                border: 1px solid rgba(0,212,255,0.2); border-radius:16px; padding:1.5rem; margin-bottom:1.5rem;'>
        <h3 style='color:#00D4FF; margin:0 0 0.5rem;'>📊 Platform Overview</h3>
        <p style='color:#CBD5E1; line-height:1.7; margin:0;'>
        SalesPulse AI uses machine learning to predict sales from TV, Radio, and Newspaper advertising budgets.
        Train multiple models, compare performance, and generate actionable business insights — all in one place.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    total = len(df)
    avg_sales = df["Sales"].mean()
    max_sales = df["Sales"].max()
    best_ch = "TV" if df["TV"].corr(df["Sales"]) > df["Radio"].corr(df["Sales"]) else "Radio"

    cols = st.columns(4)
    metrics = [
        ("🗂️ Records", f"{total:,}", "Dataset size"),
        ("📈 Avg Sales", f"{avg_sales:.1f}K", "Mean units sold"),
        ("🏆 Peak Sales", f"{max_sales:.1f}K", "Maximum achieved"),
        ("⭐ Top Channel", best_ch, "Highest correlation"),
    ]
    for col, (label, val, sub) in zip(cols, metrics):
        col.markdown(f"""
        <div style='background:linear-gradient(135deg,#111827,#1E293B);
                    border:1px solid rgba(0,212,255,0.15); border-radius:12px;
                    padding:1.2rem; text-align:center;'>
            <div style='font-size:1.6rem; font-weight:800; color:#00D4FF;'>{val}</div>
            <div style='font-size:0.75rem; color:#94A3B8; text-transform:uppercase; letter-spacing:1px;'>{label}</div>
            <div style='font-size:0.7rem; color:#64748B; margin-top:0.2rem;'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style='background:#111827; border:1px solid rgba(123,47,255,0.2);
                    border-radius:12px; padding:1.2rem;'>
            <h4 style='color:#7B2FFF; margin:0 0 0.8rem;'>💼 Business Use Case</h4>
            <ul style='color:#CBD5E1; line-height:2; padding-left:1.2rem; margin:0;'>
                <li>Optimize advertising spend allocation</li>
                <li>Forecast revenue from campaigns</li>
                <li>Compare channel effectiveness</li>
                <li>Reduce guesswork, maximize ROI</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='background:#111827; border:1px solid rgba(255,107,53,0.2);
                    border-radius:12px; padding:1.2rem;'>
            <h4 style='color:#FF6B35; margin:0 0 0.8rem;'>🚀 Features</h4>
            <ul style='color:#CBD5E1; line-height:2; padding-left:1.2rem; margin:0;'>
                <li>4 ML models with auto-selection</li>
                <li>Interactive visual analytics</li>
                <li>Budget optimization simulator</li>
                <li>Export models & predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#111827; border:1px solid rgba(0,255,159,0.15);
                border-radius:12px; padding:1rem 1.5rem;'>
        <h4 style='color:#00FF9F; margin:0 0 0.6rem;'>🗺️ Navigation Guide</h4>
    </div>
    """, unsafe_allow_html=True)

    nav_cols = st.columns(3)
    nav_items = [
        ("🔍 Data Exploration", "Visualize correlations, distributions & dataset stats"),
        ("🤖 Model Training", "Train & compare 4 ML algorithms"),
        ("🎯 Prediction", "Input budgets, get instant sales forecast"),
        ("💡 Insights", "Feature importance & optimization tips"),
        ("📊 Visual Analytics", "Interactive Plotly dashboards"),
        ("⚙️ Export", "Download model (.pkl) & predictions (.csv)"),
    ]
    for i, col in enumerate(nav_cols):
        if i < len(nav_items):
            title, desc = nav_items[i]
            col.markdown(f"""
            <div style='background:linear-gradient(135deg,#0F172A,#1E293B);
                        border:1px solid rgba(0,212,255,0.1); border-radius:10px;
                        padding:1rem; height:100%;'>
                <div style='font-size:1rem; font-weight:700; color:#E2E8F0; margin-bottom:0.4rem;'>{title}</div>
                <div style='font-size:0.78rem; color:#64748B;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    nav_cols2 = st.columns(3)
    for i, col in enumerate(nav_cols2):
        idx = i + 3
        if idx < len(nav_items):
            title, desc = nav_items[idx]
            col.markdown(f"""
            <div style='background:linear-gradient(135deg,#0F172A,#1E293B);
                        border:1px solid rgba(0,212,255,0.1); border-radius:10px;
                        padding:1rem; height:100%; margin-top:0.5rem;'>
                <div style='font-size:1rem; font-weight:700; color:#E2E8F0; margin-bottom:0.4rem;'>{title}</div>
                <div style='font-size:0.78rem; color:#64748B;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
