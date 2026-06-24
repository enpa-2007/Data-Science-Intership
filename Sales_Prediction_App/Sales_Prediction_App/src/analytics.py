import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_data, preprocess_data
from utils.training import train_all_models
from utils.visualization import budget_vs_sales_3d, actual_vs_predicted, residual_plot, PALETTE, PLOTLY_TEMPLATE

def ensure_trained():
    if "best_model" not in st.session_state:
        df = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(df)
        results, trained_models, best_name, best_model = train_all_models(
            X_train, X_test, y_train, y_test
        )
        st.session_state.update({
            "best_model": best_model, "best_name": best_name,
            "results": results, "trained_models": trained_models,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        })

def show():
    st.markdown("""
    <h2 style='background: linear-gradient(135deg,#00D4FF,#7B2FFF);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               margin-bottom:1.5rem;'>📊 Visual Analytics</h2>
    """, unsafe_allow_html=True)

    ensure_trained()
    df = load_data()

    tab1, tab2, tab3, tab4 = st.tabs(["🌐 3D Explorer", "📉 Prediction Analysis", "📡 Budget Simulator", "📈 Trends"])

    with tab1:
        st.plotly_chart(budget_vs_sales_3d(df), use_container_width=True)

        st.markdown("#### Interactive Budget vs Sales")
        channel = st.selectbox("Select Channel", ["TV", "Radio", "Newspaper"])
        fig = px.scatter(df, x=channel, y="Sales", trendline="lowess",
                         color="Sales", color_continuous_scale="Plasma",
                         size="Sales", hover_data=["TV", "Radio", "Newspaper"],
                         template=PLOTLY_TEMPLATE, title=f"{channel} Budget → Sales (LOWESS)")
        fig.update_layout(paper_bgcolor=PALETTE["card"], font_color=PALETTE["text"], height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "results" in st.session_state:
            results = st.session_state["results"]
            y_test = st.session_state["y_test"]
            model_sel = st.selectbox("Select Model", list(results.keys()))
            preds = results[model_sel]["Predictions"]
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(actual_vs_predicted(y_test, preds, model_sel), use_container_width=True)
            with c2:
                st.plotly_chart(residual_plot(y_test, preds, model_sel), use_container_width=True)

            # Prediction error distribution
            errors = np.array(y_test) - np.array(preds)
            fig = go.Figure(go.Histogram(x=errors, nbinsx=20,
                                         marker_color=PALETTE["secondary"], opacity=0.85))
            fig.update_layout(title="Prediction Error Distribution",
                              xaxis_title="Error", yaxis_title="Count",
                              template=PLOTLY_TEMPLATE, paper_bgcolor=PALETTE["card"],
                              font_color=PALETTE["text"], height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train models first on the **Model Training** page.")

    with tab3:
        st.markdown("#### 🎛️ Budget Optimization Simulator")
        st.markdown("<small style='color:#64748B;'>Adjust total budget and allocation to simulate sales outcomes</small>",
                    unsafe_allow_html=True)

        total_budget = st.slider("Total Budget ($K)", 50.0, 500.0, 200.0, 10.0)
        tv_pct = st.slider("TV %", 0, 100, 55)
        radio_pct = st.slider("Radio %", 0, 100 - tv_pct, 30)
        newspaper_pct = 100 - tv_pct - radio_pct

        tv_b = total_budget * tv_pct / 100
        radio_b = total_budget * radio_pct / 100
        newspaper_b = total_budget * newspaper_pct / 100

        X_sim = pd.DataFrame([[tv_b, radio_b, newspaper_b]], columns=["TV", "Radio", "Newspaper"])
        sim_pred = st.session_state["best_model"].predict(X_sim)[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TV", f"${tv_b:.0f}K", f"{tv_pct}%")
        c2.metric("Radio", f"${radio_b:.0f}K", f"{radio_pct}%")
        c3.metric("Newspaper", f"${newspaper_b:.0f}K", f"{newspaper_pct}%")
        c4.metric("Predicted Sales", f"{sim_pred:.2f}K")

        # Sweep TV budget to show optimal allocation
        tv_range = np.linspace(0, total_budget, 50)
        predictions = []
        for tv_val in tv_range:
            remaining = total_budget - tv_val
            r = remaining * 0.6
            n = remaining * 0.4
            Xr = pd.DataFrame([[tv_val, r, n]], columns=["TV", "Radio", "Newspaper"])
            predictions.append(st.session_state["best_model"].predict(Xr)[0])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tv_range, y=predictions, mode="lines+markers",
                                 line=dict(color=PALETTE["primary"], width=3),
                                 marker=dict(size=4)))
        fig.add_vline(x=tv_b, line_dash="dash", line_color=PALETTE["accent"],
                      annotation_text=f"Current: ${tv_b:.0f}K")
        fig.update_layout(title="TV Budget Sweep (Remaining split 60/40 Radio/Newspaper)",
                          xaxis_title="TV Budget ($K)", yaxis_title="Predicted Sales (K)",
                          template=PLOTLY_TEMPLATE, paper_bgcolor=PALETTE["card"],
                          font_color=PALETTE["text"], height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("#### Sales Trend by Budget Quartile")
        df["TV_quartile"] = pd.qcut(df["TV"], 4, labels=["Q1 Low", "Q2 Med-Low", "Q3 Med-High", "Q4 High"])
        grouped = df.groupby("TV_quartile", observed=True)["Sales"].agg(["mean", "std"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=grouped["TV_quartile"].astype(str), y=grouped["mean"],
                             error_y=dict(type="data", array=grouped["std"].fillna(0)),
                             marker_color=[PALETTE["secondary"], PALETTE["primary"],
                                          PALETTE["success"], PALETTE["accent"]]))
        fig.update_layout(title="Mean Sales by TV Budget Quartile",
                          xaxis_title="TV Budget Quartile", yaxis_title="Mean Sales (K)",
                          template=PLOTLY_TEMPLATE, paper_bgcolor=PALETTE["card"],
                          font_color=PALETTE["text"], height=380)
        st.plotly_chart(fig, use_container_width=True)

        # Advertising ROI calculator
        st.markdown("#### 💰 ROI Calculator")
        c1, c2, c3 = st.columns(3)
        with c1:
            roi_tv = st.number_input("TV Budget ($K)", 0.0, 300.0, 100.0, key="roi_tv")
        with c2:
            roi_radio = st.number_input("Radio Budget ($K)", 0.0, 50.0, 20.0, key="roi_radio")
        with c3:
            roi_np = st.number_input("Newspaper Budget ($K)", 0.0, 120.0, 20.0, key="roi_np")

        roi_X = pd.DataFrame([[roi_tv, roi_radio, roi_np]], columns=["TV", "Radio", "Newspaper"])
        roi_pred = st.session_state["best_model"].predict(roi_X)[0]
        total_spend = (roi_tv + roi_radio + roi_np) * 1000
        revenue = roi_pred * 1000
        roi_pct = ((revenue - total_spend) / total_spend * 100) if total_spend > 0 else 0

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Total Spend", f"${total_spend:,.0f}")
        rc2.metric("Est. Revenue", f"${revenue:,.0f}")
        rc3.metric("Net Profit", f"${revenue - total_spend:,.0f}")
        rc4.metric("ROI", f"{roi_pct:.1f}%",
                   delta="Positive" if roi_pct > 0 else "Negative")
