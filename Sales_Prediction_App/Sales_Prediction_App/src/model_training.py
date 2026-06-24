import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_data, preprocess_data
from utils.training import train_all_models
from utils.visualization import model_comparison_chart, actual_vs_predicted, residual_plot

def show():
    st.markdown("""
    <h2 style='background: linear-gradient(135deg,#00D4FF,#7B2FFF);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               margin-bottom:1.5rem;'>🤖 Model Training</h2>
    """, unsafe_allow_html=True)

    df = load_data()

    with st.expander("⚙️ Training Configuration", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        with c2:
            random_state = st.number_input("Random State", 0, 999, 42)

    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        with st.spinner("Training models… this may take a moment"):
            X_train, X_test, y_train, y_test = preprocess_data(df, test_size, random_state)
            results, trained_models, best_name, best_model = train_all_models(
                X_train, X_test, y_train, y_test
            )

        st.session_state["results"] = results
        st.session_state["trained_models"] = trained_models
        st.session_state["best_name"] = best_name
        st.session_state["best_model"] = best_model
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.success(f"✅ Training complete! Best model: **{best_name}**")

    if "results" in st.session_state:
        results = st.session_state["results"]
        best_name = st.session_state["best_name"]

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(0,255,159,0.1),rgba(0,212,255,0.1));
                    border:1px solid rgba(0,255,159,0.3); border-radius:12px; padding:1rem;
                    margin-bottom:1.5rem; text-align:center;'>
            <span style='font-size:1.1rem; color:#00FF9F; font-weight:700;'>
                🏆 Best Model: {best_name}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Performance Leaderboard")
        rows = []
        for name, m in results.items():
            rows.append({
                "Model": ("🥇 " if name == best_name else "") + name,
                "MAE": m["MAE"], "MSE": m["MSE"], "RMSE": m["RMSE"],
                "R² Score": m["R2 Score"], "CV R² Mean": m["CV R2 Mean"],
            })
        leaderboard = pd.DataFrame(rows).sort_values("R² Score", ascending=False)
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)

        st.plotly_chart(model_comparison_chart(results), use_container_width=True)

        st.markdown("#### Detailed Model Analysis")
        model_tabs = st.tabs(list(results.keys()))
        y_test = st.session_state["y_test"]
        trained_models = st.session_state["trained_models"]
        X_test = st.session_state["X_test"]
        for tab, (name, m) in zip(model_tabs, results.items()):
            with tab:
                cols = st.columns(4)
                cols[0].metric("MAE", m["MAE"])
                cols[1].metric("RMSE", m["RMSE"])
                cols[2].metric("R² Score", m["R2 Score"])
                cols[3].metric("CV R² Mean", m["CV R2 Mean"])
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        actual_vs_predicted(y_test, m["Predictions"], name),
                        use_container_width=True
                    )
                with c2:
                    st.plotly_chart(
                        residual_plot(y_test, m["Predictions"], name),
                        use_container_width=True
                    )
    else:
        st.info("👆 Configure settings and click **Train All Models** to begin.")
