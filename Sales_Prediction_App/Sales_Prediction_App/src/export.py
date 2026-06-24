import streamlit as st
import pandas as pd
import pickle, io, os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_data, preprocess_data
from utils.training import train_all_models

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
            "X_test": X_test, "y_test": y_test,
        })

def show():
    st.markdown("""
    <h2 style='background: linear-gradient(135deg,#00D4FF,#7B2FFF);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               margin-bottom:1.5rem;'>⚙️ Model Export & Downloads</h2>
    """, unsafe_allow_html=True)

    ensure_trained()

    model = st.session_state["best_model"]
    model_name = st.session_state["best_name"]
    results = st.session_state["results"]
    y_test = st.session_state["y_test"]
    X_test = st.session_state["X_test"]

    st.markdown("#### 🤖 Export Trained Model")
    st.markdown(f"""
    <div style='background:#111827; border:1px solid rgba(0,212,255,0.2);
                border-radius:12px; padding:1.2rem; margin-bottom:1rem;'>
        <div style='color:#94A3B8; font-size:0.85rem;'>Active Model</div>
        <div style='color:#00D4FF; font-size:1.2rem; font-weight:700;'>{model_name}</div>
        <div style='color:#64748B; font-size:0.8rem; margin-top:0.3rem;'>
            R² Score: {results[model_name]["R2 Score"]} | RMSE: {results[model_name]["RMSE"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    buffer = io.BytesIO()
    pickle.dump({"model": model, "name": model_name}, buffer)
    buffer.seek(0)
    st.download_button(
        "⬇️ Download Best Model (.pkl)",
        buffer,
        file_name=f"{model_name.replace(' ', '_')}_best_model.pkl",
        mime="application/octet-stream",
        use_container_width=True,
    )

    st.markdown("#### 📊 Export Predictions")
    preds = results[model_name]["Predictions"]
    pred_df = pd.DataFrame({
        "TV ($K)": X_test["TV"].values,
        "Radio ($K)": X_test["Radio"].values,
        "Newspaper ($K)": X_test["Newspaper"].values,
        "Actual Sales (K)": y_test.values,
        "Predicted Sales (K)": [round(p, 3) for p in preds],
        "Error": [round(a - p, 3) for a, p in zip(y_test.values, preds)],
    })
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    csv = pred_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Predictions (.csv)",
        csv,
        file_name="sales_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("#### 🏆 Full Model Leaderboard Export")
    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name, "MAE": m["MAE"], "MSE": m["MSE"],
            "RMSE": m["RMSE"], "R2 Score": m["R2 Score"],
            "CV R2 Mean": m["CV R2 Mean"],
        })
    leaderboard_df = pd.DataFrame(rows).sort_values("R2 Score", ascending=False)
    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)
    csv2 = leaderboard_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Leaderboard (.csv)",
        csv2,
        file_name="model_leaderboard.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("#### 📦 Export Raw Dataset")
    df = load_data()
    st.download_button(
        "⬇️ Download Dataset (.csv)",
        df.to_csv(index=False),
        file_name="sales_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )
