import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_data, preprocess_data, classify_sales
from utils.training import train_all_models, load_best_model
from utils.visualization import roi_chart

def ensure_model():
    if "best_model" not in st.session_state:
        df = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(df)
        results, trained_models, best_name, best_model = train_all_models(
            X_train, X_test, y_train, y_test
        )
        st.session_state["best_model"] = best_model
        st.session_state["best_name"] = best_name
        st.session_state["results"] = results
    return st.session_state["best_model"], st.session_state["best_name"]

def show():
    st.markdown("""
    <h2 style='background: linear-gradient(135deg,#00D4FF,#7B2FFF);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               margin-bottom:1.5rem;'>🎯 Sales Prediction</h2>
    """, unsafe_allow_html=True)

    model, model_name = ensure_model()

    st.markdown(f"""
    <div style='background:rgba(0,212,255,0.06); border:1px solid rgba(0,212,255,0.2);
                border-radius:10px; padding:0.8rem 1.2rem; margin-bottom:1.5rem;'>
        Active Model: <strong style='color:#00D4FF;'>{model_name}</strong>
        — go to <em>Model Training</em> to retrain.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### 💰 Enter Advertising Budgets")
        tv = st.slider("📺 TV Advertising Budget ($K)", 0.0, 300.0, 150.0, 1.0)
        radio = st.slider("📻 Radio Advertising Budget ($K)", 0.0, 50.0, 25.0, 0.5)
        newspaper = st.slider("📰 Newspaper Advertising Budget ($K)", 0.0, 120.0, 30.0, 1.0)

        st.markdown("#### Or Enter Exact Values")
        c1, c2, c3 = st.columns(3)
        with c1:
            tv_exact = st.number_input("TV ($K)", 0.0, 500.0, float(tv), key="tv_exact")
        with c2:
            radio_exact = st.number_input("Radio ($K)", 0.0, 100.0, float(radio), key="radio_exact")
        with c3:
            newspaper_exact = st.number_input("Newspaper ($K)", 0.0, 200.0, float(newspaper), key="np_exact")

        use_exact = st.checkbox("Use exact values above")
        final_tv = tv_exact if use_exact else tv
        final_radio = radio_exact if use_exact else radio
        final_newspaper = newspaper_exact if use_exact else newspaper

    with col2:
        st.markdown("#### 📊 Budget Summary")
        total = final_tv + final_radio + final_newspaper
        for ch, val, color in [("📺 TV", final_tv, "#00D4FF"), ("📻 Radio", final_radio, "#7B2FFF"), ("📰 Newspaper", final_newspaper, "#FF6B35")]:
            pct = (val / total * 100) if total > 0 else 0
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; align-items:center;
                        background:#111827; border-radius:8px; padding:0.6rem 1rem; margin-bottom:0.4rem;'>
                <span style='color:#CBD5E1;'>{ch}</span>
                <span style='color:{color}; font-weight:700;'>${val:.1f}K ({pct:.0f}%)</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E293B,#111827);
                    border:1px solid rgba(0,212,255,0.2); border-radius:8px;
                    padding:0.6rem 1rem; margin-top:0.4rem; text-align:center;'>
            <span style='color:#94A3B8;'>Total Budget: </span>
            <span style='color:#00D4FF; font-weight:800; font-size:1.1rem;'>${total:.1f}K</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Predict Sales", type="primary", use_container_width=True):
        X_input = pd.DataFrame([[final_tv, final_radio, final_newspaper]],
                               columns=["TV", "Radio", "Newspaper"])
        prediction = model.predict(X_input)[0]
        category, emoji = classify_sales(prediction)

        confidence = min(95, max(60, 70 + prediction * 0.8))

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(0,212,255,0.12),rgba(123,47,255,0.12));
                    border:2px solid rgba(0,212,255,0.4); border-radius:20px;
                    padding:2rem; text-align:center; margin:1rem 0;'>
            <div style='font-size:3.5rem; font-weight:900; letter-spacing:-2px;
                        background:linear-gradient(135deg,#00D4FF,#7B2FFF);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
                {prediction:.2f}K units
            </div>
            <div style='color:#94A3B8; font-size:1rem; margin-top:0.5rem;'>Predicted Sales</div>
            <div style='margin-top:1rem; font-size:1.3rem;'>{emoji} <strong style='color:#E2E8F0;'>{category}</strong></div>
            <div style='margin-top:0.8rem;'>
                <span style='background:rgba(0,255,159,0.1); border:1px solid rgba(0,255,159,0.3);
                             color:#00FF9F; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.85rem;'>
                    Confidence: {confidence:.0f}%
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            est_revenue = prediction * 1000
            roi = ((est_revenue - total * 1000) / (total * 1000) * 100) if total > 0 else 0
            st.metric("Estimated Revenue", f"${est_revenue:,.0f}", delta=f"ROI: {roi:.1f}%")
        with c2:
            st.metric("Revenue per $1 Spent", f"${est_revenue / (total*1000):.2f}" if total > 0 else "N/A")

        if total > 0:
            st.plotly_chart(roi_chart(final_tv, final_radio, final_newspaper, prediction),
                            use_container_width=True)

        result_df = pd.DataFrame({
            "TV ($K)": [final_tv], "Radio ($K)": [final_radio],
            "Newspaper ($K)": [final_newspaper], "Predicted Sales (K)": [round(prediction, 3)],
            "Category": [category], "Model Used": [model_name],
        })
        st.session_state["last_prediction"] = result_df

        csv = result_df.to_csv(index=False)
        st.download_button("⬇️ Download Prediction", csv, "prediction.csv", "text/csv")
