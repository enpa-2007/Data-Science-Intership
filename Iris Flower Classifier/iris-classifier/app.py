"""
app.py  –  Iris Flower Classifier  |  Main entry-point
------------------------------------------------------
Run with:  streamlit run app.py
"""

import streamlit as st
import importlib
import sys
import os

# ── Make sub-pages importable ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pages"))

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── CSS Variables ─────────────────────────────── */
:root {
    --green-dark:   #1a6b3c;
    --green-mid:    #2d9e5f;
    --green-light:  #4eca85;
    --green-pale:   #e6f7ef;
    --blue-dark:    #0d3b6b;
    --blue-mid:     #1a6bbd;
    --blue-light:   #3d9de8;
    --blue-pale:    #e8f3fc;
    --white:        #ffffff;
    --off-white:    #f8fafb;
    --gray-100:     #f1f5f9;
    --gray-200:     #e2e8f0;
    --gray-400:     #94a3b8;
    --gray-700:     #334155;
    --gray-900:     #0f172a;
    --shadow-sm:    0 1px 3px rgba(0,0,0,.08);
    --shadow-md:    0 4px 16px rgba(0,0,0,.10);
    --shadow-lg:    0 8px 32px rgba(0,0,0,.12);
    --radius:       12px;
    --radius-lg:    20px;
}

/* ── Base ──────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

/* ── Gradient Header ───────────────────────────── */
.iris-header {
    background: linear-gradient(135deg, #1a6b3c 0%, #1a6bbd 60%, #0d3b6b 100%);
    border-radius: var(--radius-lg);
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.iris-header::before {
    content: "🌸";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: .15;
}
.iris-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    margin: 0 0 .4rem 0;
    letter-spacing: -.5px;
}
.iris-header p {
    font-size: 1.05rem;
    opacity: .88;
    margin: 0;
    font-weight: 300;
}

/* ── Metric Cards ──────────────────────────────── */
.metric-card {
    background: var(--white);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    box-shadow: var(--shadow-md);
    border-left: 4px solid var(--green-mid);
    transition: transform .15s, box-shadow .15s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-lg); }
.metric-card .label { font-size: .78rem; font-weight: 600; color: var(--gray-400); text-transform: uppercase; letter-spacing: .8px; margin-bottom: .3rem; }
.metric-card .value { font-size: 2rem; font-weight: 700; color: var(--green-dark); line-height: 1; }
.metric-card .sub   { font-size: .82rem; color: var(--gray-400); margin-top: .25rem; }

/* ── Section Headers ───────────────────────────── */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: var(--gray-900);
    margin: 1.8rem 0 .8rem 0;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.section-title::after {
    content: "";
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, var(--green-light), transparent);
    margin-left: .5rem;
}

/* ── Species Badges ────────────────────────────── */
.species-badge {
    display: inline-block;
    padding: .3rem .9rem;
    border-radius: 50px;
    font-size: .82rem;
    font-weight: 600;
    letter-spacing: .4px;
}
.badge-setosa     { background: #e6f7ef; color: #1a6b3c; }
.badge-versicolor { background: #e8f3fc; color: #0d3b6b; }
.badge-virginica  { background: #fdf2fa; color: #8b2fc9; }

/* ── Sidebar ───────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d3b6b 0%, #1a6b3c 100%) !important;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label { font-size: .95rem !important; }
[data-testid="stSidebarNav"] { display: none; }

/* ── Buttons ───────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--green-mid), var(--blue-mid)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: .55rem 1.5rem !important;
    transition: opacity .15s !important;
}
.stButton > button:hover { opacity: .88 !important; }

/* ── Prediction Result Box ─────────────────────── */
.prediction-box {
    background: linear-gradient(135deg, var(--green-pale), var(--blue-pale));
    border-radius: var(--radius-lg);
    padding: 2rem;
    text-align: center;
    border: 2px solid var(--green-light);
    margin: 1rem 0;
}
.prediction-box .species-name {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: var(--green-dark);
}
.prediction-box .confidence {
    font-size: 1rem;
    color: var(--gray-700);
    margin-top: .4rem;
}

/* ── DataFrame ─────────────────────────────────── */
.stDataFrame { border-radius: var(--radius) !important; overflow: hidden; }

/* ── Progress Bars ─────────────────────────────── */
.stProgress > div > div { background: var(--green-mid) !important; }

/* ── Tabs ──────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 500;
}

/* ── Dark Mode ─────────────────────────────────── */
@media (prefers-color-scheme: dark) {
    .metric-card { background: #1e293b; }
    .metric-card .value { color: var(--green-light); }
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar Navigation ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 1rem'>
        <div style='font-size:3rem'>🌸</div>
        <div style='font-family:"DM Serif Display",serif; font-size:1.3rem; font-weight:700; margin-top:.4rem'>
            Iris Classifier
        </div>
        <div style='font-size:.8rem; opacity:.7; margin-top:.2rem'>
            ML Portfolio Project
        </div>
    </div>
    <hr style='border-color:rgba(255,255,255,.2); margin: 0 0 1rem 0'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home",
         "📊  Dataset Explorer",
         "📈  Visualizations",
         "🤖  Model Training",
         "🔮  Predict Species",
         "📋  Model Evaluation",
         "⭐  Feature Importance"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,.2); margin:1.5rem 0 1rem'>", unsafe_allow_html=True)
    if st.button("🔄  Retrain Models"):
        with st.spinner("Retraining…"):
            from train_model import get_trained_models
            get_trained_models(force_retrain=True)
            if "model_data" in st.session_state:
                del st.session_state["model_data"]
        st.success("Models retrained!")

    st.markdown("""
    <div style='font-size:.72rem; opacity:.55; text-align:center; margin-top:2rem; line-height:1.6'>
        Built with Streamlit · Scikit-Learn<br>
        Pandas · Plotly · Seaborn
    </div>
    """, unsafe_allow_html=True)


# ─── Route to page ─────────────────────────────────────────────────────────
page_key = page.split("  ", 1)[1].strip()

if page_key == "Home":
    import Home as pg
elif page_key == "Dataset Explorer":
    import Dataset as pg
elif page_key == "Visualizations":
    import Visualization as pg
elif page_key == "Model Training":
    import ModelTraining as pg
elif page_key == "Predict Species":
    import Prediction as pg
elif page_key == "Model Evaluation":
    import Evaluation as pg
elif page_key == "Feature Importance":
    import FeatureImportance as pg

pg.show()
