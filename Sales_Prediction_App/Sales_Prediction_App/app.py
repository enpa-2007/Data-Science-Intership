import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="SalesPulse AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global Styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif !important; }

.stApp {
    background: #050A14;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(0,212,255,0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(123,47,255,0.05) 0%, transparent 50%);
}

[data-testid="stSidebar"] {
    background: #080D1A !important;
    border-right: 1px solid rgba(0,212,255,0.1);
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #00D4FF !important;
    font-size: 1.4rem !important;
}
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.78rem !important; }

.stButton > button {
    background: linear-gradient(135deg, #00D4FF, #7B2FFF) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,212,255,0.3) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0F172A !important;
    border-radius: 10px;
    padding: 0.3rem;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748B !important;
    border-radius: 8px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(123,47,255,0.15)) !important;
    color: #00D4FF !important;
    border-bottom: 2px solid #00D4FF !important;
}

.stSelectbox > div > div {
    background: #111827 !important;
    border-color: rgba(0,212,255,0.2) !important;
    color: #E2E8F0 !important;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #0F172A, #1E293B) !important;
    color: #00D4FF !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
}

.stSuccess { border-left: 4px solid #00FF9F; background: rgba(0,255,159,0.07) !important; }
.stInfo    { border-left: 4px solid #00D4FF; background: rgba(0,212,255,0.07) !important; }
.stWarning { border-left: 4px solid #FFD700; background: rgba(255,215,0,0.07) !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080D1A; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1.5rem 0 1rem;'>
        <div style='font-size:2rem; font-weight:900;
                    background:linear-gradient(135deg,#00D4FF,#7B2FFF);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            📊 SalesPulse
        </div>
        <div style='font-size:0.65rem; letter-spacing:3px; color:#334155;
                    text-transform:uppercase; margin-top:0.2rem;'>AI · v2.0</div>
    </div>
    <hr style='border-color:rgba(0,212,255,0.1); margin:0.5rem 0 1rem;'>
    """, unsafe_allow_html=True)

    PAGES = {
        "🏠  Dashboard":          "dashboard",
        "🔍  Data Exploration":   "data_exploration",
        "🤖  Model Training":     "model_training",
        "🎯  Prediction":         "prediction",
        "💡  Business Insights":  "insights",
        "📊  Visual Analytics":   "analytics",
        "⚙️  Export & Downloads": "export",
    }

    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "dashboard"

    for label, key in PAGES.items():
        is_active = st.session_state["active_page"] == key
        if st.sidebar.button(label, use_container_width=True, key=f"nav_{key}"):
            st.session_state["active_page"] = key
            st.rerun()

    st.markdown("""
    <hr style='border-color:rgba(0,212,255,0.1); margin:1rem 0 0.5rem;'>
    <div style='font-size:0.7rem; color:#334155; text-align:center;'>
        Built with Streamlit · Scikit-learn<br>© 2025 SalesPulse AI
    </div>
    """, unsafe_allow_html=True)

# ── Route to Page ─────────────────────────────────────────────────────────────
page = st.session_state.get("active_page", "dashboard")

if page == "dashboard":
    from src import dashboard; dashboard.show()
elif page == "data_exploration":
    from src import data_exploration; data_exploration.show()
elif page == "model_training":
    from src import model_training; model_training.show()
elif page == "prediction":
    from src import prediction; prediction.show()
elif page == "insights":
    from src import insights; insights.show()
elif page == "analytics":
    from src import analytics; analytics.show()
elif page == "export":
    from src import export; export.show()
