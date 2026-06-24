"""
app.py — Movie Rating Prediction Streamlit Dashboard
=====================================================
Run with:  streamlit run app.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
from pathlib import Path
import json
import io
import base64
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "data" / "IMDb Movies India.csv"
MODEL_DIR  = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 IMDb Rating Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
def inject_css(dark_mode: bool):
    if dark_mode:
        bg       = "#0f0f1a"
        card_bg  = "#1a1a2e"
        accent   = "#6366f1"
        accent2  = "#f59e0b"
        text     = "#e2e8f0"
        subtext  = "#94a3b8"
        border   = "#2d2d4e"
        inp_bg   = "#1e1e3a"
        metric_bg= "#16213e"
        star_col = "#f59e0b"
    else:
        bg       = "#f8f9ff"
        card_bg  = "#ffffff"
        accent   = "#4f46e5"
        accent2  = "#f59e0b"
        text     = "#1e293b"
        subtext  = "#64748b"
        border   = "#e2e8f0"
        inp_bg   = "#f1f5f9"
        metric_bg= "#eef2ff"
        star_col = "#d97706"

    st.markdown(f"""
    <style>
    /* ── Base ── */
    .stApp {{
        background: {bg};
        color: {text};
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background: {card_bg};
        border-right: 1px solid {border};
    }}
    /* ── Cards ── */
    .movie-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 24px 28px;
        margin: 10px 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        transition: transform .15s;
    }}
    .movie-card:hover {{ transform: translateY(-2px); }}
    /* ── Hero banner ── */
    .hero {{
        background: linear-gradient(135deg, {accent} 0%, #7c3aed 50%, {accent2} 100%);
        border-radius: 20px;
        padding: 40px 36px;
        color: white;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }}
    .hero h1 {{ font-size: 2.4rem; font-weight: 800; margin: 0; letter-spacing: -.5px; }}
    .hero p  {{ font-size: 1.05rem; opacity: .88; margin: 8px 0 0; }}
    /* ── Rating badge ── */
    .rating-badge {{
        background: linear-gradient(135deg, {accent2}, #ef4444);
        color: white;
        font-size: 3.2rem;
        font-weight: 900;
        border-radius: 20px;
        padding: 28px 40px;
        text-align: center;
        letter-spacing: -1px;
        box-shadow: 0 8px 32px rgba(245,158,11,.35);
    }}
    .rating-stars {{ font-size: 2rem; letter-spacing: 2px; }}
    /* ── Metrics ── */
    .metric-box {{
        background: {metric_bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }}
    .metric-box .val {{ font-size: 1.8rem; font-weight: 800; color: {accent}; }}
    .metric-box .lbl {{ font-size: .8rem; color: {subtext}; margin-top: 4px; text-transform: uppercase; letter-spacing: .5px; }}
    /* ── Section header ── */
    .section-header {{
        font-size: 1.35rem;
        font-weight: 700;
        color: {text};
        border-left: 4px solid {accent};
        padding-left: 12px;
        margin: 24px 0 16px;
    }}
    /* ── Best model badge ── */
    .best-badge {{
        display: inline-block;
        background: linear-gradient(90deg, {accent}, #7c3aed);
        color: white;
        border-radius: 999px;
        padding: 6px 18px;
        font-weight: 700;
        font-size: .95rem;
    }}
    /* ── Table ── */
    .styled-table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
    .styled-table th {{ background: {accent}; color: white; padding: 10px 14px; text-align: left; }}
    .styled-table td {{ padding: 10px 14px; border-bottom: 1px solid {border}; color: {text}; }}
    .styled-table tr:nth-child(even) td {{ background: {inp_bg}; }}
    /* ── Sidebar nav ── */
    .nav-item {{
        display: flex; align-items: center; gap: 10px;
        padding: 10px 14px; border-radius: 10px;
        font-weight: 600; cursor: pointer; margin: 3px 0;
        transition: background .15s;
    }}
    /* ── Footer ── */
    .footer {{ text-align: center; color: {subtext}; font-size: .8rem; padding: 28px 0 8px; }}
    /* ── Hide Streamlit branding ── */
    #MainMenu, footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH, encoding="latin1")

@st.cache_data
def load_clean_data():
    df = load_raw_data()
    df["Year"]     = df["Year"].astype(str).str.extract(r"(\d{4})").astype(float)
    df["Duration"] = df["Duration"].astype(str).str.extract(r"(\d+)").astype(float)
    df["Votes"]    = (df["Votes"].astype(str).str.replace(",","",regex=False)
                      .str.extract(r"(\d+\.?\d*)").astype(float))
    df = df.dropna(subset=["Rating"])
    df = df.drop_duplicates()
    for col in ["Genre","Director","Actor 1","Actor 2","Actor 3"]:
        df[col] = df[col].fillna("Unknown").str.strip()
    for col in ["Year","Duration","Votes"]:
        df[col] = df[col].fillna(df[col].median())
    df = df[(df["Rating"] >= 1) & (df["Rating"] <= 10)]
    df["Log_Votes"] = np.log1p(df["Votes"])
    df["Movie_Age"] = 2025 - df["Year"]
    return df

@st.cache_resource
def load_model_artifacts():
    model     = joblib.load(MODEL_DIR / "best_model.pkl")
    encoders  = joblib.load(MODEL_DIR / "encoders.pkl")
    scaler    = joblib.load(MODEL_DIR / "scaler.pkl")
    feat_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
    return model, encoders, scaler, feat_cols

def models_trained():
    return (MODEL_DIR / "best_model.pkl").exists()

def load_results():
    path = MODEL_DIR / "model_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

def load_summary():
    path = MODEL_DIR / "summary.pkl"
    if path.exists():
        return joblib.load(path)
    return {}

def star_string(rating: float) -> str:
    full  = int(round(rating / 2))
    empty = 5 - full
    return "⭐" * full + "☆" * empty

def render_asset(fname):
    p = ASSETS_DIR / fname
    if p.exists():
        st.image(str(p), use_container_width=True)

# ── Prediction logic (inline, same as predict.py) ─────────────────────────────

def predict_rating(genre, director, actor1, actor2, actor3, duration, votes, year):
    model, encoders, scaler, feat_cols = load_model_artifacts()
    primary_genre = genre.split(",")[0].strip() if genre else "Unknown"
    genre_count   = len(genre.split(",")) if genre else 1
    log_votes     = np.log1p(float(votes))
    movie_age     = 2025 - int(year)

    row = {
        "Year":              float(year),
        "Duration":          float(duration),
        "Log_Votes":         log_votes,
        "Movie_Age":         movie_age,
        "Genre_Count":       genre_count,
        "Director_freq":     encoders["Director"].get(director, 0.0),
        "Actor 1_freq":      encoders["Actor 1"].get(actor1, 0.0),
        "Actor 2_freq":      encoders["Actor 2"].get(actor2, 0.0),
        "Actor 3_freq":      encoders["Actor 3"].get(actor3, 0.0),
        "Primary_Genre_freq": encoders["Primary_Genre"].get(primary_genre, 0.0),
    }
    for g in encoders.get("top_genres", []):
        row[f"genre_{g.replace(' ','_')}"] = 1 if primary_genre == g else 0

    df_in   = pd.DataFrame([row]).reindex(columns=feat_cols, fill_value=0)
    X_sc    = scaler.transform(df_in.values)
    rating  = float(np.clip(model.predict(X_sc)[0], 1.0, 10.0))

    results = load_results()
    rmse = float(results.iloc[0]["RMSE"]) if results is not None else 0.8
    return {
        "predicted_rating": round(rating, 1),
        "confidence_range": (round(max(1.0, rating - rmse/2), 1),
                              round(min(10.0, rating + rmse/2), 1)),
        "rmse": round(rmse, 3),
    }

# ── Download report helper ─────────────────────────────────────────────────────

def make_report(inputs: dict, result: dict) -> str:
    lines = [
        "=" * 50,
        "  MOVIE RATING PREDICTION REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 50,
        "",
        "INPUT DETAILS:",
    ]
    for k, v in inputs.items():
        lines.append(f"  {k:<18}: {v}")
    lines += [
        "",
        "PREDICTION RESULT:",
        f"  Predicted Rating  : {result['predicted_rating']} / 10",
        f"  Confidence Range  : {result['confidence_range'][0]} – {result['confidence_range'][1]}",
        f"  Model RMSE        : ± {result['rmse']}",
        "",
        "=" * 50,
    ]
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE RENDERERS
# ═══════════════════════════════════════════════════════════════════════════════

def page_home():
    summary = load_summary()
    st.markdown("""
    <div class="hero">
        <h1>🎬 IMDb Movie Rating Predictor</h1>
        <p>Machine-learning powered rating prediction for Indian cinema — powered by IMDb data</p>
    </div>
    """, unsafe_allow_html=True)

    if summary:
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            (c1, summary.get("n_movies", "–"),  "Movies in Dataset"),
            (c2, summary.get("avg_rating", "–"), "Avg IMDb Rating"),
            (c3, summary.get("best_model", "–"), "Best Model"),
            (c4, f'{summary.get("best_r2","–")}', "R² Score"),
        ]
        for col, val, label in metrics:
            col.markdown(f"""
            <div class="metric-box">
                <div class="val">{val}</div>
                <div class="lbl">{label}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.warning("⚠️  Models not trained yet. Run **train.py** first, then restart the app.")

    st.markdown('<div class="section-header">📂 Dataset Overview</div>', unsafe_allow_html=True)
    df = load_raw_data()
    st.dataframe(df.head(20), use_container_width=True, height=320)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">🔍 Missing Values</div>', unsafe_allow_html=True)
        miss = df.isnull().sum().reset_index()
        miss.columns = ["Column", "Missing"]
        miss["% Missing"] = (miss["Missing"] / len(df) * 100).round(1)
        st.dataframe(miss, use_container_width=True, hide_index=True)
    with c2:
        st.markdown('<div class="section-header">📊 Column Types</div>', unsafe_allow_html=True)
        dtypes = df.dtypes.reset_index()
        dtypes.columns = ["Column", "Type"]
        dtypes["Sample"] = [str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0] else "" for c in df.columns]
        st.dataframe(dtypes, use_container_width=True, hide_index=True)


def page_analysis():
    df = load_clean_data()
    st.markdown('<div class="section-header">📈 Rating Distribution</div>', unsafe_allow_html=True)
    render_asset("rating_dist.png")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">🎭 Genre Popularity</div>', unsafe_allow_html=True)
        render_asset("genre_popularity.png")
    with c2:
        st.markdown('<div class="section-header">🎬 Top Directors</div>', unsafe_allow_html=True)
        render_asset("top_directors.png")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">🗳️ Votes vs Rating</div>', unsafe_allow_html=True)
        render_asset("votes_vs_rating.png")
    with c2:
        st.markdown('<div class="section-header">🔗 Correlation Heatmap</div>', unsafe_allow_html=True)
        render_asset("correlation.png")

    # Live interactive stats
    st.markdown('<div class="section-header">🔢 Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df[["Rating","Year","Duration","Votes","Movie_Age"]].describe().round(2),
                 use_container_width=True)

    st.markdown('<div class="section-header">🏆 Top 10 Rated Movies (≥ 100 votes)</div>', unsafe_allow_html=True)
    top_movies = (df[df["Votes"] >= 100]
                  .sort_values("Rating", ascending=False)
                  .head(10)[["Name","Year","Genre","Director","Rating","Votes"]]
                  .reset_index(drop=True))
    top_movies.index += 1
    st.dataframe(top_movies, use_container_width=True)


def page_model_performance():
    if not models_trained():
        st.warning("⚠️  Run `train.py` first.")
        return

    results = load_results()
    summary = load_summary()

    st.markdown(f"""
    <div class="hero" style="padding:28px 32px;">
        <h1 style="font-size:1.8rem;">🏆 Best Model: {summary.get('best_model','–')}</h1>
        <p>R² = {summary.get('best_r2','–')} &nbsp;|&nbsp; RMSE = {summary.get('best_rmse','–')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📋 Model Comparison Table</div>', unsafe_allow_html=True)
    if results is not None:
        # Highlight best row
        def highlight_best(row):
            return ["background-color: #4f46e520; font-weight:bold" if row.name == 0 else "" for _ in row]
        st.dataframe(
            results.style.apply(highlight_best, axis=1).format(
                {"MAE": "{:.4f}", "MSE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}"}
            ),
            use_container_width=True, hide_index=True
        )

    # Bar chart comparison
    if results is not None:
        st.markdown('<div class="section-header">📊 R² Score Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ["#6366f1" if i == 0 else "#94a3b8" for i in range(len(results))]
        bars = ax.barh(results["Model"], results["R2"], color=colors)
        ax.set_xlabel("R² Score")
        ax.set_title("Model R² Comparison", fontweight="bold")
        ax.bar_label(bars, fmt="%.4f", padding=4)
        ax.set_xlim(0, max(results["R2"].max() + 0.05, 0.2))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="section-header">📉 RMSE Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        colors2 = ["#f59e0b" if i == 0 else "#cbd5e1" for i in range(len(results))]
        bars2 = ax.barh(results["Model"], results["RMSE"], color=colors2)
        ax.set_xlabel("RMSE (lower is better)")
        ax.set_title("Model RMSE Comparison", fontweight="bold")
        ax.bar_label(bars2, fmt="%.4f", padding=4)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Feature importance
    st.markdown('<div class="section-header">🔍 Feature Importance</div>', unsafe_allow_html=True)
    render_asset("feature_importance.png")


def page_predict():
    if not models_trained():
        st.warning("⚠️  Run `train.py` first, then restart the app.")
        return

    df = load_clean_data()

    st.markdown("""
    <div class="hero" style="padding:28px 32px;">
        <h1 style="font-size:1.9rem;">⭐ Predict Movie Rating</h1>
        <p>Enter movie details below to get an AI-predicted IMDb rating</p>
    </div>
    """, unsafe_allow_html=True)

    # Pre-populate drop-down options from data
    all_genres   = sorted(set(g.strip() for genres in df["Genre"].dropna()
                               for g in genres.split(",")))
    all_directors = sorted(df["Director"].dropna().unique().tolist())
    all_actors    = sorted(set(
        df["Actor 1"].dropna().tolist() +
        df["Actor 2"].dropna().tolist() +
        df["Actor 3"].dropna().tolist()
    ))

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🎭 Movie Details")
            genre_options = all_genres
            genre_sel = st.multiselect(
                "Genre(s)", genre_options,
                default=["Drama"],
                help="Select one or more genres"
            )
            director = st.selectbox("Director", ["(New / Unknown)"] + all_directors)
            year     = st.slider("Release Year", 1950, 2025, 2020)
            duration = st.number_input("Duration (minutes)", min_value=20, max_value=300, value=130)

        with c2:
            st.markdown("#### 🌟 Cast & Popularity")
            actor1   = st.selectbox("Lead Actor",   ["(New / Unknown)"] + all_actors)
            actor2   = st.selectbox("Actor 2",      ["(New / Unknown)"] + all_actors)
            actor3   = st.selectbox("Actor 3",      ["(New / Unknown)"] + all_actors)
            votes    = st.number_input("Expected Votes", min_value=0, max_value=500000, value=5000, step=500)

        submitted = st.form_submit_button(
            "🎯  Predict Rating", use_container_width=True, type="primary"
        )

    if submitted:
        genre_str = ", ".join(genre_sel) if genre_sel else "Drama"
        dir_val   = "" if director.startswith("(") else director
        a1_val    = "" if actor1.startswith("(")   else actor1
        a2_val    = "" if actor2.startswith("(")   else actor2
        a3_val    = "" if actor3.startswith("(")   else actor3

        with st.spinner("Predicting …"):
            result = predict_rating(genre_str, dir_val, a1_val, a2_val, a3_val,
                                     duration, votes, year)

        r = result["predicted_rating"]
        lo, hi = result["confidence_range"]
        stars = star_string(r)

        st.markdown("---")
        c1, c2, c3 = st.columns([2, 3, 2])
        with c2:
            st.markdown(f"""
            <div class="rating-badge">
                <div style="font-size:1rem;opacity:.85;margin-bottom:4px;">PREDICTED IMDb RATING</div>
                <div>{r} / 10</div>
                <div class="rating-stars">{stars}</div>
                <div style="font-size:1rem;opacity:.75;margin-top:8px;">
                    Confidence range: {lo} – {hi}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-header">📝 Input Summary</div>', unsafe_allow_html=True)
        inputs = {
            "Genre": genre_str, "Director": dir_val or "(New)",
            "Lead Actor": a1_val or "(New)", "Actor 2": a2_val or "(New)",
            "Actor 3": a3_val or "(New)", "Duration": f"{duration} min",
            "Expected Votes": votes, "Year": year,
        }
        summary_df = pd.DataFrame(inputs.items(), columns=["Field", "Value"])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Download report
        report = make_report(inputs, result)
        st.download_button(
            "⬇️  Download Prediction Report",
            data=report,
            file_name=f"rating_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )


def page_about():
    st.markdown("""
    <div class="hero">
        <h1>ℹ️ About This Project</h1>
        <p>An end-to-end machine-learning pipeline for Indian movie rating prediction</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="movie-card">
        <div class="section-header" style="margin-top:0">🎯 Objective</div>
        <p>Predict IMDb ratings for Indian movies based on genre, cast, director,
        release year, duration and vote count — enabling informed production decisions
        and audience targeting before release.</p>

        <div class="section-header">🗃️ Dataset</div>
        <ul>
          <li><b>Source:</b> IMDb India Movies dataset</li>
          <li><b>Size:</b> ~15,000+ movies</li>
          <li><b>Features:</b> Name, Year, Duration, Genre, Rating, Votes, Director, 3 Actors</li>
          <li><b>Target:</b> IMDb Rating (1–10)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="movie-card">
        <div class="section-header" style="margin-top:0">🛠️ Technologies</div>
        <ul>
          <li><b>Python 3.10+</b></li>
          <li><b>Pandas / NumPy</b> — data processing</li>
          <li><b>Scikit-learn</b> — ML models, preprocessing</li>
          <li><b>Matplotlib / Seaborn</b> — visualisations</li>
          <li><b>Joblib</b> — model persistence</li>
          <li><b>Streamlit</b> — interactive web dashboard</li>
        </ul>

        <div class="section-header">🤖 Models Trained</div>
        <ul>
          <li>Linear Regression</li>
          <li>Decision Tree Regressor</li>
          <li>Random Forest Regressor</li>
          <li>Gradient Boosting Regressor</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="movie-card">
    <div class="section-header" style="margin-top:0">⚙️ Feature Engineering Techniques</div>
    <table class="styled-table">
      <tr><th>Technique</th><th>Applied to</th><th>Rationale</th></tr>
      <tr><td>Frequency Encoding</td><td>Director, Actors, Genre</td><td>Captures popularity proxy without high-dim OHE</td></tr>
      <tr><td>One-Hot Encoding</td><td>Top 15 primary genres</td><td>Explicit genre signal for common categories</td></tr>
      <tr><td>Log Transform</td><td>Votes</td><td>Normalises heavy right-skew distribution</td></tr>
      <tr><td>Derived Features</td><td>Movie_Age, Genre_Count</td><td>Additional temporal and diversity signals</td></tr>
      <tr><td>StandardScaler</td><td>All numeric features</td><td>Required by linear models; helps tree models too</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="movie-card">
    <div class="section-header" style="margin-top:0">🚀 Deployment Options</div>
    <table class="styled-table">
      <tr><th>Platform</th><th>Steps</th></tr>
      <tr><td><b>Streamlit Cloud</b></td><td>Push to GitHub → connect at share.streamlit.io → set requirements.txt</td></tr>
      <tr><td><b>Render</b></td><td>New Web Service → connect GitHub repo → Build: <code>pip install -r requirements.txt</code> → Start: <code>streamlit run app.py --server.port $PORT</code></td></tr>
      <tr><td><b>Railway</b></td><td>Connect repo → add <code>Procfile</code>: <code>web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0</code></td></tr>
    </table>
    </div>
    <div class="footer">Built with ❤️ using Python + Streamlit &nbsp;|&nbsp; IMDb India Movies Dataset</div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Dark/light mode toggle
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        dark_mode = st.toggle("🌙 Dark Mode", value=True)
        st.markdown("---")

    inject_css(dark_mode)

    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🎬 Movie Rating Predictor")
        st.markdown("---")
        pages = {
            "🏠 Home":               "home",
            "📊 Data Analysis":      "analysis",
            "🤖 Model Performance":  "models",
            "⭐ Predict Rating":     "predict",
            "ℹ️ About":              "about",
        }
        selected = st.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
        st.markdown("---")
        st.markdown("""
        <div style="font-size:.8rem;opacity:.6;text-align:center;">
        IMDb India Movies<br>ML Rating Prediction
        </div>
        """, unsafe_allow_html=True)

    page_key = pages[selected]
    if page_key == "home":
        page_home()
    elif page_key == "analysis":
        page_analysis()
    elif page_key == "models":
        page_model_performance()
    elif page_key == "predict":
        page_predict()
    elif page_key == "about":
        page_about()


if __name__ == "__main__":
    main()
