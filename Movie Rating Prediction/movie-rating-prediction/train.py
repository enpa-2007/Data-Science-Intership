"""
train.py — Movie Rating Prediction Model Training Pipeline
===========================================================
Loads IMDb India data, cleans it, engineers features, trains multiple
regression models, evaluates them, and saves the best model + artefacts.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import defaultdict

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "IMDb Movies India.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ─── 1. Load Data ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print("📂 Loading dataset …")
    df = pd.read_csv(path, encoding="latin1")
    print(f"   Shape: {df.shape}")
    return df

# ─── 2. Clean Data ────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("🧹 Cleaning data …")
    df = df.copy()

    # --- Year: extract 4-digit year from strings like "(2019)"
    df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})").astype(float)

    # --- Duration: extract numeric minutes
    df["Duration"] = df["Duration"].astype(str).str.extract(r"(\d+)").astype(float)

    # --- Votes: remove commas, cast to float
    df["Votes"] = (
        df["Votes"].astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+\.?\d*)")
        .astype(float)
    )

    # --- Drop rows where target (Rating) is missing
    df = df.dropna(subset=["Rating"])

    # --- Drop duplicates
    df = df.drop_duplicates()

    # --- Fill remaining NaN for categorical columns with "Unknown"
    for col in ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]:
        df[col] = df[col].fillna("Unknown").str.strip()

    # --- Fill numeric NaN with medians
    for col in ["Year", "Duration", "Votes"]:
        df[col] = df[col].fillna(df[col].median())

    # --- Remove obvious outliers (rating must be 1-10)
    df = df[(df["Rating"] >= 1.0) & (df["Rating"] <= 10.0)]

    print(f"   Clean shape: {df.shape}")
    return df

# ─── 3. Feature Engineering ───────────────────────────────────────────────────

def build_frequency_encoder(series: pd.Series) -> dict:
    """Map each category to its frequency (proportion) in the series."""
    freq = series.value_counts(normalize=True).to_dict()
    return defaultdict(float, freq)


def engineer_features(df: pd.DataFrame):
    print("⚙️  Engineering features …")
    df = df.copy()

    # --- Primary genre (first genre listed)
    df["Primary_Genre"] = df["Genre"].str.split(",").str[0].str.strip()

    # --- Genre count
    df["Genre_Count"] = df["Genre"].str.split(",").str.len()

    # --- Log-transform Votes (heavy right-skew)
    df["Log_Votes"] = np.log1p(df["Votes"])

    # --- Movie age
    df["Movie_Age"] = 2025 - df["Year"]

    # --- Frequency encodings for high-cardinality categoricals
    encoders = {}
    for col in ["Director", "Actor 1", "Actor 2", "Actor 3", "Primary_Genre"]:
        enc = build_frequency_encoder(df[col])
        encoders[col] = enc
        df[f"{col}_freq"] = df[col].map(enc)

    # --- One-hot encode top-N genres only (keep top 15)
    top_genres = df["Primary_Genre"].value_counts().nlargest(15).index.tolist()
    for g in top_genres:
        df[f"genre_{g.replace(' ', '_')}"] = (df["Primary_Genre"] == g).astype(int)
    encoders["top_genres"] = top_genres

    return df, encoders


def get_feature_columns(df: pd.DataFrame) -> list:
    base = [
        "Year", "Duration", "Log_Votes", "Movie_Age",
        "Genre_Count",
        "Director_freq", "Actor 1_freq", "Actor 2_freq", "Actor 3_freq",
        "Primary_Genre_freq",
    ]
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    return base + genre_cols

# ─── 4. Train Models ──────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    mse  = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, preds)
    return {"Model": name, "MAE": round(mae, 4), "MSE": round(mse, 4),
            "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def train_models(X_train, X_test, y_train, y_test):
    print("🤖 Training models …")
    models = {
        "Linear Regression":        LinearRegression(),
        "Decision Tree":            DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest":            RandomForestRegressor(n_estimators=200, max_depth=12,
                                                          n_jobs=-1, random_state=42),
        "Gradient Boosting":        GradientBoostingRegressor(n_estimators=200,
                                                               learning_rate=0.05,
                                                               max_depth=5,
                                                               random_state=42),
    }

    results = []
    trained = {}
    for name, model in models.items():
        print(f"   Training {name} …", end=" ")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, name)
        trained[name] = model
        results.append(metrics)
        print(f"R²={metrics['R2']}  RMSE={metrics['RMSE']}")

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    return trained, results_df

# ─── 5. Save Artifacts ────────────────────────────────────────────────────────

def save_artifacts(best_model, encoders, scaler, feature_cols, results_df):
    joblib.dump(best_model,   os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(encoders,     os.path.join(MODEL_DIR, "encoders.pkl"))
    joblib.dump(scaler,       os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))
    results_df.to_csv(os.path.join(MODEL_DIR, "model_results.csv"), index=False)
    print("💾 Artifacts saved to models/")

# ─── 6. EDA Plots ─────────────────────────────────────────────────────────────

def save_eda_plots(df: pd.DataFrame):
    print("📊 Generating EDA plots …")

    # Rating distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Rating"], bins=30, kde=True, color="#6366f1", ax=ax)
    ax.set_title("Rating Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("IMDb Rating")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "rating_dist.png"), dpi=120)
    plt.close()

    # Top directors by average rating
    top_dir = (
        df[df["Director"] != "Unknown"]
        .groupby("Director")["Rating"]
        .agg(["mean", "count"])
        .query("count >= 5")
        .sort_values("mean", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x="mean", y=top_dir.index, data=top_dir.reset_index(),
                palette="viridis", ax=ax)
    ax.set_title("Top Directors by Avg Rating (≥5 movies)", fontweight="bold")
    ax.set_xlabel("Average Rating")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "top_directors.png"), dpi=120)
    plt.close()

    # Genre popularity
    genre_counts = df["Genre"].str.split(",").explode().str.strip().value_counts().head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="magma", ax=ax)
    ax.set_title("Top 15 Genres by Movie Count", fontweight="bold")
    ax.set_xlabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "genre_popularity.png"), dpi=120)
    plt.close()

    # Votes vs Rating scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(np.log1p(df["Votes"]), df["Rating"], alpha=0.15, s=6, color="#f59e0b")
    ax.set_title("Log(Votes) vs Rating", fontweight="bold")
    ax.set_xlabel("log(1 + Votes)")
    ax.set_ylabel("Rating")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "votes_vs_rating.png"), dpi=120)
    plt.close()

    # Correlation heatmap
    num_cols = ["Rating", "Year", "Duration", "Votes", "Movie_Age", "Log_Votes"]
    corr_df = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "correlation.png"), dpi=120)
    plt.close()

    print("   EDA plots saved to assets/")

# ─── 7. Feature Importance Plot ───────────────────────────────────────────────

def save_feature_importance(model, feature_cols):
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=imp.values, y=imp.index, palette="rocket", ax=ax)
    ax.set_title("Top 20 Feature Importances (Best Model)", fontweight="bold")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS_DIR, "feature_importance.png"), dpi=120)
    plt.close()
    print("   Feature importance plot saved.")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  🎬  Movie Rating Prediction — Training Pipeline")
    print("="*60 + "\n")

    df_raw   = load_data(DATA_PATH)
    df_clean = clean_data(df_raw)
    df_feat, encoders = engineer_features(df_clean)
    feature_cols = get_feature_columns(df_feat)

    X = df_feat[feature_cols].fillna(0)
    y = df_feat["Rating"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    trained_models, results_df = train_models(X_train, X_test, y_train, y_test)

    print("\n📋 Model Comparison:")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    print(f"\n🏆 Best Model: {best_name}")

    # Save plots
    save_eda_plots(df_feat)
    save_feature_importance(best_model, feature_cols)

    # Save model artifacts
    save_artifacts(best_model, encoders, scaler, feature_cols, results_df)

    # Save dataset summary for the app
    summary = {
        "n_movies":   int(df_clean.shape[0]),
        "n_features": len(feature_cols),
        "avg_rating": round(float(df_clean["Rating"].mean()), 2),
        "top_genres": df_clean["Genre"].str.split(",").explode().str.strip()
                       .value_counts().head(10).to_dict(),
        "best_model": best_name,
        "best_r2":    float(results_df.iloc[0]["R2"]),
        "best_rmse":  float(results_df.iloc[0]["RMSE"]),
    }
    joblib.dump(summary, os.path.join(MODEL_DIR, "summary.pkl"))

    print("\n✅  Training complete!\n")


if __name__ == "__main__":
    main()
