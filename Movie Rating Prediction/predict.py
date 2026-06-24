"""
predict.py — Movie Rating Prediction Module
============================================
Loads saved model artefacts and provides a prediction pipeline.
"""

import os
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")


def load_artifacts():
    """Load model, encoders, scaler, and feature columns."""
    model      = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    encoders   = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    scaler     = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feat_cols  = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    return model, encoders, scaler, feat_cols


def predict_rating(
    genre:    str,
    director: str,
    actor1:   str,
    actor2:   str,
    actor3:   str,
    duration: float,
    votes:    float,
    year:     int,
) -> dict:
    """
    Predict IMDb rating for a movie.

    Returns
    -------
    dict with keys:
        predicted_rating  – float, clipped to [1, 10]
        confidence_range  – tuple (low, high)  ±0.5 approximation
        input_summary     – dict echoing inputs
    """
    model, encoders, scaler, feat_cols = load_artifacts()

    # --- Parse primary genre
    primary_genre = genre.split(",")[0].strip() if genre else "Unknown"
    genre_count   = len(genre.split(",")) if genre else 1

    # --- Feature vector
    log_votes = np.log1p(float(votes))
    movie_age = 2025 - int(year)

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

    # genre one-hot columns
    top_genres = encoders.get("top_genres", [])
    for g in top_genres:
        col_name = f"genre_{g.replace(' ', '_')}"
        row[col_name] = 1 if primary_genre == g else 0

    # Build dataframe aligned to feat_cols (fill any missing with 0)
    df_input = pd.DataFrame([row]).reindex(columns=feat_cols, fill_value=0)
    X_scaled = scaler.transform(df_input.values)

    rating = float(model.predict(X_scaled)[0])
    rating = float(np.clip(rating, 1.0, 10.0))

    # Confidence range — rough ±RMSE/2 heuristic (load from results)
    try:
        results = pd.read_csv(os.path.join(MODEL_DIR, "model_results.csv"))
        rmse = float(results.iloc[0]["RMSE"])
    except Exception:
        rmse = 0.8  # fallback

    low  = round(max(1.0, rating - rmse / 2), 1)
    high = round(min(10.0, rating + rmse / 2), 1)

    return {
        "predicted_rating": round(rating, 1),
        "confidence_range": (low, high),
        "rmse": round(rmse, 3),
        "input_summary": {
            "Genre": genre, "Director": director,
            "Actor 1": actor1, "Actor 2": actor2, "Actor 3": actor3,
            "Duration (min)": duration, "Votes": votes, "Year": year,
        },
    }


if __name__ == "__main__":
    # Quick smoke-test
    result = predict_rating(
        genre="Drama, Romance",
        director="Raj Kumar Hirani",
        actor1="Aamir Khan",
        actor2="Kareena Kapoor",
        actor3="Boman Irani",
        duration=153,
        votes=50000,
        year=2009,
    )
    print(f"\n⭐  Predicted Rating: {result['predicted_rating']} / 10")
    print(f"   Confidence range : {result['confidence_range'][0]} – {result['confidence_range'][1]}")
