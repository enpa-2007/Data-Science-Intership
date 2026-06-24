"""
train_model.py
--------------
Trains multiple ML models on the Iris dataset, evaluates them,
and persists the best-performing model as iris_model.pkl.
Run this script once before launching the Streamlit app, or let
the app call it automatically on first launch.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")


# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "iris_model.pkl")
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "IRIS.csv")
RANDOM_SEED = 42
TEST_SIZE   = 0.20


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load the Iris dataset from CSV or sklearn fallback."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=[
            "sepal_length", "sepal_width", "petal_length", "petal_width"
        ])
        df["species"] = [iris.target_names[t] for t in iris.target]
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return df


# ─── Training Pipeline ────────────────────────────────────────────────────────
def train_all_models(df: pd.DataFrame) -> dict:
    """
    Trains Logistic Regression, Decision Tree, Random Forest, KNN, and SVM.
    Returns a dict with all artefacts needed by the Streamlit pages.
    """
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    target_col   = "species"

    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df[target_col].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Feature scaling (fit on train, transform both)
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Model definitions
    model_defs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
    }

    results   = {}
    trained   = {}

    for name, clf in model_defs.items():
        clf.fit(X_train_sc, y_train)
        y_pred = clf.predict(X_test_sc)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)
        cv   = cross_val_score(clf, scaler.transform(X), y, cv=5).mean()

        results[name] = {
            "accuracy":  round(acc  * 100, 2),
            "precision": round(prec * 100, 2),
            "recall":    round(rec  * 100, 2),
            "f1_score":  round(f1   * 100, 2),
            "cv_score":  round(cv   * 100, 2),
            "confusion_matrix": cm,
            "classification_report": classification_report(
                y_test, y_pred, target_names=le.classes_
            ),
        }
        trained[name] = clf

    # Best model by accuracy
    best_name = max(results, key=lambda n: results[n]["accuracy"])

    payload = {
        "models":        trained,
        "scaler":        scaler,
        "label_encoder": le,
        "results":       results,
        "best_model":    best_name,
        "feature_cols":  feature_cols,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_sc": X_train_sc, "X_test_sc": X_test_sc,
    }

    joblib.dump(payload, MODEL_PATH)
    print(f"[train_model] Models saved → {MODEL_PATH}")
    print(f"[train_model] Best model  : {best_name} "
          f"({results[best_name]['accuracy']}% accuracy)")
    return payload


# ─── Public Helper ─────────────────────────────────────────────────────────────
def get_trained_models(force_retrain: bool = False) -> dict:
    """Load persisted models or train from scratch."""
    if not force_retrain and os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    df = load_data()
    return train_all_models(df)


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    get_trained_models(force_retrain=True)
