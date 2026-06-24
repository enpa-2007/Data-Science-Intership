import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2 Score": round(r2, 4),
        "CV R2 Mean": round(cv_scores.mean(), 4),
        "CV R2 Std": round(cv_scores.std(), 4),
        "Predictions": y_pred,
    }

def train_all_models(X_train, X_test, y_train, y_test):
    models = get_models()
    results = {}
    trained_models = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        results[name] = metrics
        trained_models[name] = model
    best_name = max(results, key=lambda k: results[k]["R2 Score"])
    best_model = trained_models[best_name]
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump({"model": best_model, "name": best_name}, f)
    return results, trained_models, best_name, best_model

def load_best_model():
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["model"], data["name"]
    return None, None

def get_feature_importance(model, model_name, feature_names):
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_)
        total = coefs.sum()
        return dict(zip(feature_names, coefs / total))
    return {}
