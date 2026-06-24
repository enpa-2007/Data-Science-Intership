import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sales.csv")

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def get_data_info(df):
    info = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "stats": df.describe().to_dict(),
    }
    return info

def preprocess_data(df, test_size=0.2, random_state=42):
    df_clean = df.dropna()
    X = df_clean[["TV", "Radio", "Newspaper"]]
    y = df_clean["Sales"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def classify_sales(value):
    if value < 10:
        return "Low Sales", "🔴"
    elif value < 18:
        return "Medium Sales", "🟡"
    else:
        return "High Sales", "🟢"
