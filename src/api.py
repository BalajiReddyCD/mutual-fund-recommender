# api.py – Backend Utility Module for Mutual Fund Recommender

import os
import pandas as pd
from datetime import datetime

# Define output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")


def load_leaderboard():
    leaderboard_path = os.path.join(OUTPUT_DIR, "model_leaderboard.csv")
    if os.path.exists(leaderboard_path):
        return pd.read_csv(leaderboard_path)
    return None


def get_scheme_codes(df):
    return sorted(df["Scheme_Code"].astype(str).unique().tolist())


def get_best_scheme_code(df):
    return df.sort_values("RMSE").iloc[0]["Scheme_Code"]


def get_model_list(df):
    return sorted(df["Model"].unique().tolist())


def filter_leaderboard(df, scheme_code=None, model_names=None):
    filtered = df.copy()
    if scheme_code and scheme_code != "All":
        filtered = filtered[filtered["Scheme_Code"].astype(str) == scheme_code]
    if model_names:
        filtered = filtered[filtered["Model"].isin(model_names)]
    return filtered


def get_model_prediction_files():
    if not os.path.exists(OUTPUT_DIR):
        return []
    return [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv") and "preds" in f]


def load_prediction_file(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_predictions(scheme_code, model_name):
    filename = f"{scheme_code}_{model_name}_preds.csv"
    return load_prediction_file(filename)


def get_monthly_nav_summary(df):
    if 'Date' not in df or 'Actual_NAV' not in df or 'Predicted_NAV' not in df:
        return pd.DataFrame()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df['Month'] = df['Date'].dt.to_period('M')
    return df.groupby('Month')[['Actual_NAV', 'Predicted_NAV']].mean().reset_index()


def calculate_metrics(df):
    mae = round((df["Actual_NAV"] - df["Predicted_NAV"]).abs().mean(), 4)
    rmse = round(((df["Actual_NAV"] - df["Predicted_NAV"])**2).mean()**0.5, 4)
    r2 = round(1 - (((df["Actual_NAV"] - df["Predicted_NAV"])**2).sum() /
                  ((df["Actual_NAV"] - df["Actual_NAV"].mean())**2).sum()), 4)
    return mae, rmse, r2
