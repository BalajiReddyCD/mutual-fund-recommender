# app/utils/loader.py

import pandas as pd
import os

# Define a base directory constant
BASE_DIR = "C:/Users/BALA/OneDrive - University of Hertfordshire/Desktop/mutual-fund-recommender/app/data/results"

def load_leaderboard():
    path = os.path.join(BASE_DIR, "final_model_comparison.csv")
    return pd.read_csv(path)

def load_recommendations():
    path = os.path.join(BASE_DIR, "model_recommendations.csv")
    return pd.read_csv(path)

def load_forecast_csv(model_name, scheme_code):
    path = os.path.join(BASE_DIR, f"{model_name}_predictions", f"{scheme_code}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_model_leaderboard(model_name):
    path = os.path.join(BASE_DIR, f"{model_name.lower()}_leaderboard.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_fund_metadata():
    path = os.path.join(BASE_DIR, "fund_metadata.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()
