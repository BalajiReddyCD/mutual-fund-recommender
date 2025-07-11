# app/utils/loader.py

import pandas as pd
import os

DATA_DIR = "C:/Users/BALA/OneDrive - University of Hertfordshire/Desktop/final project/app/data/results"

def load_leaderboard():
    path = os.path.join(DATA_DIR, "final_model_comparison.csv")
    return pd.read_csv(path)

def load_recommendations():
    path = os.path.join(DATA_DIR, "model_recommendations.csv")
    return pd.read_csv(path)
