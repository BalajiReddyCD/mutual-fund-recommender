# main.py – Full Project Runner: LSTM, ARIMA, Prophet + Dashboard

import os
import pandas as pd

from lstm_model import train_and_save_model
from arima_model import train_and_save_arima
from prophet_model import train_and_save_prophet  # Make sure this exists

# ✅ Define load_data() here, not in lstm_utils
def load_data():
    df = pd.read_csv("data/processed/preprocessed_mutual_funds.csv")
    return df

def get_best_scheme():
    df = load_data()
    scheme_counts = df['Scheme_Code'].value_counts()
    best = scheme_counts[scheme_counts > 100].index[0]
    print(f"\n🏁 Selected Scheme_Code: {best} with {scheme_counts[best]} entries")
    return best

def run_all_models(scheme_code):
    df = load_data()

    print("\n🔁 Training LSTM model...")
    train_and_save_model(scheme_code)  # assuming your LSTM function doesn't need df

    print("\n🔁 Training ARIMA model...")
    train_and_save_arima()  # assumes arima_model handles loading internally

    print("\n🔁 Training Prophet model...")
    train_and_save_prophet()  # same assumption here

def launch_dashboard():
    print("\n🚀 Launching Streamlit Dashboard...")
    os.system("streamlit run src/app.py")

if __name__ == "__main__":
    print("🏗️ Starting Mutual Fund Recommender Pipeline")

    scheme_code = get_best_scheme()

    run_all_models(scheme_code)

    launch_dashboard()
