# main.py – Full Project Runner: LSTM, ARIMA, Prophet, Dashboard

import os
import pandas as pd
from lstm_model import train_lstm, load_data, load_scheme_codes
from arima_model import train_optimized_arima
from prophet_model import train_prophet

def run_all_models():
    """Run LSTM, ARIMA, and Prophet models on top 5 scheme codes"""
    df = load_data()
    scheme_codes = load_scheme_codes()
    output_dir = "outputs/models"

    # ARIMA configuration
    p_values = [0, 1, 2, 4, 6]
    d_values = range(0, 3)
    q_values = range(0, 3)

    for code in scheme_codes:
        print(f"\n Running models for Scheme_Code: {code}")

        print("→ LSTM:")
        train_lstm(df, code)

        print("\n Training ARIMA model...")
        train_optimized_arima(df, code, output_dir, p_values, d_values, q_values)

        print("→ Prophet:")
        train_prophet(df, code, output_dir)

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("\n Launching Streamlit Dashboard...")
    os.system("streamlit run src/app.py")

if __name__ == "__main__":
    print(" Starting Mutual Fund Recommender System Pipeline")
    run_all_models()
    launch_dashboard()
