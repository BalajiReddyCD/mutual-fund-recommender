# main.py – Full Project Runner: LSTM, ARIMA, Prophet, Clustering + Dashboard

import os
import pandas as pd

from lstm_model import train_and_save_model
from arima_model import train_and_save_arima
from prophet_model import train_and_save_prophet
from clustering_model import run_kmeans_clustering, plot_elbow_curve, plot_cluster_summary


def load_data():
    """Load preprocessed mutual fund data"""
    df = pd.read_csv("data/processed/preprocessed_mutual_funds.csv")
    return df


def get_best_scheme():
    """Automatically select a valid Scheme_Code with enough data"""
    df = load_data()
    scheme_counts = df['Scheme_Code'].value_counts()
    best = scheme_counts[scheme_counts > 100].index[0]
    print(f"\n Selected Scheme_Code: {best} with {scheme_counts[best]} entries")
    return best


def run_all_models(scheme_code):
    """Run LSTM, ARIMA, Prophet, and Clustering models"""
    df = load_data()

    print("\n🔁 Training LSTM model...")
    train_and_save_model(scheme_code)  # assuming your LSTM takes scheme_code

    print("\n🔁 Training ARIMA model...")
    train_and_save_arima()  # assuming ARIMA handles its own data loading

    print("\n🔁 Training Prophet model...")
    train_and_save_prophet()  # same assumption

    # print("\n🔍 Running K-Means Clustering for fund profiles...")
    # clustered_df, kmeans_model = run_kmeans_clustering()

    # print("\n📊 Cluster Summary:")
    # plot_cluster_summary(clustered_df)


def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("\n🚀 Launching Streamlit Dashboard...")
    os.system("streamlit run src/app.py")


if __name__ == "__main__":
    print("Starting Mutual Fund Recommender Pipeline")

    scheme_code = get_best_scheme()

    run_all_models(scheme_code)

    launch_dashboard()
