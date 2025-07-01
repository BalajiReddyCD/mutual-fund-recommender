# arima_model.py 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
import warnings

warnings.filterwarnings("ignore")

# =============================
# Utility Functions
# =============================
def load_data():
    return pd.read_csv("data/processed/preprocessed_mutual_funds.csv")

def select_scheme_code(df, min_records=100):
    scheme_counts = df['Scheme_Code'].value_counts()
    valid = scheme_counts[scheme_counts > min_records]
    if valid.empty:
        raise ValueError("No Scheme_Code with enough data points.")
    code = valid.index[0]
    print(f" Using Scheme_Code: {code} with {valid.iloc[0]} entries")
    return code

# =============================
# ARIMA Trainer
# =============================
def is_stationary(series, alpha=0.05):
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")
    return result[1] < alpha

def train_arima(df, scheme_code):
    df = df[df['Scheme_Code'] == scheme_code].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    # Monthly average NAV
    series = df['NAV'].resample('M').mean().dropna()
    if not is_stationary(series):
        print(" Series is not stationary. Differencing will be applied by auto_arima.")

    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
    print(f"\n Selected ARIMA order: {model.order}")

    forecast = model.predict(n_periods=len(test))
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    r2 = r2_score(test, forecast)

    print("\n\U0001F4CA Evaluation Metrics:")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(test.index, test, label='Actual NAV')
    plt.plot(test.index, forecast, label='Predicted NAV', linestyle='--')
    plt.title(f"ARIMA Forecast - Scheme {scheme_code}")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save outputs
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    predictions_df = pd.DataFrame({
        "Date": test.index,
        "Actual_NAV": test.values,
        "Predicted_NAV": forecast
    })
    pred_file = f"ARIMA_preds_{scheme_code}_{timestamp}.csv"
    pred_path = os.path.join(output_dir, pred_file)
    predictions_df.to_csv(pred_path, index=False)
    print(f" Predictions saved to: {pred_path}")

    # Leaderboard
    leaderboard_path = os.path.join(output_dir, "model_leaderboard.csv")
    leaderboard_entry = pd.DataFrame([{
        "Model": "ARIMA",
        "Scheme_Code": scheme_code,
        "Timestamp": timestamp,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }])
    if os.path.exists(leaderboard_path):
        current = pd.read_csv(leaderboard_path)
        updated = pd.concat([current, leaderboard_entry], ignore_index=True)
    else:
        updated = leaderboard_entry
    updated.to_csv(leaderboard_path, index=False)
    print(f" Leaderboard updated: {leaderboard_path}")

# =============================
# CLI Entrypoint
# =============================
if __name__ == "__main__":
    df = load_data()
    scheme = select_scheme_code(df)
    train_arima(df, scheme)

# Reusable API Method

def train_and_save_arima():
    df = load_data()
    scheme = select_scheme_code(df)
    train_arima(df, scheme)
