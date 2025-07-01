# prophet_model.py

import os
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =============================
# Utility Functions
# =============================
def load_data():
    return pd.read_csv("data/processed/preprocessed_mutual_funds.csv")

def select_scheme_code(df, min_records=100):
    counts = df['Scheme_Code'].value_counts()
    valid = counts[counts > min_records]
    if valid.empty:
        raise ValueError("No Scheme_Code with enough data.")
    code = valid.index[0]
    print(f" Using Scheme_Code: {code} with {valid.iloc[0]} entries")
    return code

# =============================
# Prophet Trainer
# =============================
def train_prophet(df, scheme_code):
    df = df[df['Scheme_Code'] == scheme_code].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")

    prophet_df = df[['Date', 'NAV']].rename(columns={'Date': 'ds', 'NAV': 'y'})

    train_size = int(len(prophet_df) * 0.8)
    train = prophet_df[:train_size]
    test = prophet_df[train_size:]

    model = Prophet(
        yearly_seasonality=True,
        changepoint_prior_scale=0.1,
        seasonality_mode='multiplicative'
    )
    model.fit(train)

    future = test[['ds']].copy()
    forecast = model.predict(future)

    # Metrics
    mae = mean_absolute_error(test['y'], forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(test['y'], forecast['yhat']))
    r2 = r2_score(test['y'], forecast['yhat'])

    print("\n Evaluation Metrics:")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(test['ds'], test['y'], label='Actual NAV')
    plt.plot(test['ds'], forecast['yhat'], label='Predicted NAV (Prophet)', linestyle='--')
    plt.fill_between(test['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2, label='Confidence Interval')
    plt.title(f"Prophet Forecast - Scheme {scheme_code}")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
    os.makedirs(output_dir, exist_ok=True)

    pred_df = pd.DataFrame({
        "Date": test['ds'],
        "Actual_NAV": test['y'],
        "Predicted_NAV": forecast['yhat']
    })
    pred_file = f"Prophet_preds_{scheme_code}_{timestamp}.csv"
    pred_path = os.path.join(output_dir, pred_file)
    pred_df.to_csv(pred_path, index=False)
    print(f" Predictions saved to: {pred_path}")

    # Save to leaderboard
    leaderboard_path = os.path.join(output_dir, "model_leaderboard.csv")
    leaderboard_entry = pd.DataFrame([{
        "Model": "Prophet",
        "Scheme_Code": scheme_code,
        "Timestamp": timestamp,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }])
    if os.path.exists(leaderboard_path):
        existing = pd.read_csv(leaderboard_path)
        updated = pd.concat([existing, leaderboard_entry], ignore_index=True)
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
    train_prophet(df, scheme)

# Reusable API Method

def train_and_save_prophet():
    df = load_data()
    scheme = select_scheme_code(df)
    train_prophet(df, scheme)
