import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

def load_data():
    df = pd.read_csv("data/processed/preprocessed_mutual_funds.csv")
    return df

def select_scheme_code(df):
    scheme_counts = df['Scheme_Code'].value_counts()
    valid_schemes = scheme_counts[scheme_counts > 100]
    if valid_schemes.empty:
        raise ValueError("No Scheme_Code with more than 100 data points.")
    scheme_code = valid_schemes.index[0]
    print(f"✅ Using Scheme_Code: {scheme_code} with {valid_schemes.iloc[0]} entries")
    return scheme_code

def train_arima_model(df, scheme_code):
    df_selected = df[df['Scheme_Code'] == scheme_code].copy()
    df_selected['Date'] = pd.to_datetime(df_selected['Date'])
    df_selected = df_selected.sort_values('Date')
    df_selected.set_index('Date', inplace=True)

    # Resample monthly average NAV (to smooth and follow paper-III guidance)
    data = df_selected['NAV'].resample('M').mean().dropna()
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    # Iterative forecasting (fixes the flat line bug)
    forecast = []
    history = list(train)

    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))  # You can experiment with (2,1,2) or (1,1,1) for smoother fit
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        forecast.append(yhat)
        history.append(test[t])

    # Evaluation
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    r2 = r2_score(test, forecast)

    print("\n📊 Evaluation Metrics:")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(test.index, test, label='Actual NAV')
    plt.plot(test.index, forecast, label='Predicted NAV', linestyle='--')
    plt.title(f"ARIMA NAV Prediction - Scheme {scheme_code}")
    plt.xlabel("Date")
    plt.ylabel("Monthly Average NAV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
    os.makedirs(output_dir, exist_ok=True)

    # Export predictions
    predictions_df = pd.DataFrame({
        "Date": test.index,
        "Actual_NAV": test.values,
        "Predicted_NAV": forecast
    })
    predictions_filename = f"ARIMA_preds_{scheme_code}_{timestamp}.csv"
    predictions_path = os.path.join(output_dir, predictions_filename)
    predictions_df.to_csv(predictions_path, index=False)
    print(f"📁 Predictions saved to: {predictions_path}")

    # Update leaderboard
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
        existing_df = pd.read_csv(leaderboard_path)
        updated_df = pd.concat([existing_df, leaderboard_entry], ignore_index=True)
    else:
        updated_df = leaderboard_entry

    updated_df.to_csv(leaderboard_path, index=False)
    print(f"🏁 Leaderboard updated: {leaderboard_path}")

if __name__ == "__main__":
    df = load_data()
    scheme_code = select_scheme_code(df)
    train_arima_model(df, scheme_code)

def train_and_save_arima():
    df = load_data()
    scheme_code = select_scheme_code(df)
    train_arima_model(df, scheme_code)
