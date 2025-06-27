import os
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
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


def train_prophet_model(df, scheme_code):
    df_selected = df[df['Scheme_Code'] == scheme_code].copy()
    df_selected['Date'] = pd.to_datetime(df_selected['Date'])
    df_selected = df_selected.sort_values("Date")

    prophet_df = df_selected[['Date', 'NAV']].rename(columns={'Date': 'ds', 'NAV': 'y'})

    train_size = int(len(prophet_df) * 0.8)
    train = prophet_df[:train_size]
    test = prophet_df[train_size:]

    model = Prophet(daily_seasonality=False, yearly_seasonality=True)
    model.fit(train)

    future = test[['ds']].copy()
    forecast = model.predict(future)

    # Evaluation metrics
    mae = mean_absolute_error(test['y'], forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(test['y'], forecast['yhat']))
    r2 = r2_score(test['y'], forecast['yhat'])

    print("\nEvaluation Metrics:")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Plot actual vs predicted NAV
    plt.figure(figsize=(10, 5))
    plt.plot(test['ds'], test['y'], label='Actual NAV')
    plt.plot(test['ds'], forecast['yhat'], label='Predicted NAV (Prophet)', linestyle='--')
    plt.title(f"Prophet NAV Forecast - Scheme {scheme_code}")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # Save predictions and metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
    os.makedirs(output_dir, exist_ok=True)

    predictions_df = pd.DataFrame({
        "Date": test['ds'],
        "Actual_NAV": test['y'],
        "Predicted_NAV": forecast['yhat']
    })
    predictions_filename = f"Prophet_preds_{scheme_code}_{timestamp}.csv"
    predictions_path = os.path.join(output_dir, predictions_filename)
    predictions_df.to_csv(predictions_path, index=False)
    print(f"📁 Predictions saved to: {predictions_path}")

    # Leaderboard logging
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
        existing_df = pd.read_csv(leaderboard_path)
        updated_df = pd.concat([existing_df, leaderboard_entry], ignore_index=True)
    else:
        updated_df = leaderboard_entry

    updated_df.to_csv(leaderboard_path, index=False)
    print(f"🏁 Leaderboard updated: {leaderboard_path}")


if __name__ == "__main__":
    df = load_data()
    scheme_code = select_scheme_code(df)
    train_prophet_model(df, scheme_code)

def train_and_save_prophet():
    df = load_data()
    scheme_code = select_scheme_code(df)
    train_prophet_model(df, scheme_code)
