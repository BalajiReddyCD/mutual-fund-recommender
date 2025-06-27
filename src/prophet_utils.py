import os
import pandas as pd
import numpy as np
from datetime import datetime
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def prepare_prophet_df(df, scheme_code):
    df_selected = df[df['Scheme_Code'] == scheme_code].copy()
    df_selected['Date'] = pd.to_datetime(df_selected['Date'])
    df_selected = df_selected.sort_values('Date')
    prophet_df = df_selected[['Date', 'NAV']].rename(columns={'Date': 'ds', 'NAV': 'y'})
    return prophet_df


def split_train_test(prophet_df, split_ratio=0.8):
    split_index = int(len(prophet_df) * split_ratio)
    train = prophet_df.iloc[:split_index]
    test = prophet_df.iloc[split_index:]
    return train, test


def evaluate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return mae, rmse, r2


def save_predictions(test, forecast, scheme_code, timestamp):
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
    os.makedirs(output_dir, exist_ok=True)
    merged = test[['ds', 'y']].copy()
    merged['Predicted_NAV'] = forecast['yhat'].values
    predictions_filename = f"Prophet_preds_{scheme_code}_{timestamp}.csv"
    predictions_path = os.path.join(output_dir, predictions_filename)
    merged.to_csv(predictions_path, index=False)
    print(f"📁 Predictions saved to: {predictions_path}")
    return predictions_path


def log_to_leaderboard(model_name, scheme_code, timestamp, mae, rmse, r2):
    leaderboard_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "models", "model_leaderboard.csv")
    leaderboard_entry = pd.DataFrame([{
        "Model": model_name,
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
