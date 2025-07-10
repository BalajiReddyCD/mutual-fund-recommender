# prophet_model.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objs as go
import plotly.io as pio

pio.renderers.default = "notebook_connected"

def load_data():
    return pd.read_csv("data/processed/preprocessed_mutual_funds.csv")

def load_top_scheme_codes():
    df = pd.read_csv("data/processed/top5_scheme_summary.csv")
    return df['Scheme_Code'].dropna().unique().tolist()

def calculate_accuracy(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return 100 - mape

def train_prophet(df, scheme_code, output_dir="outputs/models"):
    df = df[df['Scheme_Code'] == scheme_code].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    series = df.groupby('Date')['NAV'].mean().reset_index()
    series.rename(columns={'Date': 'ds', 'NAV': 'y'}, inplace=True)

    train_size = int(len(series) * 0.8)
    train, test = series.iloc[:train_size], series.iloc[train_size:]

    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)

    forecast_eval = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')
    test.set_index('ds', inplace=True)
    aligned = forecast_eval.join(test, how='inner')

    mae = mean_absolute_error(aligned['y'], aligned['yhat'])
    rmse = np.sqrt(mean_squared_error(aligned['y'], aligned['yhat']))
    r2 = r2_score(aligned['y'], aligned['yhat'])
    acc = calculate_accuracy(aligned['y'], aligned['yhat'])

    print(f"\nScheme_Code {scheme_code} → Accuracy: {acc:.2f}%, RMSE: {rmse:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preds_df = aligned.reset_index()
    preds_df.columns = ["Date", "Predicted_NAV", "Lower_CI", "Upper_CI", "Actual_NAV"]
    os.makedirs(output_dir, exist_ok=True)
    preds_df.to_csv(os.path.join(output_dir, f"Prophet_preds_{scheme_code}_{timestamp}.csv"), index=False)

    leaderboard_path = os.path.join(output_dir, "model_leaderboard.csv")
    entry = pd.DataFrame([{
        "Model": "Prophet",
        "Scheme_Code": scheme_code,
        "Timestamp": timestamp,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Accuracy (%)": round(acc, 2)
    }])
    if os.path.exists(leaderboard_path):
        current = pd.read_csv(leaderboard_path)
        updated = pd.concat([current, entry], ignore_index=True)
    else:
        updated = entry
    updated.to_csv(leaderboard_path, index=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=preds_df["Date"], y=preds_df["Actual_NAV"], mode='lines', name='Actual NAV'))
    fig.add_trace(go.Scatter(x=preds_df["Date"], y=preds_df["Predicted_NAV"], mode='lines', name='Predicted NAV'))
    fig.add_trace(go.Scatter(x=preds_df["Date"], y=preds_df["Lower_CI"], mode='lines', name='Lower CI', line=dict(dash='dot'), opacity=0.3))
    fig.add_trace(go.Scatter(x=preds_df["Date"], y=preds_df["Upper_CI"], mode='lines', name='Upper CI', line=dict(dash='dot'), opacity=0.3))

    fig.update_layout(title=f"Prophet Forecast: Scheme {scheme_code}",
                      xaxis_title='Date', yaxis_title='NAV',
                      template='plotly_white')

    html_path = os.path.join(output_dir, f"Prophet_plot_{scheme_code}_{timestamp}.html")
    fig.write_html(html_path)
    print(f" Interactive plot saved → {html_path}")

def run_all():
    df = load_data()
    scheme_codes = load_top_scheme_codes()
    for code in scheme_codes:
        train_prophet(df, code)

if __name__ == "__main__":
    run_all()
