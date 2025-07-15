# LSTM for NAV Forecasting
# LSTM (Long Short-Term Memory) is a deep learning model perfect for:
# Capturing complex patterns in sequences
# Handling longer dependencies in time series
# LSTM Forecasting Plan
# Prepare NAV time series for each Scheme_Code

# Normalize NAV values
# Create sliding window sequences
# Build & train LSTM model
# Forecast next n NAV values
# Plot using Plotly
# Evaluate (MAE, RMSE, R²)
# Store in leaderboard

# LSTM +  Plotly Forecasting (for Top 3 Funds)
# Assumptions:
# Forecast next 30 days
# Use 50-day lookback window

# Process top 3 Scheme_Codes: [100051, 100047, 100048]

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import os

import warnings

df = pd.read_csv(r"C:/Users/BALA/OneDrive - University of Hertfordshire/Desktop/mutual-fund-recommender/app/data/processed/preprocessed_mutual_funds.csv")


warnings.filterwarnings("ignore", category=FutureWarning)

# Parameters
top_scheme_codes = [100051, 100047, 100048]
lookback = 50
forecast_days = 30
lstm_results = []

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

# Loop through each scheme
for code in top_scheme_codes:
    fund = df[df['Scheme_Code'] == code][['Date', 'NAV']].sort_values('Date')
    fund['Date'] = pd.to_datetime(fund['Date'])
    
    if len(fund) < (lookback + forecast_days + 10):
        print(f"Skipping {code} due to insufficient data.")
        continue

    fund.set_index('Date', inplace=True)
    scaler = MinMaxScaler()
    scaled_nav = scaler.fit_transform(fund[['NAV']])

    # Split train/test
    train_size = int(len(scaled_nav) * 0.8)
    train, test = scaled_nav[:train_size], scaled_nav[train_size - lookback:]

    # Create sequences
    X_train, y_train = create_sequences(train, lookback)
    X_test, y_test = create_sequences(test, lookback)

    # Reshape for LSTM input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=25, batch_size=16, verbose=0)

    # Predict on test
    preds = model.predict(X_test).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

    # Forecast future NAV
    last_window = scaled_nav[-lookback:]
    future_preds = []
    input_seq = last_window.reshape(1, lookback, 1)
    for _ in range(forecast_days):
        next_pred = model.predict(input_seq, verbose=0)[0][0]
        future_preds.append(next_pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[next_pred]]], axis=1)

    future_dates = pd.date_range(fund.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
    forecast_nav = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

    # Evaluation
    mae = mean_absolute_error(y_test_inv, preds_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
    r2 = r2_score(y_test_inv, preds_inv)

    lstm_results.append({
        'Scheme_Code': code,
        'Fund_Name': df[df['Scheme_Code'] == code]['Scheme_Name'].iloc[0],
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2_Score': round(r2, 3)
    })

    # Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fund.index, y=fund['NAV'], mode='lines', name='Historical NAV'))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_nav, mode='lines+markers', name='LSTM Forecast'))
    fig.update_layout(
        title=f"LSTM NAV Forecast – Scheme Code {code}",
        xaxis_title="Date", yaxis_title="NAV",
        template="plotly_dark", height=500
    )
    fig.show()

# Show leaderboard
leaderboard_lstm = pd.DataFrame(lstm_results).sort_values(by='RMSE')
print("LSTM Forecast Leaderboard:")
print(leaderboard_lstm)

os.makedirs('data/results', exist_ok=True)
leaderboard_lstm.to_csv('C:/Users/BALA/OneDrive - University of Hertfordshire/Desktop/mutual-fund-recommender/app/data/results/lstm_leaderboard.csv', index=False)
print("LSTM leaderboard saved to data/results/lstm_leaderboard.csv")
