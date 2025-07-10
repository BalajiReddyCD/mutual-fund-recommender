# lstm_model.py
import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings

warnings.filterwarnings("ignore")

# =============================
# Utility Functions
# =============================
def load_data():
    return pd.read_csv("data/processed/preprocessed_mutual_funds.csv")

def load_scheme_codes():
    return pd.read_csv("data/processed/top5_scheme_summary.csv")["Scheme_Code"].unique().tolist()

def calculate_accuracy(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return 100 - mape

# =============================
# LSTM Trainer
# =============================
def train_lstm(df, scheme_code):
    df = df[df['Scheme_Code'] == scheme_code].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    series = df[['Date', 'NAV']].dropna().copy()
    series.set_index('Date', inplace=True)

    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)

    def create_dataset(data, look_back=5):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i+look_back, 0])
            y.append(data[i+look_back, 0])
        return np.array(X), np.array(y)

    look_back = 5
    X, y = create_dataset(series_scaled, look_back)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential([
        LSTM(50, input_shape=(look_back, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    checkpoint_path = f"outputs/models/best_lstm_{scheme_code}.h5"
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, verbose=0)
    ]

    print(f"Training LSTM for Scheme_Code {scheme_code}...")
    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_data=(X_test, y_test), callbacks=callbacks, verbose=0)

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(np.concatenate([y_pred_scaled, np.zeros((len(y_pred_scaled), 0))], axis=1))[:, 0]
    y_test_actual = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), 0))], axis=1))[:, 0]

    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)
    acc = calculate_accuracy(y_test_actual, y_pred)

    print("\n Evaluation Metrics:")
    print(f"MAE         : {mae:.4f}")
    print(f"RMSE        : {rmse:.4f}")
    print(f"R² Score    : {r2:.4f}")
    print(f"Accuracy(%) : {acc:.2f}")

    # Save interactive HTML plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", "models")
    os.makedirs(output_dir, exist_ok=True)

    test_dates = series.index[-len(y_test_actual):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=y_test_actual, name='Actual NAV'))
    fig.add_trace(go.Scatter(x=test_dates, y=y_pred, name='Predicted NAV', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=test_dates, y=y_pred - rmse, mode='lines', name='Lower CI'))
    fig.add_trace(go.Scatter(x=test_dates, y=y_pred + rmse, mode='lines', name='Upper CI', fill='tonexty', fillcolor='rgba(0,100,80,0.2)'))
    fig.update_layout(title=f"LSTM Forecast - Scheme {scheme_code}", xaxis_title="Date", yaxis_title="NAV")
    plot_path = os.path.join(output_dir, f"LSTM_plot_{scheme_code}_{timestamp}.html")
    fig.write_html(plot_path)
    print(f"Interactive plot saved → {plot_path}")

    # Save CSV
    pd.DataFrame({
        "Date": test_dates,
        "Actual_NAV": y_test_actual,
        "Predicted_NAV": y_pred,
        "Lower_CI": y_pred - rmse,
        "Upper_CI": y_pred + rmse
    }).to_csv(os.path.join(output_dir, f"LSTM_preds_{scheme_code}_{timestamp}.csv"), index=False)

    # Update Leaderboard
    leaderboard_path = os.path.join(output_dir, "model_leaderboard.csv")
    entry = pd.DataFrame([{
        "Model": "LSTM",
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
    print(f"Leaderboard updated → {leaderboard_path}")

# =============================
# CLI Entrypoint
# =============================
if __name__ == "__main__":
    df = load_data()
    scheme_codes = load_scheme_codes()
    for code in scheme_codes:
        train_lstm(df, code)
