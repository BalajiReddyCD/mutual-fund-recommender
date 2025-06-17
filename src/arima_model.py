import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def run_arima_forecast(df, scheme_code, output_path="outputs/arima_forecast.png"):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.sort_values(['Scheme_Code', 'Date'], inplace=True)

    scheme_df = df[df['Scheme_Code'] == scheme_code].copy()
    scheme_df.reset_index(drop=True, inplace=True)

    nav_series = scheme_df['NAV'].values
    train_size = int(len(nav_series) * 0.8)
    train_nav, test_nav = nav_series[:train_size], nav_series[train_size:]

    model = auto_arima(
        train_nav,
        seasonal=False,
        stepwise=True,
        trace=True,
        suppress_warnings=True,
        error_action="ignore"
    )

    forecast = model.predict(n_periods=len(test_nav))

    mae = mean_absolute_error(test_nav, forecast)
    rmse = np.sqrt(mean_squared_error(test_nav, forecast))

    print(f"\nARIMA Summary:\n{model.summary()}")
    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    # Create outputs folder if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Plot and save
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(train_nav)), train_nav, label="Train")
    plt.plot(range(len(train_nav), len(nav_series)), test_nav, label="Test", color='orange')
    plt.plot(range(len(train_nav), len(nav_series)), forecast, label="ARIMA Forecast", linestyle='--')
    plt.title(f"ARIMA NAV Forecast - Scheme {scheme_code}")
    plt.xlabel("Time Index")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return forecast, test_nav, mae, rmse
