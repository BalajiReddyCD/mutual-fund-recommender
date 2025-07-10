# arima_model.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

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

def train_optimized_arima(df, scheme_code, output_dir, p_values, d_values, q_values):
    df = df[df['Scheme_Code'] == scheme_code].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()

    series = np.log(df['NAV'].resample('M').mean().dropna())
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    best_model, best_metrics, best_order = None, None, None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(train, order=(p, d, q)).fit()
                    forecast_result = model.get_forecast(steps=len(test))
                    forecast = np.exp(forecast_result.predicted_mean)
                    conf_int = np.exp(forecast_result.conf_int(alpha=0.05).values)

                    actual = np.exp(test)
                    mae = mean_absolute_error(actual, forecast)
                    rmse = np.sqrt(mean_squared_error(actual, forecast))
                    r2 = r2_score(actual, forecast)
                    acc = calculate_accuracy(actual.values, forecast)
                    aic = model.aic

                    if best_metrics is None or r2 > best_metrics["R2"]:
                        best_model = model
                        best_order = (p, d, q)
                        best_metrics = {
                            "Model": "ARIMA",
                            "Scheme_Code": scheme_code,
                            "Order": best_order,
                            "AIC": round(aic, 2),
                            "MAE": round(mae, 4),
                            "RMSE": round(rmse, 4),
                            "R2": round(r2, 4),
                            "Accuracy (%)": round(acc, 2),
                            "Forecast": forecast,
                            "Conf_Int": conf_int
                        }
                except Exception as e:
                    print(f" ARIMA({p},{d},{q}) failed: {e}")

    if best_metrics:
        print(f"\n Best ARIMA(p,d,q) for Scheme {scheme_code} → {best_order} | R²={best_metrics['R2']:.4f}, AIC={best_metrics['AIC']}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_df = pd.DataFrame({
            "Date": test.index,
            "Actual_NAV": np.exp(test.values),
            "Predicted_NAV": best_metrics["Forecast"],
            "Lower_CI": best_metrics["Conf_Int"][:, 0],
            "Upper_CI": best_metrics["Conf_Int"][:, 1]
        })

        os.makedirs(output_dir, exist_ok=True)
        result_csv = os.path.join(output_dir, f"ARIMA_preds_{scheme_code}_{timestamp}.csv")
        result_df.to_csv(result_csv, index=False)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test.index, y=result_df["Actual_NAV"], name='Actual NAV'))
        fig.add_trace(go.Scatter(x=test.index, y=result_df["Predicted_NAV"], name='Predicted NAV', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=test.index, y=result_df["Lower_CI"], mode='lines', name='Lower CI'))
        fig.add_trace(go.Scatter(x=test.index, y=result_df["Upper_CI"], mode='lines', name='Upper CI', fill='tonexty', fillcolor='rgba(0,100,80,0.2)'))
        fig.update_layout(title=f"Optimized ARIMA Forecast - Scheme {scheme_code} (Best {best_order})", xaxis_title="Date", yaxis_title="NAV")
        plot_path = os.path.join(output_dir, f"ARIMA_plot_{scheme_code}_{timestamp}.html")
        fig.write_html(plot_path)

        metrics = {
            "Model": "ARIMA",
            "Scheme_Code": scheme_code,
            "Timestamp": timestamp,
            "MAE": best_metrics["MAE"],
            "RMSE": best_metrics["RMSE"],
            "R2": best_metrics["R2"],
            "Accuracy (%)": best_metrics["Accuracy (%)"],
            "AIC": best_metrics["AIC"]
        }

        leaderboard_path = os.path.join(output_dir, "model_leaderboard.csv")
        entry = pd.DataFrame([metrics])
        if os.path.exists(leaderboard_path):
            existing = pd.read_csv(leaderboard_path)
            updated = pd.concat([existing, entry], ignore_index=True)
        else:
            updated = entry
        updated.to_csv(leaderboard_path, index=False)
        print(f"Leaderboard updated → {leaderboard_path}")

        return metrics

    return None

# adf_stationarity_check.py

# Function to perform ADF test
def run_adf_test(series):
    result = adfuller(series.dropna())
    return result[0], result[1]  # ADF statistic, p-value

# Load data before using it
df = load_data()

# Collect results
adf_results = []

for code in load_top_scheme_codes():
    temp_df = df[df['Scheme_Code'] == code].copy()
    temp_df['Date'] = pd.to_datetime(temp_df['Date'])
    temp_df.set_index('Date', inplace=True)
    
    monthly_nav = temp_df['NAV'].resample('M').mean().dropna()
    adf_stat, p_val = run_adf_test(monthly_nav)
    
    adf_results.append({
        "Scheme_Code": code,
        "ADF Statistic": round(adf_stat, 4),
        "p-value": round(p_val, 4),
        "Stationary": "Yes" if p_val < 0.05 else "No"
    })

# Save as table
adf_df = pd.DataFrame(adf_results)
output_path = "outputs/models/adf_stationarity_summary.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
adf_df.to_csv(output_path, index=False)

print(" ADF Stationarity check completed.")
print(adf_df)

import plotly.graph_objects as go

# Create a bar plot of ADF statistics and p-values
fig = go.Figure()

# ADF Statistic bars
fig.add_trace(go.Bar(
    x=adf_df["Scheme_Code"],
    y=adf_df["ADF Statistic"],
    name="ADF Statistic",
    marker_color='indianred'
))

# P-Value line
fig.add_trace(go.Scatter(
    x=adf_df["Scheme_Code"],
    y=adf_df["p-value"],
    name="p-value",
    mode="lines+markers",
    line=dict(color='royalblue', width=2),
    yaxis="y2"
))

# Add horizontal line for significance threshold (0.05)
fig.add_shape(
    type="line",
    x0=-0.5, x1=len(adf_df["Scheme_Code"]) - 0.5,
    y0=0.05, y1=0.05,
    line=dict(color="green", dash="dash"),
    yref="y2"
)

fig.update_layout(
    title="ADF Stationarity Test per Scheme_Code",
    xaxis_title="Scheme_Code",
    yaxis=dict(title="ADF Statistic"),
    yaxis2=dict(title="p-value", overlaying='y', side='right'),
    legend=dict(x=0.01, y=0.99),
    template='plotly_white'
)

# Save as interactive HTML
os.makedirs("outputs/models", exist_ok=True)
html_path = "outputs/models/adf_stationarity_plot.html"
fig.write_html(html_path)
print(f" ADF Stationarity Plot saved → {html_path}")


def main():
    df = load_data()
    output_dir = os.path.join("outputs", "models")
    top_codes = pd.read_csv("data/processed/top5_scheme_summary.csv")["Scheme_Code"].tolist()

    p_values = [0, 1, 2, 4, 6, 8, 10]
    d_values = range(0, 3)
    q_values = range(0, 3)

    leaderboard = []
    for code in top_codes:
        metrics = train_optimized_arima(df, code, output_dir, p_values, d_values, q_values)
        if metrics:
            leaderboard.append(metrics)

    if leaderboard:
        leaderboard_df = pd.DataFrame(leaderboard)
        leaderboard_path = os.path.join(output_dir, "model_leaderboard.csv")
        if os.path.exists(leaderboard_path):
            old_df = pd.read_csv(leaderboard_path)
            leaderboard_df = pd.concat([old_df, leaderboard_df], ignore_index=True).drop_duplicates()
        leaderboard_df.to_csv(leaderboard_path, index=False)
        print(f"\n Optimized ARIMA leaderboard saved → {leaderboard_path}")

if __name__ == "__main__":
    main()
