# Prophet Forecasting Plan
# Format NAV data for Prophet (ds, y)
# Fit model for each top Scheme_Code
# Forecast 30 days ahead
# Plot forecast with uncertainty intervals
# Evaluate using MAE, RMSE, R²
# Create leaderboard

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import os
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(r"C:/Users/BALA/OneDrive - University of Hertfordshire/Desktop/final project/app/data/processed/preprocessed_mutual_funds.csv")


# Top scheme codes (based on earlier analysis)
top_scheme_codes = [100051, 100047, 100048]
prophet_results = []

for scheme_code in top_scheme_codes:
    fund_df = df[df['Scheme_Code'] == scheme_code][['Date', 'NAV']].copy()
    fund_df['Date'] = pd.to_datetime(fund_df['Date'])
    fund_df = fund_df.sort_values('Date')
    
    # Rename columns for Prophet
    prophet_df = fund_df.rename(columns={'Date': 'ds', 'NAV': 'y'})
    
    if len(prophet_df) < 100:
        continue  # skip small series
    
    # Train-test split
    train_size = int(len(prophet_df) * 0.8)
    train_df = prophet_df.iloc[:train_size]
    test_df = prophet_df.iloc[train_size:]
    
    # Fit Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(train_df)
    
    # Future prediction (for test period)
    future = model.make_future_dataframe(periods=len(test_df), freq='B')
    forecast = model.predict(future)
    
    # Merge actual test NAVs with forecast
    merged = pd.merge(test_df, forecast[['ds', 'yhat']], on='ds', how='inner')

    # If there’s no overlap in dates, skip gracefully
    if merged.empty:
        print(f"No overlapping dates for Scheme_Code {scheme_code}, skipping...")
        continue

    # Extract actual and predicted
    true = merged['y'].values
    pred = merged['yhat'].values

    # Evaluation
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    
    # Store results
    prophet_results.append({
        'Scheme_Code': scheme_code,
        'Fund_Name': df[df['Scheme_Code'] == scheme_code]['Scheme_Name'].iloc[0],
        'MAE': round(mae, 3),
        'RMSE': round(rmse, 3),
        'R2_Score': round(r2, 3)
    })
    
    fig = go.Figure()

    # Historical NAV
    fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines', name='Historical NAV'))

    # Forecast + Confidence Interval
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines+markers', name='Prophet Forecast'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot')))

    fig.update_layout(
        title=f"Prophet Forecast – Scheme Code {scheme_code}",
        xaxis_title="Date",
        yaxis_title="NAV",
        template="plotly_dark",
        height=500
    )
    fig.show()

# Leaderboard
leaderboard_prophet = pd.DataFrame(prophet_results).sort_values(by='RMSE')
print(" Prophet Forecast Leaderboard:")
print(leaderboard_prophet)

os.makedirs('data/results', exist_ok=True)
leaderboard_prophet.to_csv('C:/Users/BALA/OneDrive - University of Hertfordshire/Desktop/final project/app/data/results/prophet_leaderboard.csv', index=False)
print("Prophet leaderboard saved to data/results/prophet_leaderboard.csv")
