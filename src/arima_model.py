# ARIMA Forecasting Plan
# Pick Top Scheme_Codes (by CAGR or data volume)
# Prepare NAV Time Series per fund
# Test stationarity (ADF Test)
# Auto-fit ARIMA with pmdarima.auto_arima()
# Forecast NAV and plot
# Evaluate with MAE, RMSE, R²
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:/Users/BALA/OneDrive - University of Hertfordshire/Desktop/final project/app/data/processed/preprocessed_mutual_funds.csv")

# Identify top 3 Scheme_Codes by record count
top_scheme_codes = df['Scheme_Code'].value_counts().head(3).index.tolist()

# Display the selected Scheme Codes
top_scheme_codes

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Step 1: Prepare the top scheme codes
top_scheme_codes = [100051, 100047, 100048]
results = []

# Step 2: Loop through each scheme
for scheme_code in top_scheme_codes:
    try:
        ts_df = df[df['Scheme_Code'] == scheme_code][['Date', 'NAV']].sort_values('Date')
        ts_df['Date'] = pd.to_datetime(ts_df['Date'])
        ts_df.set_index('Date', inplace=True)

        print(f"\n⏳ Processing Scheme Code: {scheme_code} - Total records: {len(ts_df)}")
        if len(ts_df) < 100:
            print(f"Skipped {scheme_code}: Less than 100 records.")
            continue
        
        # Train-test split
        train_size = int(len(ts_df) * 0.8)
        train, test = ts_df['NAV'].iloc[:train_size], ts_df['NAV'].iloc[train_size:]
        
        # Fit ARIMA model
        model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore")
        
        # Forecast
        forecast = model.predict(n_periods=30)
        forecast_index = pd.date_range(ts_df.index[-1], periods=31, freq='B')[1:]
        forecast_series = pd.Series(forecast, index=forecast_index)
        
        # Evaluate
        test_preds = model.predict(n_periods=len(test))
        mae = mean_absolute_error(test, test_preds)
        rmse = np.sqrt(mean_squared_error(test, test_preds))
        r2 = r2_score(test, test_preds)

        # Save results
        results.append({
            'Scheme_Code': scheme_code,
            'Fund_Name': df[df['Scheme_Code'] == scheme_code]['Scheme_Name'].iloc[0],
            'MAE': round(mae, 3),
            'RMSE': round(rmse, 3),
            'R2_Score': round(r2, 3)
        })

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df['NAV'], mode='lines', name='Historical NAV'))
        fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series, mode='lines+markers', name='ARIMA Forecast'))

        split_date = ts_df.index[train_size]
        fig.add_shape(type="line", x0=split_date, x1=split_date, y0=ts_df['NAV'].min(), y1=ts_df['NAV'].max(),
                      line=dict(dash="dot", color="black", width=2))
        fig.add_annotation(x=split_date, y=ts_df['NAV'].max(), text="Train/Test Split", showarrow=True, arrowhead=2, ax=0, ay=-40)

        fig.update_layout(title=f"ARIMA Forecast – Scheme Code {scheme_code}",
                          xaxis_title="Date", yaxis_title="NAV", height=500)

        fig.show()

    except Exception as e:
        print(f" Error processing Scheme Code {scheme_code}: {e}")


# Step 3: Display leaderboard
leaderboard_df = pd.DataFrame(results).sort_values(by='RMSE')
print("ARIMA Forecast Leaderboard:")
print(leaderboard_df)

# Save to CSV
os.makedirs('data/results', exist_ok=True)
leaderboard_df.to_csv('C:/Users/BALA/OneDrive - University of Hertfordshire/Desktop/final project/app/data/results/arima_leaderboard.csv', index=False)
print("ARIMA leaderboard saved to data/results/arima_leaderboard.csv")
