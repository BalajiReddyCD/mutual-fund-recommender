import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

def run_arima_forecast(df, steps=30):
    nav_series = df['NAV'].values
    train_size = len(nav_series) - steps
    train, test = nav_series[:train_size], nav_series[train_size:]

    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=steps)

    return predictions, test
