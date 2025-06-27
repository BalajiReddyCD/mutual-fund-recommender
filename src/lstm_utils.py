import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df, scheme_code, sequence_length=10, test_size=0.2, return_dates=False):
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    scheme_df = df[df["Scheme_Code"] == scheme_code].copy()
    scheme_df.sort_values("Date", inplace=True)

    if len(scheme_df) < sequence_length + 1:
        raise ValueError("Not enough data for the selected Scheme_Code.")

    nav_values = scheme_df["NAV"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    nav_scaled = scaler.fit_transform(nav_values)

    X, y, dates = [], [], []

    for i in range(sequence_length, len(nav_scaled)):
        X.append(nav_scaled[i - sequence_length:i])
        y.append(nav_scaled[i])
        if return_dates:
            dates.append(scheme_df.iloc[i]["Date"])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * (1 - test_size))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if return_dates:
        train_dates, test_dates = dates[:split], dates[split:]
        return X_train, X_test, y_train, y_test, scaler, train_dates, test_dates

    return X_train, X_test, y_train, y_test, scaler
