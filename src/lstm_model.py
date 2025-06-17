import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def run_lstm_forecast(df, sequence_len=30, epochs=20):
    df = df.copy()
    scaler = MinMaxScaler()
    df['scaled_nav'] = scaler.fit_transform(df[['NAV']])

    nav = df['scaled_nav'].values
    X, y = [], []

    for i in range(sequence_len, len(nav)):
        X.append(nav[i-sequence_len:i])
        y.append(nav[i])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Use last 30 points as test
    X_train, y_train = X[:-30], y[:-30]
    X_test, y_test = X[-30:], y[-30:]

    model = Sequential([
        LSTM(50, input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

    return predictions.flatten(), y_test_orig.flatten()
