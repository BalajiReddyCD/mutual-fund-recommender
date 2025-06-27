import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from lstm_utils import preprocess_data

def load_data():
    df = pd.read_csv("data/processed/preprocessed_mutual_funds.csv")
    return df

def get_best_scheme_code(df):
    scheme_counts = df['Scheme_Code'].value_counts()
    best = scheme_counts[scheme_counts > 100].index[0]
    print(f"✅ Using Scheme_Code: {best} with {scheme_counts[best]} entries")
    return best

def plot_predictions(actual, predicted, scheme_code):
    plt.figure(figsize=(10, 4))
    plt.plot(actual, label='Actual NAV')
    plt.plot(predicted, label='Predicted NAV')
    plt.title(f"LSTM NAV Prediction - Scheme {scheme_code}")
    plt.xlabel("Time Steps")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_and_save_model(scheme_code):
    df = load_data()

    try:
        X_train, X_test, y_train, y_test, scaler, train_dates, test_dates = preprocess_data(df, scheme_code=scheme_code, return_dates=True)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=30, batch_size=32, callbacks=[early_stop], verbose=1)

    predictions = model.predict(X_test)
    predicted_nav = scaler.inverse_transform(predictions)
    actual_nav = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Metrics
    mae = mean_absolute_error(actual_nav, predicted_nav)
    rmse = np.sqrt(mean_squared_error(actual_nav, predicted_nav))
    r2 = r2_score(actual_nav, predicted_nav)

    print("\n📊 Evaluation Metrics:")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Save model
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"LSTM_scheme_{scheme_code}_{timestamp}.h5"
    model_path = os.path.join(output_dir, model_filename)
    model.save(model_path)
    print(f"✅ Model trained and saved to: {model_path}")

    # ✅ Save predictions with Date
    predictions_df = pd.DataFrame({
        "Date": test_dates,
        "Actual_NAV": actual_nav.flatten(),
        "Predicted_NAV": predicted_nav.flatten()
    })
    predictions_filename = f"{scheme_code}_LSTM_preds.csv"
    predictions_path = os.path.join(output_dir, predictions_filename)
    predictions_df.to_csv(predictions_path, index=False)
    print(f"📁 Predictions saved to: {predictions_path}")

    # ✅ Update leaderboard
    leaderboard_path = os.path.join(output_dir, "model_leaderboard.csv")
    leaderboard_entry = pd.DataFrame([{
        "Model": "LSTM",
        "Scheme_Code": scheme_code,
        "Timestamp": timestamp,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }])

    if os.path.exists(leaderboard_path):
        existing_df = pd.read_csv(leaderboard_path)
        updated_df = pd.concat([existing_df, leaderboard_entry], ignore_index=True)
    else:
        updated_df = leaderboard_entry

    updated_df.to_csv(leaderboard_path, index=False)
    print(f"🏁 Leaderboard updated: {leaderboard_path}")

    # Plot
    plot_predictions(actual_nav, predicted_nav, scheme_code)


if __name__ == "__main__":
    df = load_data()
    scheme_code = get_best_scheme_code(df)
    train_and_save_model(scheme_code)
