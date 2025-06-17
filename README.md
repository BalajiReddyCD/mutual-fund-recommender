# Smart Mutual Fund Recommendation System

## Overview

This project builds a machine learning-powered recommendation engine for mutual fund investments using historical NAV data from Indian markets (2006–2023). The system leverages time series analysis (ARIMA, LSTM) and user profiling to recommend top-performing funds aligned with the investor’s risk appetite and financial goals.

---

## Objectives

- Analyze NAV trends and volatility for various mutual fund schemes  
- Forecast fund growth using ARIMA and LSTM time series models  
- Recommend top mutual funds based on clustering and risk profiling  
- Deploy the system as a REST API for real-world usability  

---

## Dataset

- Source: Kaggle - Indian Mutual Funds Dataset (2006–2023)  
  https://www.kaggle.com/datasets/balajisr/indian-mutual-funds-dataset-20062023  
- Collected from: Association of Mutual Funds in India (AMFI)  
- Size: ~29 million NAV records from over 35,000 schemes

---

## Technologies Used

- Languages: Python  
- Libraries:  
  - Pandas, NumPy  
  - scikit-learn, statsmodels, pmdarima, Keras  
  - Flask, FastAPI  
  - Matplotlib, Seaborn  

---

## How to Run

1. Clone the repo:
   git clone https://github.com/BalajiReddyCD/mutual-fund-recommender.git

2. Navigate to project folder:
   cd mutual-fund-recommender

3. Install dependencies:
   pip install -r requirements.txt

4. Run the API (if implemented):
   python src/api.py

---

## Project Structure

mutual-fund-recommender/  
├── data/  
│   ├── raw/                      # Original raw data (optional)  
│   └── processed/                # Cleaned & preprocessed data  
│  
├── notebooks/  
│   ├── eda.ipynb                 # Exploratory data analysis  
│   └── modeling.ipynb            # Model training and evaluation (ARIMA, LSTM)  
│  
├── src/  
│   ├── arima_model.py            # ARIMA model logic and plotting  
│   ├── lstm_model.py             # LSTM model training and prediction  
│   ├── lstm_utils.py             # Helper functions for LSTM preprocessing  
│   └── api.py                    # (Optional) API deployment with Flask/FastAPI  
│  
├── outputs/  
│   ├── arima_forecast.png        # Saved prediction plots  
│   └── nav_predictions.csv       # Output CSV with model predictions  
│  
├── LICENSE  
├── README.md  
├── requirements.txt  
└── .gitignore  

---

## Project Status

✅ ARIMA-based NAV forecasting complete  
✅ LSTM model under training  
🚧 Personalized recommendation (risk profiling) in progress  
🚧 API deployment pending  

---

## License

This project is licensed under the MIT License.

---

## Contributions

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change or contribute.

---

## Maintainer

Balaji Manjulamma Sriramareddy  
University of Hertfordshire | MSc Data Science
