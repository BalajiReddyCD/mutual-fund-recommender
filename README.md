# Smart Mutual Fund Recommendation System

## 📈 Overview
This project aims to build a machine learning-powered recommendation engine for mutual fund investments using historical NAV data from Indian markets (2006–2023). It leverages time series analysis and user profiling to suggest optimal funds based on risk appetite and investment goals.

## 🔍 Objectives
- Analyze NAV trends and volatility
- Forecast fund growth using ML/time series models
- Recommend top mutual funds using clustering & profiling
- Deploy as an API for real-world usability

## 🗃️ Dataset
- Source: [Kaggle - Indian Mutual Funds Dataset (2006–2023)](https://www.kaggle.com/datasets/balajisr/indian-mutual-funds-dataset-2023)
- Collected from: [AMFI India](https://www.amfiindia.com)

## 💻 Technologies
- Python, Pandas, NumPy
- scikit-learn, statsmodels, Keras
- Flask/FastAPI for deployment
- Matplotlib, Seaborn for visualizations

## 🚀 How to Run
```bash
pip install -r requirements.txt
python src/api.py
