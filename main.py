# app/main.py

from fastapi import FastAPI
from app.routers import recommend, funds

app = FastAPI(
    title="📈 Smart Mutual Fund Recommender API",
    description="Compare ARIMA, Prophet, and LSTM models for mutual fund forecasting and get recommendations.",
    version="1.0.0"
)

app.include_router(recommend.router)
app.include_router(funds.router) 
