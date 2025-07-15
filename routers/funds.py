# app/routers/funds.py

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from app.utils.loader import (
    load_leaderboard,
    load_forecast_csv,
    load_model_leaderboard
)
import pandas as pd

router = APIRouter()

@router.get("/fund/{scheme_code}")
def get_fund_details(scheme_code: int):
    df = load_leaderboard()
    fund_info = df[df["Scheme_Code"] == scheme_code]

    if fund_info.empty:
        raise HTTPException(status_code=404, detail="Fund not found")

    # Load last 5 forecasts from each model if available
    forecasts = {}
    for model in ["arima", "prophet", "lstm"]:
        df_forecast = load_forecast_csv(model, scheme_code)
        if df_forecast is not None:
            forecasts[model] = df_forecast.tail(5).to_dict(orient="records")

    return {
        "fund": fund_info.to_dict(orient="records")[0],
        "forecasts": forecasts
    }

@router.get("/model/{model_name}")
def get_model_stats(model_name: str):
    df = load_model_leaderboard(model_name)
    if df is None:
        raise HTTPException(status_code=404, detail="Model not found")

    top5 = df.sort_values(by="RMSE").head(5)
    return {
        "model": model_name.upper(),
        "top_5_funds": top5.to_dict(orient="records"),
        "total_funds": len(df)
    }

@router.get("/funds/search")
def search_funds(name: str = Query(..., description="Search by partial fund name")):
    df = load_leaderboard()
    results = df[df["Fund_Name"].str.contains(name, case=False)]

    if results.empty:
        return JSONResponse(status_code=404, content={"message": "No match found."})

    return results[["Scheme_Code", "Fund_Name", "Model", "RMSE"]].to_dict(orient="records")
