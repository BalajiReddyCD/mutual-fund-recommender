# app/routers/funds.py

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import pandas as pd
import os

router = APIRouter()

def load_leaderboard():
    path = "data/results/final_model_comparison.csv"
    return pd.read_csv(path)

@router.get("/fund/{scheme_code}")
def get_fund_details(scheme_code: int):
    df = load_leaderboard()
    fund_info = df[df["Scheme_Code"] == scheme_code]

    if fund_info.empty:
        return JSONResponse(status_code=404, content={"error": "Fund not found"})

    models = ["arima", "prophet", "lstm"]
    forecasts = {}
    for model in models:
        fpath = f"data/results/{model}_predictions/{scheme_code}.csv"
        if os.path.exists(fpath):
            forecasts[model] = pd.read_csv(fpath).tail(5).to_dict(orient="records")

    return {
        "fund": fund_info.to_dict(orient="records")[0],
        "forecasts": forecasts
    }

@router.get("/model/{model_name}")
def get_model_stats(model_name: str):
    file_path = f"data/results/{model_name.lower()}_leaderboard.csv"

    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Model not found"})

    df = pd.read_csv(file_path)
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
