# app/routers/recommend.py

from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pandas as pd
import os

router = APIRouter()

LEADERBOARD_PATH = "C:/Users/BALA/OneDrive - University of Hertfordshire/Desktop/final project/app/data/results/final_model_comparison.csv"

@router.get("/leaderboard", summary="Get model leaderboard")
def get_leaderboard():
    if not os.path.exists(LEADERBOARD_PATH):
        return JSONResponse(status_code=404, content={"error": "Leaderboard not found."})
    
    df = pd.read_csv(LEADERBOARD_PATH)
    return df.to_dict(orient="records")

@router.get("/recommend", summary="Get top mutual fund recommendations")
def get_recommendations():
    if not os.path.exists(LEADERBOARD_PATH):
        return JSONResponse(status_code=404, content={"error": "Leaderboard not found."})

    df = pd.read_csv(LEADERBOARD_PATH)

    top_funds = (
        df.sort_values(by=["Model", "RMSE"])  # Lower RMSE = better
          .groupby("Model")
          .first()
          .reset_index()
    )

    return top_funds.to_dict(orient="records")
