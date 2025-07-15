# app/routers/recommend.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import os
from app.utils.loader import load_leaderboard, load_recommendations

router = APIRouter()

@router.get("/leaderboard", summary="Get model leaderboard")
def get_leaderboard():
    try:
        df = load_leaderboard()
        return df.to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Leaderboard file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommend", summary="Get top mutual fund recommendations")
def get_recommendations():
    try:
        rec_df = load_recommendations()

        if rec_df.empty:
            return JSONResponse(status_code=204, content={"message": "No recommendations found."})

        return rec_df.to_dict(orient="records")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Recommendations file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
