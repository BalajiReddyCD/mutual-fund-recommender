from fastapi import FastAPI
from app.routers import recommend, funds

app = FastAPI(
    title="Mutual Fund Recommender API",
    description="API for fund-level insights, model forecasts, and investment recommendations",
    version="1.0"
)

# ✅ Add this root route
@app.get("/")
def root():
    return {
        "message": "Welcome to the Mutual Fund Recommender API 🚀",
        "available_endpoints": [
            "/fund/{scheme_code}",
            "/model/{model_name}",
            "/funds/search",
            "/leaderboard",
            "/recommend"
        ],
        "docs": "/docs"
    }

# Register routers
app.include_router(funds.router)
app.include_router(recommend.router)
