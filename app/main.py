"""
FastAPI backend for Nila Baby Shop demand forecasting.

Wraps the existing joblib model artifact (models/demand_forecast_model.pkl)
produced by Nila_baby_shop/modeling/demand_forecast.py, and exposes it as a
versioned HTTP API instead of a script only the Streamlit app can call.

Run locally:
    uvicorn app.main:app --reload --port 8000

Docs (auto-generated):
    http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "demand_forecast_model.pkl"

# Populated at startup by the lifespan handler below.
_state: dict = {"forecast_model": None, "viral_model": None, "model_version": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup: load model once, keep it in memory for the life of the process ---
    if not MODEL_PATH.exists():
        # Don't crash the whole app if the model hasn't been trained yet locally;
        # /health will report this so it's obvious in a demo or CI run.
        _state["forecast_model"] = None
        _state["viral_model"] = None
        _state["model_version"] = None
    else:
        payload = joblib.load(MODEL_PATH)
        _state["forecast_model"] = payload["models"]["forecast_model"]
        _state["viral_model"] = payload["models"]["viral_model"]
        # Simple content-based version: file mtime. Swap for an MLflow run id
        # once experiment tracking is wired in.
        _state["model_version"] = str(int(MODEL_PATH.stat().st_mtime))
    yield
    # --- shutdown: nothing to clean up yet ---


app = FastAPI(
    title="Nila Baby Shop - Forecast API",
    description="Serves demand forecast and viral-potential predictions.",
    version="0.1.0",
    lifespan=lifespan,
)


class ForecastRequest(BaseModel):
    """One row of features, matching what demand_forecast.py trained on."""

    likes: float = 0
    comment_count: float = 0
    estimated_price_ksh: float = 0
    is_weekend: int = 0
    year: int
    month: int
    day: int
    week_of_year: int
    lag_1_views: float = 0
    lag_7_views: float = 0
    rolling_mean_7: float = 0
    category: Optional[str] = Field(
        default=None, description="e.g. 'rompers', 'strollers' - used to build category_* one-hot columns"
    )
    views: float = Field(default=0, description="Only needed for the viral-probability model")
    engagement_rate: float = Field(default=0, description="Only needed for the viral-probability model")


class ForecastResponse(BaseModel):
    demand_prediction: float
    viral_probability: float
    viral_prediction: int
    model_version: Optional[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness/readiness probe for load balancers, k8s, Cloud Run, etc."""
    loaded = _state["forecast_model"] is not None
    return HealthResponse(
        status="ok" if loaded else "model_not_loaded",
        model_loaded=loaded,
        model_version=_state["model_version"],
    )


@app.post("/v1/predict", response_model=ForecastResponse, tags=["predict"])
def predict(payload: ForecastRequest) -> ForecastResponse:
    """
    Predict expected demand and viral probability for a single product/day row.

    This mirrors the feature construction in
    Nila_baby_shop/modeling/predict_demand.py, but takes a single JSON row
    instead of a CSV, so the frontend (or any client) can call it directly.
    """
    forecast_model = _state["forecast_model"]
    viral_model = _state["viral_model"]

    if forecast_model is None or viral_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train it first: "
            "python -m nila_baby_shop.modeling.demand_forecast",
        )

    row = payload.model_dump()
    category = row.pop("category")
    df = pd.DataFrame([row])

    # Recreate the one-hot category_* columns the model was trained on.
    forecast_features = list(getattr(forecast_model, "feature_names_in_", []))
    for col in forecast_features:
        if col.startswith("category_"):
            df[col] = 1 if category and col == f"category_{category}" else 0
        elif col not in df.columns:
            df[col] = 0

    X_forecast = df[forecast_features]
    demand_prediction = float(forecast_model.predict(X_forecast)[0])

    viral_features = list(getattr(viral_model, "feature_names_in_", ["views", "likes", "engagement_rate"]))
    for col in viral_features:
        if col not in df.columns:
            df[col] = demand_prediction if col == "views" else 0

    X_viral = df[viral_features]
    viral_probability = float(viral_model.predict_proba(X_viral)[:, 1][0])
    viral_prediction = int(viral_model.predict(X_viral)[0])

    return ForecastResponse(
        demand_prediction=demand_prediction,
        viral_probability=viral_probability,
        viral_prediction=viral_prediction,
        model_version=_state["model_version"],
    )
