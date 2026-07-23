"""
Tests for the FastAPI backend. Run with: pytest tests/test_api.py -v

These are intentionally light — they check the contract (does /health respond,
does /v1/predict return the right shape) rather than model accuracy, which
belongs in a separate model-evaluation step (e.g. tracked via MLflow).
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_responds():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "model_loaded" in body


def test_predict_returns_expected_shape_when_model_loaded():
    response = client.post(
        "/v1/predict",
        json={
            "likes": 120,
            "comment_count": 8,
            "estimated_price_ksh": 1500,
            "is_weekend": 1,
            "year": 2026,
            "month": 7,
            "day": 22,
            "week_of_year": 30,
            "lag_1_views": 300,
            "lag_7_views": 250,
            "rolling_mean_7": 275,
            "category": "rompers",
            "views": 300,
            "engagement_rate": 0.05,
        },
    )
    # 503 is an acceptable response if the model hasn't been trained yet
    # (e.g. fresh CI checkout with no models/*.pkl committed) — the important
    # thing is the API never 500s.
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert "demand_prediction" in body
        assert "viral_probability" in body
        assert 0.0 <= body["viral_probability"] <= 1.0
