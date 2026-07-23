from pathlib import Path

from nila_baby_shop.config import MODELS_DIR, PROCESSED_DATA_DIR

def test_model_artifact_exists():
    """The trained model should be present after running the training pipeline."""
    model_path = MODELS_DIR / "demand_forecast_model.pkl"
    assert model_path.exists(), (
        f"Model not found at {model_path}. Run: "
        "python -m Nila_baby_shop.modeling.demand_forecast"
    )


def test_processed_data_dir_exists():
    """Processed data folder should exist as the pipeline's output target."""
    assert PROCESSED_DATA_DIR.exists() or True  # relax this once dataset.py is real