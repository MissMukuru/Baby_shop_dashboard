from pathlib import Path

import joblib
import pandas as pd
import typer
from loguru import logger

from nila_baby_shop.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def _normalize_bool_like_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0})
                .fillna(0)
                .astype(int)
            )
    return df


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "forecast_features.csv",
    model_path: Path = MODELS_DIR / "demand_forecast_model.pkl",
    output_path: Path = PROCESSED_DATA_DIR / "demand_predictions.csv",
):
    logger.info(f"Loading input data from {input_path}")
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"comments_count": "comment_count"})

    payload = joblib.load(model_path)
    forecast_model = payload["models"]["forecast_model"]
    viral_model = payload["models"]["viral_model"]

    forecast_features = list(getattr(forecast_model, "feature_names_in_", []))
    if not forecast_features:
        raise ValueError("Forecast model is missing feature names; retrain model and save again.")

    # Fill missing forecast features with zeros to keep inference robust.
    for col in forecast_features:
        if col not in df.columns:
            df[col] = 0

    category_cols = [c for c in forecast_features if c.startswith("category_")]
    df = _normalize_bool_like_columns(df, category_cols)

    X_forecast = df[forecast_features]
    df["demand_prediction"] = forecast_model.predict(X_forecast)

    viral_features = list(getattr(viral_model, "feature_names_in_", ["views", "likes", "engagement_rate"]))
    for col in viral_features:
        if col not in df.columns:
            if col == "views":
                # At inference time true views may be unknown; use predicted demand as proxy.
                df[col] = df["demand_prediction"]
            else:
                df[col] = 0

    X_viral = df[viral_features]
    df["viral_prob"] = viral_model.predict_proba(X_viral)[:, 1]
    df["viral_prediction"] = viral_model.predict(X_viral).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    app()
