import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
import joblib
import typer
from pathlib import Path
from loguru import logger
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')


from nila_baby_shop.config import PROCESSED_DATA_DIR, MODELS_DIR
app = typer.Typer()


@app.command()
def main(
    forecast_features_path: Path = PROCESSED_DATA_DIR / "forecast_features.csv",
    normal_input_path: Path = PROCESSED_DATA_DIR / "base_features.csv",
    model_output: Path = MODELS_DIR / "demand_forecast_model.pkl",
):
    normal_df = pd.read_csv(normal_input_path)
    normal_df.columns = normal_df.columns.str.strip()
    normal_df = normal_df.rename(columns={"comments_count": "comment_count"})
    normal_df["trend_score"] = (
        normal_df["views"] * 0.5
        + normal_df["likes"] * 2
        + normal_df["comment_count"] * 3
    )

    forecast_df = pd.read_csv(forecast_features_path)
    forecast_df.columns = forecast_df.columns.str.strip()
    forecast_df = forecast_df.rename(columns={"comments_count": "comment_count"})
    logger.info(
        f"Loaded demand forecast features. First rows:\n{forecast_df.head().to_string(index=False)}"
    )

    feature_columns = [
            "likes",
            "comment_count",
            "estimated_price_ksh",
            "is_weekend",
            "year",
            "month",
            "day",
            "week_of_year",
            "lag_1_views",
            "lag_7_views",
            "rolling_mean_7"
    ]

    cat_columns = [c for c in forecast_df.columns if c.startswith("category_")]
    feature_columns += cat_columns

    # Normalize one-hot category columns that may be loaded as string/object.
    for col in cat_columns:
        forecast_df[col] = (
            forecast_df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )

    forecast_df = forecast_df.dropna()

    X_df = forecast_df[feature_columns]
    y_df = forecast_df["views"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2, shuffle=False
    )

    forecast_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.5,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    logger.info("Training....")
    forecast_model.fit(X_train, y_train)
    y_pred = forecast_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    logger.success(f"RMSE: {rmse:.4f}")

    # VIRAL DETECTION MODEL
    normal_df["viral"] = (
        normal_df["engagement_rate"] > normal_df["engagement_rate"].quantile(0.75)
    ).astype(int)
    X_v = normal_df[["views", "likes", "engagement_rate"]]
    y_v = normal_df["viral"]


    X_train, X_test, y_train, y_test = train_test_split(X_v, y_v, test_size=0.2, shuffle=True)

    viral_model = RandomForestClassifier()
    viral_model.fit(X_train, y_train)
    y_v_pred = viral_model.predict(X_test)
    logger.info(f"Viral model accuracy: {accuracy_score(y_test, y_v_pred):.3f}")
    logger.info(f"Viral model report:\n{classification_report(y_test, y_v_pred)}")

    # Stock recommendation
    normal_df["viral_prob"] = viral_model.predict_proba(X_v)[:, 1]

    normal_df["stock_score"] = (
        0.4 * normal_df["trend_score"] + 0.4 * normal_df["views"] + 0.2 * normal_df["viral_prob"]
    )

    normal_df["recommended_stock_units"] = (normal_df["stock_score"] / 100).astype(int)

    # Social media insights
    sia = SentimentIntensityAnalyzer()

    if "sample_comment" not in normal_df.columns:
        normal_df["sample_comment"] = ""
    normal_df["sentiment"] = normal_df["sample_comment"].astype(str).apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

    print("Demand Model RMSE:", rmse)
    print(normal_df[[
        "product",
        "views",
        "trend_score",
        "viral_prob",
        "stock_score",
        "recommended_stock_units",
        "sentiment"
    ]].head())

    joblib.dump(
        {
            "models": {
                "forecast_model": forecast_model,
                "viral_model": viral_model,
            },
        },
        model_output,
    )

    logger.success(f"All models saved to {model_output}")



if __name__ == "__main__":
    app()


