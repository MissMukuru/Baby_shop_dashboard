from pathlib import Path
from loguru import logger
import typer
import pandas as pd

from nila_baby_shop.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "baby_shop_dataset_10000_rows.csv",
    base_features_output: Path = PROCESSED_DATA_DIR / "base_features.csv",
    forecast_features_output: Path = PROCESSED_DATA_DIR / "forecast_features.csv",
    viral_features_output: Path = PROCESSED_DATA_DIR / "viral_features.csv",
    labels_output: Path = PROCESSED_DATA_DIR / "labels.csv"
):
    logger.info("Loading dataset...")

    df = pd.read_csv(input_path)


    df['date'] = pd.to_datetime(df['date'])

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

    df['engagement_rate'] = df['likes'] / (df['views'] + 1)
    df['comment_rate'] = df['comments_count'] / (df['views'] + 1)

    df = pd.get_dummies(df, columns=['category'], drop_first=True)

    df_base = df.drop(columns=['sample_comment', 'simulated_sales_units'], errors='ignore')
    df_base.to_csv(base_features_output, index=False)

    #Forecast Feature Pipeline → reads base features + creates lags, rolling stats
    df_forecast = df_base.copy()
    df_forecast = df_forecast.sort_values(['product','date'])
    df_forecast['lag_1_views'] = df_forecast.group_by('product')['view'].shift(1)
    df_forecast['lag_7_views'] = df_forecast.group_by('product')['view'].shift(7)
    df_forecast['rolling_mean_7'] = df_forecast.groupby('product')['view'].transform(lambda x: x.rolling(7).mean())
    df_forecast = df_forecast.fillna(0)
    df_forecast.to_csv('forecast_features_output', index = False)

    #Predict which products will spike in popularity.
    df_viral = df_base.copy()
    df_viral = df_viral.sort_values(['product','date'])
    df_engagement = df_viral['views'] + df_viral['likes'] + df_viral['comments_count']
    df_engagement_growth = df_viral.groupby('product')['engagement'].diff().fillna(0)
    df_viral.to_csv(viral_features_output, index=False)

    labels = df['simulated_sales_units']
    labels.to_csv(labels_output, index=False)

    logger.success("Features and labels saved")

if __name__ == "__main__":
    app()