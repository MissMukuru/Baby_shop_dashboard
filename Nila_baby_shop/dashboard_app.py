from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from nila_baby_shop.config import PROCESSED_DATA_DIR


def _load_forecast_data() -> pd.DataFrame:
    predictions_path = PROCESSED_DATA_DIR / "demand_predictions.csv"
    if not predictions_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(predictions_path)
    df.columns = [c.strip() for c in df.columns]

    if "comments_count" in df.columns and "comment_count" not in df.columns:
        df = df.rename(columns={"comments_count": "comment_count"})

    if "date" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if "product" not in df.columns:
        df["product"] = "Unknown Product"

    if "demand_prediction" in df.columns:
        df["predicted_demand"] = pd.to_numeric(df["demand_prediction"], errors="coerce")
    elif "predicted_demand" in df.columns:
        df["predicted_demand"] = pd.to_numeric(df["predicted_demand"], errors="coerce")
    else:
        df["predicted_demand"] = pd.NA

    return df.sort_values(["product", "date"]).reset_index(drop=True)


def _demand_forecast_page(df: pd.DataFrame) -> None:
    st.subheader("Demand Forecast")
    st.caption("Prediction output table from demand_forecast (no model evaluation metrics).")
    st.info(
        "Analyst note: Think of this as two lenses on the same product story. "
        "`predicted_demand` tells us expected buying volume, while `viral_prob` tells us social buzz intensity. "
        "So a product can quietly sell very well (high demand, low viral) or trend loudly without converting as much."
    )

    if df.empty:
        st.warning("No forecast data found. Generate data/processed/demand_predictions.csv first.")
        return

    products = sorted(df["product"].dropna().astype(str).unique().tolist())
    if not products:
        st.warning("No products found in forecast data.")
        return

    product = st.selectbox("Select product", options=products)
    product_df = df[df["product"] == product].copy().sort_values("date", ascending=False)
    if product_df.empty:
        st.info("No rows available for this product.")
        return

    st.markdown("### Expected Demand Trend")
    trend_df = df[df["product"] == product].copy().sort_values("date")
    fig = px.line(
        trend_df,
        x="date",
        y="predicted_demand",
        title=f"Predicted Demand Over Time - {product}",
        markers=True,
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Predicted Demand")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Latest Prediction Output (Selected Product)")
    product_cols = ["date", "product", "predicted_demand", "viral_prob", "viral_prediction"]
    product_cols = [c for c in product_cols if c in product_df.columns]
    st.dataframe(product_df[product_cols].head(20), use_container_width=True)

    st.markdown("### Latest Prediction Output (All Products)")
    all_latest = (
        df.sort_values("date", ascending=False)
        .groupby("product", as_index=False)
        .head(1)
        .sort_values("predicted_demand", ascending=False)
    )
    all_cols = ["date", "product", "predicted_demand", "viral_prob", "viral_prediction"]
    all_cols = [c for c in all_cols if c in all_latest.columns]
    st.dataframe(all_latest[all_cols], use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Nila Baby Shop - Demand Forecast", page_icon="DF", layout="wide")
    st.title("Nila Baby Shop Demand Forecast")
    _demand_forecast_page(_load_forecast_data())


if __name__ == "__main__":
    main()
