from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np

from nila_baby_shop.config import PROCESSED_DATA_DIR


def _load_forecast_data() -> pd.DataFrame:
    predictions_path = PROCESSED_DATA_DIR / "demand_predictions.csv"
    if not predictions_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(predictions_path)
    df.columns = [c.strip() for c in df.columns]

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


def _load_inventory_data() -> pd.DataFrame:
    # Inventory intelligence uses the same predictions file for now.
    return _load_forecast_data()


def _build_weekly_outlook(df: pd.DataFrame, weeks: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    today = pd.Timestamp.today().normalize()
    future = df[df["date"] >= today].copy()

    # If there are no forward dates in file, use latest available period.
    if future.empty:
        future = df.copy()

    future["week_start"] = future["date"].dt.to_period("W-MON").dt.start_time
    future = future.sort_values("date")

    week_order = sorted(future["week_start"].dropna().unique())[:weeks]
    weekly = future[future["week_start"].isin(week_order)].copy()

    weekly_summary = (
        weekly.groupby("week_start", as_index=False)["predicted_demand"]
        .sum()
        .rename(columns={"predicted_demand": "total_predicted_demand"})
        .sort_values("week_start")
    )

    weekly_products = (
        weekly.groupby(["week_start", "product"], as_index=False)["predicted_demand"]
        .sum()
        .sort_values(["week_start", "predicted_demand"], ascending=[True, False])
    )

    return weekly_summary, weekly_products


def _build_product_growth_table(weekly_products: pd.DataFrame, product: str) -> pd.DataFrame:
    growth = (
        weekly_products[weekly_products["product"] == product]
        .sort_values("week_start")
        .copy()
        .rename(columns={"predicted_demand": "weekly_predicted_demand"})
    )
    growth["wow_growth_pct"] = growth["weekly_predicted_demand"].pct_change() * 100
    return growth


def _demand_forecast_page(df: pd.DataFrame) -> None:
    st.subheader("Demand Forecast")
    st.caption("Expected item demand for the next weeks.")
    st.info(
        "Explainer: This view summarizes expected demand from the model into weekly planning windows. "
        "Use the line chart to see overall business direction week by week, and use the table "
        "to identify which items should be prioritized each week."
    )

    if df.empty:
        st.warning("No forecast data found. Generate data/processed/demand_predictions.csv first.")
        return

    weekly_summary, weekly_products = _build_weekly_outlook(df, weeks=5)
    if weekly_summary.empty:
        st.warning("No weekly forecast rows available.")
        return

    st.markdown("### 5-Week Expected Demand Trend")
    fig = px.line(
        weekly_summary,
        x="week_start",
        y="total_predicted_demand",
        markers=True,
        title="Expected Demand (All Products) - Next 5 Weeks",
    )
    fig.update_layout(xaxis_title="Week Start", yaxis_title="Total Expected Demand")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 5-Week Business Outlook Table")
    outlook = weekly_summary.copy()
    outlook["week_start"] = outlook["week_start"].dt.date
    st.dataframe(outlook, use_container_width=True)

    st.markdown("### Weekly Item Priorities")
    selected_week = st.selectbox(
        "Select week",
        options=weekly_summary["week_start"].tolist(),
        format_func=lambda x: x.strftime("%Y-%m-%d"),
    )
    week_table = weekly_products[weekly_products["week_start"] == selected_week].copy()
    week_table["week_start"] = week_table["week_start"].dt.date
    st.dataframe(week_table, use_container_width=True)

    st.markdown("### Product Week-by-Week Growth")
    product_options = sorted(weekly_products["product"].dropna().astype(str).unique().tolist())
    selected_product = st.selectbox("Select product to analyze growth", options=product_options)

    growth_table = _build_product_growth_table(weekly_products, selected_product)
    if growth_table.empty:
        st.info("No weekly growth data available for this product.")
        return

    growth_fig = px.line(
        growth_table,
        x="week_start",
        y="weekly_predicted_demand",
        markers=True,
        title=f"Weekly Expected Demand - {selected_product}",
    )
    growth_fig.update_layout(xaxis_title="Week Start", yaxis_title="Weekly Predicted Demand")
    st.plotly_chart(growth_fig, use_container_width=True)

    growth_display = growth_table.copy()
    growth_display["week_start"] = growth_display["week_start"].dt.date
    growth_display["wow_growth_pct"] = growth_display["wow_growth_pct"].round(2)
    st.dataframe(growth_display[["week_start", "product", "weekly_predicted_demand", "wow_growth_pct"]], use_container_width=True)


def _inventory_management_page(df: pd.DataFrame) -> None:
    st.subheader("Inventory intelligence")
    st.caption("Stock recommendations based on predicted demand (simple planning helper).")

    if df.empty or "predicted_demand" not in df.columns or "date" not in df.columns or "product" not in df.columns:
        st.warning("No inventory data found. Generate data/processed/demand_predictions.csv first.")
        return

    latest = df.sort_values("date").groupby("product").tail(30).copy()
    latest["predicted_demand"] = pd.to_numeric(latest["predicted_demand"], errors="coerce")
    latest = latest.dropna(subset=["predicted_demand"])

    summary = (
        latest.groupby("product", as_index=False)["predicted_demand"]
        .mean()
        .rename(columns={"predicted_demand": "avg_daily_demand"})
    )

    # Demo stock numbers until real stock input is connected.
    rng = np.random.default_rng(42)
    summary["current_stock"] = rng.integers(10, 100, size=len(summary))

    summary["predicted_demand_7_days"] = (summary["avg_daily_demand"] * 7).round().astype("Int64")

    order_more_mask = summary["current_stock"] < summary["predicted_demand_7_days"].fillna(0)
    overstock_mask = summary["current_stock"] > (summary["predicted_demand_7_days"].fillna(0) * 1.5)

    summary["recommendation"] = "Maintain"
    summary.loc[order_more_mask, "recommendation"] = "Order more"
    summary.loc[overstock_mask, "recommendation"] = "Overstock risk"

    st.markdown("### Stock Recommendations")
    st.dataframe(
        summary.sort_values(["recommendation", "predicted_demand_7_days", "product"], ascending=[True, False, True]),
        use_container_width=True,
    )

    st.markdown("### Stock alerts")
    low_stock = summary[summary["recommendation"] == "Order more"]
    overstock = summary[summary["recommendation"] == "Overstock risk"]

    if not low_stock.empty:
        st.warning(f"Low stock alert: {len(low_stock)} products need to order more")
        for _, row in low_stock.sort_values("predicted_demand_7_days", ascending=False).iterrows():
            st.write(f"- {row['product']}")

    if not overstock.empty:
        st.error(f"Overstock risk: {len(overstock)} products")
        for _, row in overstock.sort_values("current_stock", ascending=False).iterrows():
            st.write(f"- {row['product']}")
def _customer_insights_section() -> None:
    st.subheader("💬 Customer Insights")
    st.caption("Analyze customer comments to detect buying intent.")

    # Simulated comments (replace later with real TikTok data)
    comments = [
        "How much is this?",
        "Do you deliver outside Nairobi?",
        "This is so cute!",
        "Is it available?",
        "I need this for my baby",
        "Do you have other colors?",
        "Where are you located?",
        "Can I order today?",
        "Is there a discount?",
        "Looks nice"
    ]

    df = pd.DataFrame({"comment": comments})

    # Simple intent classification
    def classify_intent(comment: str) -> str:
        comment = comment.lower()

        if any(x in comment for x in ["how much", "price", "cost"]):
            return "High Purchase Intent"
        elif any(x in comment for x in ["deliver", "location", "where"]):
            return " Ready to Buy"
        elif any(x in comment for x in ["need", "order", "available"]):
            return "Strong Intent"
        elif any(x in comment for x in ["cute", "nice", "love"]):
            return " Low Intent"
        else:
            return "Neutral"

    df["intent"] = df["comment"].apply(classify_intent)

    # Display table
    st.markdown("### Comment Analysis")
    st.dataframe(df, use_container_width=True)

    # Summary
    st.markdown("### Insights Summary")

    intent_counts = df["intent"].value_counts()

    for intent, count in intent_counts.items():
        st.write(f"{intent}: {count} comments")

    # Business insight
    high_intent = intent_counts.get("💰 High Purchase Intent", 0) + intent_counts.get("🛒 Strong Intent", 0)

    total = len(df)

    if total > 0:
        percent = (high_intent / total) * 100
        st.success(f"{percent:.1f}% of comments show strong buying intent")



def main() -> None:
    st.set_page_config(page_title="Nila Baby Shop - Demand Forecast", page_icon="DF", layout="wide")
    st.title("Nila Baby Shop Demand Forecast")
    _demand_forecast_page(_load_forecast_data())

    st.divider()
    _inventory_management_page(_load_inventory_data())

    st.divider()
    _customer_insights_section()


if __name__ == "__main__":
    main()
