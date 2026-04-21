from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nila_baby_shop.config import PROCESSED_DATA_DIR


def _build_sample_data() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=90, freq="D")
    products = ["Baby Romper", "Feeding Bottle", "Diapers", "Baby Shoes", "Blanket Set"]
    rng = np.random.default_rng(42)
    rows = []

    for day in dates:
        for product in products:
            base_views = rng.integers(80, 450)
            likes = int(base_views * rng.uniform(0.12, 0.38))
            comments = int(max(2, likes * rng.uniform(0.08, 0.22)))
            engagement = (likes + comments) / max(base_views, 1)
            sales = int(max(8, base_views * rng.uniform(0.06, 0.18)))
            price = float(rng.choice([650, 850, 1200, 1450, 1800]))
            stock = int(rng.integers(15, 220))
            predicted = int(max(5, sales * rng.uniform(0.9, 1.25)))
            rows.append(
                {
                    "date": day,
                    "product": product,
                    "views": base_views,
                    "likes": likes,
                    "comment_count": comments,
                    "engagement_rate": round(engagement, 4),
                    "sales": sales,
                    "predicted_sales": predicted,
                    "price_ksh": price,
                    "current_stock": stock,
                    "sample_comment": rng.choice(
                        [
                            "How much for this?",
                            "I need two for my twins",
                            "Looks nice but is it in stock?",
                            "I will buy this next week",
                            "Can you deliver to Nairobi?",
                            "So cute!",
                        ]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _load_dashboard_data() -> pd.DataFrame:
    predictions_path = PROCESSED_DATA_DIR / "demand_predictions.csv"
    base_path = PROCESSED_DATA_DIR / "base_features.csv"

    if predictions_path.exists():
        df = pd.read_csv(predictions_path)
    elif base_path.exists():
        df = pd.read_csv(base_path)
    else:
        df = _build_sample_data()

    df.columns = [c.strip() for c in df.columns]
    if "comments_count" in df.columns and "comment_count" not in df.columns:
        df = df.rename(columns={"comments_count": "comment_count"})

    if "date" not in df.columns:
        df["date"] = pd.date_range("2026-01-01", periods=len(df), freq="D")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").fillna(pd.Timestamp("2026-01-01"))

    for col, default in {
        "sales": 0,
        "views": 0,
        "likes": 0,
        "comment_count": 0,
        "engagement_rate": 0.0,
        "product": "Unknown Product",
        "price_ksh": 0.0,
        "current_stock": 40,
        "predicted_sales": 0,
    }.items():
        if col not in df.columns:
            df[col] = default

    if "demand_prediction" in df.columns and "predicted_sales" not in df.columns:
        df["predicted_sales"] = df["demand_prediction"]
    df["predicted_sales"] = df["predicted_sales"].fillna(df["sales"]).astype(float)

    if "viral_prob" not in df.columns:
        df["viral_prob"] = (
            0.45 * (df["likes"] / (df["views"].replace(0, 1)))
            + 0.55 * (df["comment_count"] / (df["likes"].replace(0, 1)))
        ).clip(0, 1)
    if "recommended_stock_units" not in df.columns:
        df["recommended_stock_units"] = (df["predicted_sales"] * 1.2).round().astype(int)

    df["stock_delta"] = df["recommended_stock_units"] - df["current_stock"]
    df["stock_status"] = np.where(
        df["stock_delta"] > 25,
        "Low Stock",
        np.where(df["stock_delta"] < -25, "Excess Stock", "Healthy"),
    )

    comments = df.get("sample_comment", "").fillna("").astype(str).str.lower()
    intent_terms = ["buy", "price", "deliver", "need", "order", "how much", "in stock"]
    df["buying_intent"] = comments.apply(
        lambda text: "High Intent" if any(term in text for term in intent_terms) else "Low Intent"
    )
    return df


def _metric_cards(df: pd.DataFrame) -> None:
    total_sales = int(df["sales"].sum())
    predicted_sales = int(df["predicted_sales"].sum())
    trending_row = df.groupby("product", as_index=False)["viral_prob"].mean().sort_values(
        "viral_prob", ascending=False
    )
    top_product = trending_row.iloc[0]["product"] if not trending_row.empty else "N/A"
    stock_alerts = int((df["stock_status"] != "Healthy").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"{total_sales:,}")
    col2.metric("Predicted Sales", f"{predicted_sales:,}")
    col3.metric("Trending Product", str(top_product))
    col4.metric("Stock Alerts", stock_alerts)


def _home_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Dashboard Overview")
    st.caption("Quick insights to decide what to stock, promote, or adjust today.")
    _metric_cards(df)

    sales_time = df.groupby("date", as_index=False)[["sales", "predicted_sales"]].sum()
    fig_sales = px.line(
        sales_time,
        x="date",
        y=["sales", "predicted_sales"],
        title="Sales Over Time",
        labels={"value": "Units", "variable": "Series"},
    )
    st.plotly_chart(fig_sales, use_container_width=True)

    trending = (
        df.groupby("product", as_index=False)[["sales", "viral_prob"]]
        .mean()
        .sort_values("viral_prob", ascending=False)
        .head(10)
    )
    fig_trending = px.bar(
        trending,
        x="product",
        y="sales",
        color="viral_prob",
        color_continuous_scale="Magma",
        title="Trending Products and Sales",
    )
    st.plotly_chart(fig_trending, use_container_width=True)

    fig_social = px.scatter(
        df,
        x="engagement_rate",
        y="sales",
        size="views",
        color="product",
        title="TikTok Engagement Impact on Sales",
        hover_data=["likes", "comment_count"],
    )
    st.plotly_chart(fig_social, use_container_width=True)


def _inventory_page(df: pd.DataFrame) -> None:
    st.subheader("Inventory")
    st.caption("Smart stock recommendations and low/excess stock alerts.")
    inventory_cols = [
        "product",
        "current_stock",
        "recommended_stock_units",
        "stock_delta",
        "stock_status",
        "sales",
        "predicted_sales",
    ]
    inv = (
        df[inventory_cols]
        .groupby("product", as_index=False)
        .agg(
            {
                "current_stock": "mean",
                "recommended_stock_units": "mean",
                "stock_delta": "mean",
                "sales": "sum",
                "predicted_sales": "sum",
                "stock_status": lambda s: s.value_counts().index[0],
            }
        )
    )
    inv["action"] = np.where(
        inv["stock_status"] == "Low Stock",
        "Restock",
        np.where(inv["stock_status"] == "Excess Stock", "Reduce Orders", "Maintain"),
    )
    st.dataframe(inv.round(1), use_container_width=True)

    alert_count = (inv["stock_status"] != "Healthy").sum()
    st.info(f"{alert_count} products currently need stock action.")


def _product_analytics_page(df: pd.DataFrame) -> None:
    st.subheader("Product Analytics")
    st.caption("Explore relationships between pricing, demand, and social engagement.")
    products = sorted(df["product"].dropna().unique().tolist())
    selected_products = st.multiselect("Filter products", options=products, default=products[:3])
    start_date, end_date = st.date_input(
        "Filter date range",
        [df["date"].min().date(), df["date"].max().date()],
    )

    filtered = df[
        (df["date"].dt.date >= start_date)
        & (df["date"].dt.date <= end_date)
        & (df["product"].isin(selected_products) if selected_products else True)
    ]
    if filtered.empty:
        st.warning("No data for selected filters.")
        return

    fig_price = px.scatter(
        filtered,
        x="price_ksh",
        y="sales",
        color="product",
        trendline="ols",
        title="Price vs Demand",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    fig_eng = px.scatter(
        filtered,
        x="engagement_rate",
        y="sales",
        color="product",
        size="likes",
        title="Engagement vs Sales",
    )
    st.plotly_chart(fig_eng, use_container_width=True)


def _forecast_page(df: pd.DataFrame) -> None:
    st.subheader("Demand Forecast")
    st.caption("Select a product to preview future demand.")
    product = st.selectbox("Product", sorted(df["product"].unique()))
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=45, value=21)

    product_df = df[df["product"] == product].sort_values("date")
    recent_mean = product_df["predicted_sales"].tail(14).mean()
    if np.isnan(recent_mean) or recent_mean <= 0:
        recent_mean = max(product_df["sales"].tail(14).mean(), 1)

    future_dates = pd.date_range(product_df["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    growth = np.linspace(0.98, 1.12, horizon)
    future_sales = np.maximum(1, recent_mean * growth)

    forecast_df = pd.DataFrame({"date": future_dates, "forecast_sales": future_sales})
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=product_df["date"], y=product_df["sales"], mode="lines", name="Historical Sales")
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast_sales"],
            mode="lines+markers",
            name="Forecast Sales",
        )
    )
    fig.update_layout(title=f"Future Predicted Sales - {product}", xaxis_title="Date", yaxis_title="Units")
    st.plotly_chart(fig, use_container_width=True)


def _trending_page(df: pd.DataFrame) -> None:
    st.subheader("Trending")
    st.caption("Products likely to go viral based on engagement behavior.")
    trending = (
        df.groupby("product", as_index=False)[["viral_prob", "engagement_rate", "sales", "views"]]
        .mean()
        .sort_values("viral_prob", ascending=False)
    )
    trending["viral_label"] = np.where(
        trending["viral_prob"] >= 0.65,
        "High Viral Potential",
        np.where(trending["viral_prob"] >= 0.4, "Moderate Potential", "Low Potential"),
    )
    st.dataframe(trending.round(3), use_container_width=True)
    fig = px.bar(
        trending.head(10),
        x="product",
        y="viral_prob",
        color="engagement_rate",
        title="Products Likely to Go Viral",
        color_continuous_scale="Sunset",
    )
    st.plotly_chart(fig, use_container_width=True)


def _customer_insights_page(df: pd.DataFrame) -> None:
    st.subheader("Customer Insights")
    st.caption("Comment analysis to detect buying intent and guide promotions.")
    insights = (
        df.groupby(["product", "buying_intent"], as_index=False)
        .size()
        .rename(columns={"size": "comment_count"})
    )
    st.dataframe(insights, use_container_width=True)

    fig = px.bar(
        insights,
        x="product",
        y="comment_count",
        color="buying_intent",
        barmode="group",
        title="Buying Intent from Comments",
    )
    st.plotly_chart(fig, use_container_width=True)

    high_intent_share = (
        (df["buying_intent"] == "High Intent").sum() / max(len(df), 1)
    ) * 100
    st.success(
        f"{high_intent_share:.1f}% of analyzed comments show buying intent. Promote top-demand items now."
    )


def main() -> None:
    st.set_page_config(page_title="Nila Baby Shop Dashboard", page_icon="🍼", layout="wide")
    st.title("Nila Baby Shop Insights Dashboard")
    df = _load_dashboard_data()

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Go to",
            [
                "Dashboard",
                "Inventory",
                "Products",
                "Forecast",
                "Trending",
                "Customer Insights",
            ],
        )

    page_map = {
        "Dashboard": _home_dashboard,
        "Inventory": _inventory_page,
        "Products": _product_analytics_page,
        "Forecast": _forecast_page,
        "Trending": _trending_page,
        "Customer Insights": _customer_insights_page,
    }
    page_map[page](df)


if __name__ == "__main__":
    main()
