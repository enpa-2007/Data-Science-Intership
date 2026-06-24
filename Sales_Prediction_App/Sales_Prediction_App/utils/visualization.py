import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

PALETTE = {
    "primary": "#00D4FF",
    "secondary": "#7B2FFF",
    "accent": "#FF6B35",
    "success": "#00FF9F",
    "warning": "#FFD700",
    "bg": "#0A0E1A",
    "card": "#111827",
    "text": "#E2E8F0",
}

PLOTLY_TEMPLATE = "plotly_dark"

def correlation_heatmap(df):
    corr = df.corr()
    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        title="Correlation Heatmap", template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["card"],
        font_color=PALETTE["text"], height=400,
    )
    return fig

def sales_distribution(df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["Sales"], nbinsx=25,
        marker_color=PALETTE["primary"], opacity=0.85, name="Sales"
    ))
    fig.update_layout(
        title="Sales Distribution", xaxis_title="Sales (units)",
        yaxis_title="Count", template=PLOTLY_TEMPLATE,
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["card"],
        font_color=PALETTE["text"], height=350,
    )
    return fig

def channel_vs_sales(df, channel):
    colors = {"TV": PALETTE["primary"], "Radio": PALETTE["secondary"], "Newspaper": PALETTE["accent"]}
    fig = px.scatter(
        df, x=channel, y="Sales", trendline="ols",
        title=f"{channel} Budget vs Sales",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[colors.get(channel, PALETTE["primary"])],
    )
    fig.update_layout(
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["card"],
        font_color=PALETTE["text"], height=350,
    )
    return fig

def pairplot_figure(df):
    fig = px.scatter_matrix(
        df, dimensions=["TV", "Radio", "Newspaper", "Sales"],
        color="Sales", color_continuous_scale="Viridis",
        title="Pairplot Matrix", template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.update_layout(
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["card"],
        font_color=PALETTE["text"], height=500,
    )
    return fig

def model_comparison_chart(results):
    names = list(results.keys())
    r2_vals = [results[n]["R2 Score"] for n in names]
    rmse_vals = [results[n]["RMSE"] for n in names]
    fig = go.Figure(data=[
        go.Bar(name="R² Score", x=names, y=r2_vals, marker_color=PALETTE["primary"]),
        go.Bar(name="RMSE", x=names, y=rmse_vals, marker_color=PALETTE["accent"]),
    ])
    fig.update_layout(
        barmode="group", title="Model Performance Comparison",
        template=PLOTLY_TEMPLATE, paper_bgcolor=PALETTE["card"],
        plot_bgcolor=PALETTE["card"], font_color=PALETTE["text"], height=380,
    )
    return fig

def actual_vs_predicted(y_test, y_pred, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(y_test), y=list(y_pred), mode="markers",
        marker=dict(color=PALETTE["primary"], size=8, opacity=0.7),
        name="Predictions",
    ))
    mn = min(min(y_test), min(y_pred))
    mx = max(max(y_test), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines",
        line=dict(color=PALETTE["accent"], dash="dash", width=2),
        name="Perfect Fit",
    ))
    fig.update_layout(
        title=f"Actual vs Predicted — {model_name}",
        xaxis_title="Actual Sales", yaxis_title="Predicted Sales",
        template=PLOTLY_TEMPLATE, paper_bgcolor=PALETTE["card"],
        plot_bgcolor=PALETTE["card"], font_color=PALETTE["text"], height=400,
    )
    return fig

def residual_plot(y_test, y_pred, model_name):
    residuals = np.array(y_test) - np.array(y_pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(y_pred), y=list(residuals), mode="markers",
        marker=dict(color=PALETTE["secondary"], size=8, opacity=0.7), name="Residuals",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["accent"])
    fig.update_layout(
        title=f"Residual Plot — {model_name}",
        xaxis_title="Predicted Sales", yaxis_title="Residuals",
        template=PLOTLY_TEMPLATE, paper_bgcolor=PALETTE["card"],
        plot_bgcolor=PALETTE["card"], font_color=PALETTE["text"], height=380,
    )
    return fig

def feature_importance_chart(importances):
    features = list(importances.keys())
    values = list(importances.values())
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]
    fig = go.Figure(go.Bar(
        x=features, y=values,
        marker_color=colors[:len(features)],
        text=[f"{v:.1%}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="Feature Importance", xaxis_title="Channel",
        yaxis_title="Importance", template=PLOTLY_TEMPLATE,
        paper_bgcolor=PALETTE["card"], plot_bgcolor=PALETTE["card"],
        font_color=PALETTE["text"], height=350,
    )
    return fig

def budget_vs_sales_3d(df):
    fig = px.scatter_3d(
        df, x="TV", y="Radio", z="Sales",
        color="Newspaper", color_continuous_scale="Viridis",
        title="TV & Radio Budget vs Sales (3D)", template=PLOTLY_TEMPLATE,
        size_max=10,
    )
    fig.update_layout(
        paper_bgcolor=PALETTE["card"], font_color=PALETTE["text"], height=500,
    )
    return fig

def roi_chart(tv, radio, newspaper, predicted_sales):
    total_budget = tv + radio + newspaper
    roi = ((predicted_sales * 1000 - total_budget) / total_budget * 100) if total_budget > 0 else 0
    channels = ["TV", "Radio", "Newspaper"]
    budgets = [tv, radio, newspaper]
    fig = go.Figure(go.Pie(
        labels=channels, values=budgets,
        marker_colors=[PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]],
        hole=0.4, textinfo="label+percent",
    ))
    fig.update_layout(
        title=f"Budget Allocation | Est. ROI: {roi:.1f}%",
        template=PLOTLY_TEMPLATE, paper_bgcolor=PALETTE["card"],
        font_color=PALETTE["text"], height=350,
    )
    return fig
