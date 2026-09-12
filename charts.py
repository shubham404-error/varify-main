import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import pandas as pd

COLORS = {
    "primary":   "#1f77b4",
    "danger":    "#d62728",
    "warning":   "#ff7f0e",
    "success":   "#2ca02c",
    "muted":     "#adb5bd",
    "tail_fill": "rgba(214,39,40,0.18)",
    "dist_fill": "rgba(31,119,180,0.12)",
}

def plot_intraday(intraday_df: pd.DataFrame) -> go.Figure:
    """Plot intraday price movement for selected stocks."""
    fig = go.Figure()
    n = len(intraday_df.columns)
    palette = px.colors.qualitative.Set2

    for i, col in enumerate(intraday_df.columns):
        s = intraday_df[col].dropna()
        if s.empty:
            continue
        if n == 1:
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines", name=col,
                line=dict(color=COLORS["primary"], width=2),
                fill="tozeroy", fillcolor=COLORS["dist_fill"],
            ))
        else:
            pct = (s / s.iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(
                x=pct.index, y=pct.values, mode="lines", name=col,
                line=dict(color=palette[i % len(palette)], width=2),
            ))

    if n > 1:
        fig.add_hline(y=0, line_color="#adb5bd", line_width=1, line_dash="dot")

    fig.update_layout(
        title=dict(
            text="Intraday price" if n == 1 else "Intraday price movement (% change)",
            font_size=14,
        ),
        xaxis_title="",
        yaxis_title="Price" if n == 1 else "Change (%)",
        height=320,
        margin=dict(t=50, b=40, l=50, r=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(
            gridcolor="#f0f0f0",
            ticksuffix="%" if n > 1 else "",
            tickformat=",.2f" if n == 1 else "",
        ),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    return fig

def plot_distribution(returns: pd.Series, var_pct: float, cvar_pct: float,
                       conf: float, method: str, ticker_label: str) -> go.Figure:
    """Normal distribution + histogram with VaR cutoff."""
    r = returns.values
    mu, sigma = r.mean(), r.std()
    cutoff = -var_pct
    # Ensure x_range always covers the cutoff even at extreme confidence levels
    x_min = min(mu - 4.5 * sigma, cutoff - 0.5 * sigma)
    x_max = mu + 4.5 * sigma
    x_range = np.linspace(x_min, x_max, 500)
    pdf_vals = stats.norm.pdf(x_range, mu, sigma)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Histogram of actual returns (secondary y)
    fig.add_trace(go.Histogram(
        x=r,
        nbinsx=60,
        name="Actual returns",
        marker_color=COLORS["primary"],
        opacity=0.35,
        showlegend=True,
    ), secondary_y=True)

    # Normal PDF curve
    fig.add_trace(go.Scatter(
        x=x_range, y=pdf_vals,
        mode="lines",
        name="Normal fit",
        line=dict(color=COLORS["primary"], width=2.5),
        fill="tozeroy",
        fillcolor=COLORS["dist_fill"],
    ), secondary_y=False)

    # Shaded tail — guard against empty array at extreme confidence levels
    x_tail = x_range[x_range <= cutoff]
    if len(x_tail) >= 2:
        y_tail = stats.norm.pdf(x_tail, mu, sigma)
        fig.add_trace(go.Scatter(
            x=np.concatenate([[x_tail[0]], x_tail, [x_tail[-1]]]),
            y=np.concatenate([[0], y_tail, [0]]),
            fill="toself",
            fillcolor=COLORS["tail_fill"],
            line=dict(width=0),
            name=f"Loss tail ({(1-conf)*100:.0f}%)",
            showlegend=True,
        ), secondary_y=False)
    else:
        fig.add_trace(go.Scatter(
            x=[cutoff], y=[0],
            mode="markers",
            marker=dict(size=0, opacity=0),
            name=f"Loss tail ({(1-conf)*100:.0f}%)",
            showlegend=True,
        ), secondary_y=False)

    # Glowing VaR cutoff line
    fig.add_vline(x=cutoff, line_color="rgba(214,39,40,0.15)", line_width=10, layer="below")
    fig.add_vline(x=cutoff, line_color="rgba(214,39,40,0.3)", line_width=5, layer="below")
    fig.add_vline(
        x=cutoff,
        line_dash="dash",
        line_color=COLORS["danger"],
        line_width=2.5,
        annotation_text=f"  VaR cutoff<br>  {cutoff*100:.2f}%",
        annotation_font=dict(color=COLORS["danger"], size=13),
        annotation_position="top right",
    )

    # CVaR line
    fig.add_vline(
        x=-cvar_pct,
        line_dash="dot",
        line_color=COLORS["warning"],
        line_width=1.5,
        annotation_text=f"  CVaR<br>  {-cvar_pct*100:.2f}%",
        annotation_font=dict(color=COLORS["warning"], size=11),
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=dict(text=f"Return distribution — {ticker_label} ({method})", font_size=14),
        xaxis_title="Daily log return",
        yaxis_title="Probability density",
        yaxis2_title="Frequency",
        legend=dict(orientation="h", y=1.08),
        height=420,
        margin=dict(t=70, b=40, l=50, r=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(tickformat=".1%", gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0"),
    )
    # Format x-axis as percentage
    fig.update_xaxes(tickformat=".2%")
    return fig

def plot_price_with_var(prices: pd.Series, returns: pd.Series,
                         var_pct: float, position_size: float,
                         ticker: str) -> go.Figure:
    """Price chart with rolling VaR bands (30-day window)."""
    roll = returns.rolling(30)
    roll_mu = roll.mean()
    roll_sigma = roll.std()
    z = stats.norm.ppf(0.05)  # 95% conf implied
    roll_var = -(roll_mu + z * roll_sigma)

    # Align with prices index
    aligned = roll_var.reindex(prices.index).bfill()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=prices.index, y=prices,
        mode="lines",
        name=ticker,
        line=dict(color=COLORS["primary"], width=2),
    ))

    # Rolling VaR floor
    var_floor = prices * (1 - aligned)
    fig.add_trace(go.Scatter(
        x=prices.index, y=var_floor,
        mode="lines",
        name="Rolling VaR floor (95%)",
        line=dict(color=COLORS["danger"], width=1.5, dash="dash"),
        fill=None,
    ))

    fig.add_trace(go.Scatter(
        x=prices.index, y=prices,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        fill="tonexty",
        fillcolor="rgba(214,39,40,0.07)",
    ))

    fig.update_layout(
        title=dict(text=f"{ticker} — Price with rolling VaR floor", font_size=14),
        xaxis_title="Date",
        yaxis_title="Price",
        height=360,
        margin=dict(t=60, b=40, l=60, r=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0"),
        legend=dict(orientation="h", y=1.08),
    )
    return fig

def plot_correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    corr = returns.corr()
    labels = corr.columns.tolist()
    z = corr.values.round(3)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#d62728"],
            [0.5, "#ffffff"],
            [1.0, "#1f77b4"],
        ],
        zmin=-1, zmax=1,
        text=z,
        texttemplate="%{text}",
        textfont=dict(size=12),
        showscale=True,
        colorbar=dict(title="ρ", len=0.8),
    ))

    fig.update_layout(
        title=dict(text="Correlation matrix (daily log returns)", font_size=14),
        height=max(300, 80 * len(labels) + 80),
        margin=dict(t=60, b=40, l=80, r=40),
        xaxis=dict(side="bottom"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig

def plot_mc_histogram(sim: np.ndarray, var_pct: float, cvar_pct: float,
                       conf: float) -> go.Figure:
    cutoff = -var_pct
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=sim,
        nbinsx=80,
        marker_color=COLORS["primary"],
        opacity=0.6,
        name="Simulated returns",
    ))
    # Glowing VaR cutoff line
    fig.add_vline(x=cutoff, line_color="rgba(214,39,40,0.15)", line_width=10, layer="below")
    fig.add_vline(x=cutoff, line_color="rgba(214,39,40,0.3)", line_width=5, layer="below")
    fig.add_vline(x=cutoff, line_color=COLORS["danger"], line_dash="dash",
                  line_width=2.5,
                  annotation_text=f"  VaR {cutoff*100:.2f}%",
                  annotation_font=dict(color=COLORS["danger"], size=13))
    fig.add_vline(x=-cvar_pct, line_color=COLORS["warning"], line_dash="dot",
                  line_width=1.5,
                  annotation_text=f"  CVaR {-cvar_pct*100:.2f}%",
                  annotation_font_color=COLORS["warning"])
    fig.update_layout(
        title=dict(text=f"Monte Carlo simulation — 10,000 paths ({conf*100:.0f}% conf.)", font_size=14),
        xaxis_title="Simulated cumulative return",
        yaxis_title="Frequency",
        height=380,
        margin=dict(t=60, b=40, l=60, r=30),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickformat=".2%", gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0"),
    )
    return fig

def plot_backtest(returns: pd.Series, var_pct: float, ticker: str) -> go.Figure:
    cutoff = -var_pct
    colors = ["red" if r < cutoff else COLORS["primary"] for r in returns]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=returns.index,
        y=returns.values,
        marker_color=colors,
        name="Daily return",
        showlegend=False,
    ))
    fig.add_hline(y=cutoff, line_color=COLORS["danger"], line_dash="dash",
                  line_width=2,
                  annotation_text=f"  VaR threshold {cutoff*100:.2f}%",
                  annotation_font_color=COLORS["danger"])
    fig.update_layout(
        title=dict(text=f"{ticker} — VaR breach backtesting", font_size=14),
        xaxis_title="Date",
        yaxis_title="Daily log return",
        height=340,
        margin=dict(t=60, b=40, l=60, r=30),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(tickformat=".2%", gridcolor="#f0f0f0"),
        xaxis=dict(gridcolor="#f0f0f0"),
    )
    return fig

def plot_weights_pie(weights: list, labels: list) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=labels,
        values=weights,
        hole=0.5,
        textinfo="label+percent",
        marker=dict(colors=px.colors.qualitative.Set2),
    ))
    fig.update_layout(
        title=dict(text="Portfolio weights", font_size=13),
        height=300,
        margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig
