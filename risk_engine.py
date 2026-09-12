import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns."""
    return np.log(prices / prices.shift(1)).dropna()

def parametric_var(returns: pd.Series, conf: float, horizon: int) -> dict:
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - conf)
    # Scale to horizon
    mu_h = mu * horizon
    sigma_h = sigma * np.sqrt(horizon)
    var_pct = -(mu_h + z * sigma_h)
    # CVaR (Expected Shortfall) for normal distribution
    cvar_pct = -(mu_h - sigma_h * stats.norm.pdf(z) / (1 - conf))
    return {
        "var_pct": var_pct,
        "cvar_pct": cvar_pct,
        "mu": mu,
        "sigma": sigma,
        "mu_h": mu_h,
        "sigma_h": sigma_h,
        "z": z,
    }

def historical_var(returns: pd.Series, conf: float, horizon: int) -> dict:
    # Scale single-day returns to horizon via square root of time
    scaled = returns * np.sqrt(horizon)
    var_pct = -np.percentile(scaled, (1 - conf) * 100)
    tail = scaled[scaled <= -var_pct]
    cvar_pct = -tail.mean() if len(tail) > 0 else var_pct
    return {"var_pct": var_pct, "cvar_pct": cvar_pct}

def monte_carlo_var(returns: pd.Series, conf: float, horizon: int, n_sim: int = 10000) -> dict:
    mu = returns.mean()
    sigma = returns.std()
    # Simulate n_sim paths of `horizon` daily returns, sum = cumulative log return
    rng = np.random.default_rng(42)
    sim_daily = rng.normal(mu, sigma, size=(n_sim, horizon))
    sim_cumulative = sim_daily.sum(axis=1)  # log return over horizon
    var_pct = -np.percentile(sim_cumulative, (1 - conf) * 100)
    tail = sim_cumulative[sim_cumulative <= -var_pct]
    cvar_pct = -tail.mean() if len(tail) > 0 else var_pct
    return {"var_pct": var_pct, "cvar_pct": cvar_pct, "sim": sim_cumulative}

@st.cache_data(show_spinner=False)
def portfolio_var(returns: pd.DataFrame, weights: np.ndarray, conf: float, horizon: int) -> dict:
    """Portfolio VaR using correlation matrix."""
    mu_vec = returns.mean().values
    cov = returns.cov().values
    port_mu = weights @ mu_vec
    port_var_daily = weights @ cov @ weights
    port_sigma = np.sqrt(port_var_daily)
    # Scale to horizon
    port_mu_h = port_mu * horizon
    port_sigma_h = port_sigma * np.sqrt(horizon)
    z = stats.norm.ppf(1 - conf)
    var_pct = -(port_mu_h + z * port_sigma_h)
    cvar_pct = -(port_mu_h - port_sigma_h * stats.norm.pdf(z) / (1 - conf))
    # Naive VaR (no correlation) for diversification benefit
    individual_vars = []
    for i, col in enumerate(returns.columns):
        r = returns[col]
        v = -(r.mean() * horizon + z * r.std() * np.sqrt(horizon))
        individual_vars.append(weights[i] * v)
    naive_var = sum(individual_vars)
    div_benefit = naive_var - var_pct
    return {
        "var_pct": var_pct,
        "cvar_pct": cvar_pct,
        "port_mu": port_mu,
        "port_sigma": port_sigma,
        "port_mu_h": port_mu_h,
        "port_sigma_h": port_sigma_h,
        "naive_var_pct": naive_var,
        "div_benefit_pct": div_benefit,
        "z": z,
    }

def distribution_stats(returns: pd.Series) -> dict:
    """Skewness, kurtosis, normality test."""
    skew = float(returns.skew())
    kurt = float(returns.kurtosis())  # excess kurtosis (0 = normal)
    jb_stat, jb_p = stats.jarque_bera(returns.dropna())
    return {
        "skewness": skew,
        "excess_kurtosis": kurt,
        "jb_stat": jb_stat,
        "jb_pvalue": jb_p,
        "is_normal": jb_p > 0.05,
    }

def backtest_var(returns: pd.Series, var_pct: float, conf: float) -> dict:
    """Count breaches and compute Kupiec POF test."""
    breaches = (returns < -var_pct).sum()
    total = len(returns)
    breach_rate = breaches / total
    expected_rate = 1 - conf
    # Kupiec likelihood ratio test
    if breach_rate == 0 or breach_rate == 1:
        pof_pvalue = np.nan
    else:
        lr = -2 * (
            (total - breaches) * np.log(1 - expected_rate)
            + breaches * np.log(expected_rate)
            - (total - breaches) * np.log(1 - breach_rate)
            - breaches * np.log(breach_rate)
        )
        pof_pvalue = 1 - stats.chi2.cdf(lr, df=1)
    return {
        "breaches": int(breaches),
        "total": total,
        "breach_rate": breach_rate,
        "expected_rate": expected_rate,
        "pof_pvalue": pof_pvalue,
    }
