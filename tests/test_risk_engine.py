import numpy as np
import pandas as pd
from scipy import stats
import pytest
from risk_engine import (
    compute_returns,
    parametric_var,
    historical_var,
    monte_carlo_var,
    portfolio_var,
    distribution_stats,
    backtest_var,
)

def test_compute_returns_basic():
    prices = pd.DataFrame({"Asset": [100, 102, 101, 103, 105]})
    returns = compute_returns(prices)
    assert len(returns) == 4
    assert np.isclose(returns.iloc[0]["Asset"], np.log(102 / 100))
    assert np.isclose(returns.iloc[3]["Asset"], np.log(105 / 103))

def test_compute_returns_single_price():
    prices = pd.DataFrame({"Asset": [100]})
    returns = compute_returns(prices)
    assert returns.empty

def test_parametric_var_95_1day():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.02, 10000))
    res = parametric_var(returns, conf=0.95, horizon=1)
    
    assert "var_pct" in res
    assert res["var_pct"] > 0
    assert res["cvar_pct"] > res["var_pct"]
    assert np.isclose(res["var_pct"], 0.0329, atol=0.005)

def test_parametric_var_scales_sqrt_t():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.02, 10000))
    res1 = parametric_var(returns, conf=0.95, horizon=1)
    res4 = parametric_var(returns, conf=0.95, horizon=4)
    assert np.isclose(res4["var_pct"], res1["var_pct"] * 2, atol=0.01)

def test_historical_var_percentile():
    # uniform between -0.05 and 0.05
    returns = pd.Series(np.linspace(-0.05, 0.05, 1001))
    res = historical_var(returns, conf=0.90, horizon=1)
    # 10th percentile of [-0.05, 0.05] is -0.04
    assert np.isclose(res["var_pct"], 0.04)
    assert res["cvar_pct"] > res["var_pct"]

def test_historical_var_all_positive():
    returns = pd.Series([0.01, 0.02, 0.03])
    res = historical_var(returns, conf=0.95, horizon=1)
    assert res["var_pct"] <= 0 # VaR is defined as negative returns in this context, so a guaranteed positive return means VaR <= 0

def test_monte_carlo_var_statistical():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.02, 10000))
    res_param = parametric_var(returns, conf=0.95, horizon=1)
    res_mc = monte_carlo_var(returns, conf=0.95, horizon=1, n_sim=10000)
    assert np.isclose(res_param["var_pct"], res_mc["var_pct"], rtol=0.2)

def test_monte_carlo_reproducibility():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.02, 100))
    res1 = monte_carlo_var(returns, conf=0.95, horizon=1, n_sim=100)
    res2 = monte_carlo_var(returns, conf=0.95, horizon=1, n_sim=100)
    assert res1["var_pct"] == res2["var_pct"]

def test_portfolio_var_single_asset():
    np.random.seed(42)
    returns = pd.DataFrame({"A": np.random.normal(0, 0.02, 1000)})
    res = portfolio_var(returns, weights=np.array([1.0]), conf=0.95, horizon=1)
    res_param = parametric_var(returns["A"], conf=0.95, horizon=1)
    assert np.isclose(res["var_pct"], res_param["var_pct"])

def test_portfolio_var_diversification():
    np.random.seed(42)
    # two uncorrelated assets
    returns = pd.DataFrame({
        "A": np.random.normal(0, 0.02, 10000),
        "B": np.random.normal(0, 0.02, 10000)
    })
    res = portfolio_var(returns, weights=np.array([0.5, 0.5]), conf=0.95, horizon=1)
    assert res["div_benefit_pct"] > 0
    assert res["var_pct"] < res["naive_var_pct"]

def test_portfolio_var_perfect_correlation():
    np.random.seed(42)
    series = np.random.normal(0, 0.02, 1000)
    returns = pd.DataFrame({"A": series, "B": series})
    res = portfolio_var(returns, weights=np.array([0.5, 0.5]), conf=0.95, horizon=1)
    assert np.isclose(res["div_benefit_pct"], 0, atol=1e-5)

def test_distribution_stats_normal():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 1, 10000))
    res = distribution_stats(returns)
    assert np.isclose(res["skewness"], 0, atol=0.1)
    assert np.isclose(res["excess_kurtosis"], 0, atol=0.2)
    assert res["is_normal"] == True

def test_distribution_stats_skewed():
    np.random.seed(42)
    returns = pd.Series(np.random.exponential(1, 10000))
    res = distribution_stats(returns)
    assert res["skewness"] > 0
    assert res["is_normal"] == False

def test_backtest_var_no_breaches():
    returns = pd.Series(np.random.uniform(0, 0.05, 100))
    res = backtest_var(returns, var_pct=0.10, conf=0.95)
    assert res["breaches"] == 0
    assert np.isnan(res["pof_pvalue"])

def test_backtest_var_expected_breaches():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.02, 10000))
    # True VaR at 95% is approx 3.29%
    var_pct = -np.percentile(returns, 5)
    res = backtest_var(returns, var_pct=var_pct, conf=0.95)
    assert np.isclose(res["breach_rate"], 0.05, atol=0.01)
