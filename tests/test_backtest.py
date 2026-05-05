import numpy as np
import pandas as pd
import pytest

from minvar_l2.backtest import run_monthly_backtest


# ── helpers ────────────────────────────────────────────────────────────────────

def make_returns(n_weeks=80, n_assets=3, seed=42):
    """Generate synthetic weekly returns with a monthly-frequency date index."""
    np.random.seed(seed)
    dates = pd.date_range("2019-01-07", periods=n_weeks, freq="W-MON")
    data = np.random.randn(n_weeks, n_assets) * 0.01
    cols = [chr(65 + i) for i in range(n_assets)]
    return pd.DataFrame(data, index=dates, columns=cols)


# ── return types ───────────────────────────────────────────────────────────────

def test_backtest_returns_series_and_dataframe():
    returns = make_returns()
    port_ret, weights = run_monthly_backtest(returns, lookback_weeks=26)
    assert isinstance(port_ret, pd.Series)
    assert isinstance(weights, pd.DataFrame)


def test_backtest_nonempty_output():
    returns = make_returns()
    port_ret, weights = run_monthly_backtest(returns, lookback_weeks=26)
    assert len(port_ret) > 0
    assert len(weights) > 0


# ── weight constraints ─────────────────────────────────────────────────────────

def test_backtest_weights_sum_to_one():
    returns = make_returns()
    _, weights = run_monthly_backtest(returns, lookback_weeks=26)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)


def test_backtest_long_only_weights_nonnegative():
    returns = make_returns()
    _, weights = run_monthly_backtest(returns, lookback_weeks=26, long_only=True)
    assert np.all(weights.values >= -1e-8)


def test_backtest_long_short_weights_sum_to_one():
    returns = make_returns()
    _, weights = run_monthly_backtest(returns, lookback_weeks=26, long_only=False)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)


# ── covariance methods ─────────────────────────────────────────────────────────

def test_backtest_sample_cov():
    returns = make_returns()
    port_ret, _ = run_monthly_backtest(returns, lookback_weeks=26, cov_method="sample")
    assert len(port_ret) > 0


def test_backtest_ewma_cov():
    returns = make_returns()
    port_ret, _ = run_monthly_backtest(returns, lookback_weeks=26, cov_method="ewma")
    assert len(port_ret) > 0


def test_backtest_invalid_cov_method_raises():
    returns = make_returns()
    with pytest.raises(ValueError):
        run_monthly_backtest(returns, lookback_weeks=26, cov_method="invalid")


# ── L2 regularization effect ───────────────────────────────────────────────────

def test_backtest_l2_zero_vs_nonzero():
    returns = make_returns()
    _, w_no_reg = run_monthly_backtest(returns, lookback_weeks=26, lambda_=0.0)
    _, w_reg = run_monthly_backtest(returns, lookback_weeks=26, lambda_=1.0)
    # Regularized weights should be more evenly spread (lower max weight)
    assert w_reg.max(axis=1).mean() <= w_no_reg.max(axis=1).mean() + 1e-6


# ── lookback window ────────────────────────────────────────────────────────────

def test_backtest_insufficient_lookback_skips_early_periods():
    returns = make_returns(n_weeks=80)
    # With lookback=60, fewer rebalance periods should pass the cutoff check
    _, w_short = run_monthly_backtest(returns, lookback_weeks=10)
    _, w_long = run_monthly_backtest(returns, lookback_weeks=60)
    assert len(w_long) <= len(w_short)


def test_backtest_raises_if_no_results():
    # Lookback longer than the entire series → no valid periods
    returns = make_returns(n_weeks=10)
    with pytest.raises(ValueError):
        run_monthly_backtest(returns, lookback_weeks=100)


# ── column alignment ───────────────────────────────────────────────────────────

def test_backtest_weight_columns_match_assets():
    returns = make_returns(n_assets=4)
    _, weights = run_monthly_backtest(returns, lookback_weeks=26)
    assert list(weights.columns) == list(returns.columns)


def test_backtest_portfolio_return_index_is_datetime():
    returns = make_returns()
    port_ret, _ = run_monthly_backtest(returns, lookback_weeks=26)
    assert isinstance(port_ret.index, pd.DatetimeIndex)
