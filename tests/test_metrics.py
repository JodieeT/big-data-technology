import numpy as np
import pandas as pd
import pytest

from minvar_l2.metrics import (
    total_return,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    performance_summary,
)


# ── total_return ───────────────────────────────────────────────────────────────

def test_total_return_basic():
    returns = pd.Series([0.1, -0.05, 0.02])
    expected = (1.1 * 0.95 * 1.02) - 1
    assert np.isclose(total_return(returns), expected)


def test_total_return_all_positive():
    returns = pd.Series([0.01, 0.02, 0.03])
    assert total_return(returns) > 0


def test_total_return_zero_returns():
    returns = pd.Series([0.0, 0.0, 0.0])
    assert np.isclose(total_return(returns), 0.0)


def test_total_return_single_period():
    returns = pd.Series([0.05])
    assert np.isclose(total_return(returns), 0.05)


# ── annualized_return ──────────────────────────────────────────────────────────

def test_annualized_return_positive():
    returns = pd.Series([0.01] * 52)
    result = annualized_return(returns, periods_per_year=52)
    assert result > 0


def test_annualized_return_scaling():
    # 52 weeks of 1% return → cumulative = 1.01^52
    returns = pd.Series([0.01] * 52)
    expected = (1.01 ** 52) - 1
    result = annualized_return(returns, periods_per_year=52)
    assert np.isclose(result, expected, rtol=1e-6)


def test_annualized_return_custom_periods():
    returns = pd.Series([0.005] * 12)
    result_monthly = annualized_return(returns, periods_per_year=12)
    result_weekly = annualized_return(returns, periods_per_year=52)
    # Different period assumptions → different annualized returns
    assert not np.isclose(result_monthly, result_weekly)


# ── annualized_volatility ──────────────────────────────────────────────────────

def test_annualized_volatility_positive():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01])
    vol = annualized_volatility(returns, periods_per_year=52)
    assert vol > 0


def test_annualized_volatility_zero_for_constant():
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    vol = annualized_volatility(returns, periods_per_year=52)
    assert np.isclose(vol, 0.0)


def test_annualized_volatility_scales_with_sqrt():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.005, -0.015])
    vol_weekly = annualized_volatility(returns, periods_per_year=52)
    vol_daily = annualized_volatility(returns, periods_per_year=252)
    # ratio should be sqrt(252/52)
    assert np.isclose(vol_daily / vol_weekly, np.sqrt(252 / 52), rtol=1e-6)


# ── sharpe_ratio ───────────────────────────────────────────────────────────────

def test_sharpe_ratio_positive_for_good_returns():
    np.random.seed(42)
    returns = pd.Series(0.01 + np.random.randn(52) * 0.001)
    sr = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=52)
    assert sr > 0


def test_sharpe_ratio_zero_vol_returns_nan():
    returns = pd.Series([0.0, 0.0, 0.0])
    sr = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=52)
    assert np.isnan(sr)


def test_sharpe_ratio_with_risk_free_rate():
    returns = pd.Series([0.01, 0.02, 0.03, 0.04] * 5)
    sr_no_rf = sharpe_ratio(returns, risk_free_rate=0.0)
    sr_with_rf = sharpe_ratio(returns, risk_free_rate=0.05)
    # Higher risk-free rate → lower Sharpe
    assert sr_no_rf > sr_with_rf


def test_sharpe_ratio_negative_for_bad_returns():
    returns = pd.Series([-0.01, -0.02, -0.01, -0.03])
    sr = sharpe_ratio(returns, risk_free_rate=0.0)
    assert sr < 0


# ── max_drawdown ───────────────────────────────────────────────────────────────

def test_max_drawdown_no_loss():
    returns = pd.Series([0.01, 0.02, 0.03])
    assert np.isclose(max_drawdown(returns), 0.0)


def test_max_drawdown_with_loss():
    returns = pd.Series([0.1, -0.2, 0.05])
    result = max_drawdown(returns)
    assert result < 0


def test_max_drawdown_always_nonpositive():
    np.random.seed(0)
    returns = pd.Series(np.random.randn(100) * 0.01)
    assert max_drawdown(returns) <= 0


def test_max_drawdown_known_value():
    # Goes up 10%, then drops 20%: drawdown = (1.1 * 0.8) / 1.1 - 1 = -0.2
    returns = pd.Series([0.1, -0.2])
    dd = max_drawdown(returns)
    assert np.isclose(dd, -0.2, atol=1e-8)


def test_max_drawdown_all_negative():
    returns = pd.Series([-0.05, -0.05, -0.05])
    dd = max_drawdown(returns)
    assert dd < -0.05  # cumulative loss > 5%


# ── performance_summary ────────────────────────────────────────────────────────

def test_performance_summary_contains_all_keys():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    summary = performance_summary(returns)
    for key in ["Total Return", "Annualized Return", "Annualized Volatility",
                "Sharpe Ratio", "Max Drawdown"]:
        assert key in summary.index


def test_performance_summary_returns_series():
    returns = pd.Series([0.01, 0.02, -0.01])
    summary = performance_summary(returns)
    assert isinstance(summary, pd.Series)


def test_performance_summary_values_finite():
    np.random.seed(1)
    returns = pd.Series(np.random.randn(52) * 0.01)
    summary = performance_summary(returns)
    # All values except Sharpe (which could be NaN if vol=0) should be finite
    for key in ["Total Return", "Annualized Return", "Annualized Volatility", "Max Drawdown"]:
        assert np.isfinite(summary[key])
