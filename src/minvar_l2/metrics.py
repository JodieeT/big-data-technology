import numpy as np
import pandas as pd


def total_return(portfolio_returns):
    return (1 + portfolio_returns).prod() - 1


def annualized_return(portfolio_returns, periods_per_year=52):
    cumulative = (1 + portfolio_returns).prod()
    n_periods = len(portfolio_returns)
    return cumulative ** (periods_per_year / n_periods) - 1


def annualized_volatility(portfolio_returns, periods_per_year=52):
    return portfolio_returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(portfolio_returns, risk_free_rate=0.0, periods_per_year=52):
    excess_returns = portfolio_returns - risk_free_rate / periods_per_year
    vol = annualized_volatility(portfolio_returns, periods_per_year)

    if vol == 0:
        return np.nan

    return excess_returns.mean() * periods_per_year / vol


def max_drawdown(portfolio_returns):
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return drawdown.min()


def performance_summary(portfolio_returns, periods_per_year=52):
    """
    Compute core portfolio performance statistics.
    """
    return pd.Series({
        "Total Return": total_return(portfolio_returns),
        "Annualized Return": annualized_return(portfolio_returns, periods_per_year),
        "Annualized Volatility": annualized_volatility(portfolio_returns, periods_per_year),
        "Sharpe Ratio": sharpe_ratio(portfolio_returns, periods_per_year=periods_per_year),
        "Max Drawdown": max_drawdown(portfolio_returns),
    })