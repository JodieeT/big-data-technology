import pandas as pd

from minvar_l2.covariance import sample_covariance, ewma_covariance
from minvar_l2.optimizer import min_variance_l2


def run_monthly_backtest(
    returns,
    lookback_weeks=26,
    lambda_=0.0,
    cov_method="sample",
    long_only=True,
):
    """
    Run monthly rebalanced minimum variance portfolio backtest.

    Parameters
    ----------
    returns : pd.DataFrame
        Weekly asset returns.
    lookback_weeks : int
        Number of past weekly observations used to estimate covariance.
    lambda_ : float
        L2 regularization strength.
    cov_method : str
        Covariance estimator: "sample" or "ewma".
    long_only : bool
        Whether to impose nonnegative portfolio weights.

    Returns
    -------
    pd.Series
        Backtested portfolio returns.
    pd.DataFrame
        Portfolio weights over time.
    """
    rebalance_dates = returns.resample("MS").first().index

    portfolio_returns = []
    weights_records = []

    for date in rebalance_dates:
        if date not in returns.index:
            available_dates = returns.index[returns.index >= date]
            if len(available_dates) == 0:
                continue
            date = available_dates[0]

        current_loc = returns.index.get_loc(date)

        if current_loc < lookback_weeks:
            continue

        lookback_data = returns.iloc[current_loc - lookback_weeks:current_loc]

        if cov_method == "sample":
            cov_matrix = sample_covariance(lookback_data)
        elif cov_method == "ewma":
            cov_matrix = ewma_covariance(lookback_data)
        else:
            raise ValueError("cov_method must be 'sample' or 'ewma'")

        weights = min_variance_l2(
            cov_matrix,
            lambda_=lambda_,
            long_only=long_only,
        )

        weights_records.append(
            pd.Series(weights, index=returns.columns, name=date)
        )

        next_month_end = date + pd.offsets.MonthEnd(1)
        period_returns = returns.loc[date:next_month_end]

        port_ret = period_returns @ weights
        portfolio_returns.append(port_ret)

    if len(portfolio_returns) == 0:
        raise ValueError("No backtest results generated. Check date range or lookback window.")

    portfolio_returns = pd.concat(portfolio_returns).sort_index()
    weights_df = pd.DataFrame(weights_records)

    return portfolio_returns, weights_df