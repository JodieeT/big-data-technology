"""
Covariance estimation functions for portfolio optimization.

This module provides reusable covariance estimators, including sample
covariance, exponentially weighted moving average covariance, and
Ledoit-Wolf shrinkage covariance.

The default return frequency is daily.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


Frequency = Literal["daily", "weekly", "monthly"]


PERIODS_PER_YEAR = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
}


def validate_returns(returns: pd.DataFrame) -> None:
    """
    Validate the input return DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        Return DataFrame with dates as index and asset tickers as columns.

    Raises
    ------
    TypeError
        If returns is not a pandas DataFrame.
    ValueError
        If returns is empty or has fewer than two columns.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("returns cannot be empty.")

    if returns.shape[1] < 2:
        raise ValueError("returns must contain at least two assets.")

    if not all(np.issubdtype(dtype, np.number) for dtype in returns.dtypes):
        raise ValueError("all return columns must be numeric.")


def clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Clean return data before covariance estimation.

    This function removes rows where all assets are missing and then drops
    remaining rows with missing or infinite values.

    Parameters
    ----------
    returns : pd.DataFrame
        Raw return DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned return DataFrame.
    """
    validate_returns(returns)

    cleaned = returns.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.dropna(axis=0, how="any")

    if cleaned.shape[0] < 2:
        raise ValueError("returns must contain at least two valid observations.")

    return cleaned


def annualization_factor(frequency: Frequency = "daily") -> int:
    """
    Return the annualization factor for a given data frequency.

    Parameters
    ----------
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Return frequency.

    Returns
    -------
    int
        Number of return periods per year.
    """
    if frequency not in PERIODS_PER_YEAR:
        raise ValueError("frequency must be one of: 'daily', 'weekly', 'monthly'.")

    return PERIODS_PER_YEAR[frequency]


def sample_covariance(
    returns: pd.DataFrame,
    frequency: Frequency = "daily",
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Estimate the sample covariance matrix.

    Parameters
    ----------
    returns : pd.DataFrame
        Return DataFrame with dates as index and asset tickers as columns.
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Return frequency.
    annualize : bool, default True
        Whether to annualize the covariance matrix.

    Returns
    -------
    pd.DataFrame
        Sample covariance matrix.
    """
    cleaned = clean_returns(returns)

    cov = cleaned.cov()

    if annualize:
        cov = cov * annualization_factor(frequency)

    return cov


def ewma_covariance(
    returns: pd.DataFrame,
    lambda_: float = 0.94,
    frequency: Frequency = "daily",
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Estimate the exponentially weighted moving average covariance matrix.

    The EWMA covariance estimator gives more weight to recent observations.

    Parameters
    ----------
    returns : pd.DataFrame
        Return DataFrame with dates as index and asset tickers as columns.
    lambda_ : float, default 0.94
        Decay factor. A larger value gives more persistent weights.
        Common daily choice is 0.94.
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Return frequency.
    annualize : bool, default True
        Whether to annualize the covariance matrix.

    Returns
    -------
    pd.DataFrame
        EWMA covariance matrix.
    """
    if not 0 < lambda_ < 1:
        raise ValueError("lambda_ must be between 0 and 1.")

    cleaned = clean_returns(returns)
    columns = cleaned.columns

    x = cleaned.to_numpy()
    n_obs, n_assets = x.shape

    weights = np.array([(1 - lambda_) * lambda_ ** i for i in range(n_obs - 1, -1, -1)])
    weights = weights / weights.sum()

    mean = np.average(x, axis=0, weights=weights)
    centered = x - mean

    cov_matrix = (centered * weights[:, None]).T @ centered

    if annualize:
        cov_matrix = cov_matrix * annualization_factor(frequency)

    return pd.DataFrame(cov_matrix, index=columns, columns=columns)


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
    frequency: Frequency = "daily",
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Estimate the Ledoit-Wolf shrinkage covariance matrix.

    Ledoit-Wolf shrinkage improves covariance estimation stability by shrinking
    the sample covariance matrix toward a structured target.

    Parameters
    ----------
    returns : pd.DataFrame
        Return DataFrame with dates as index and asset tickers as columns.
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Return frequency.
    annualize : bool, default True
        Whether to annualize the covariance matrix.

    Returns
    -------
    pd.DataFrame
        Ledoit-Wolf shrinkage covariance matrix.
    """
    cleaned = clean_returns(returns)
    columns = cleaned.columns

    model = LedoitWolf()
    model.fit(cleaned.to_numpy())

    cov_matrix = model.covariance_

    if annualize:
        cov_matrix = cov_matrix * annualization_factor(frequency)

    return pd.DataFrame(cov_matrix, index=columns, columns=columns)


def covariance_matrix(
    returns: pd.DataFrame,
    method: Literal["sample", "ewma", "ledoit_wolf"] = "sample",
    frequency: Frequency = "daily",
    annualize: bool = True,
    lambda_: float = 0.94,
) -> pd.DataFrame:
    """
    Estimate covariance matrix using the selected method.

    Parameters
    ----------
    returns : pd.DataFrame
        Return DataFrame with dates as index and asset tickers as columns.
    method : {"sample", "ewma", "ledoit_wolf"}, default "sample"
        Covariance estimation method.
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Return frequency.
    annualize : bool, default True
        Whether to annualize the covariance matrix.
    lambda_ : float, default 0.94
        EWMA decay factor. Only used when method="ewma".

    Returns
    -------
    pd.DataFrame
        Estimated covariance matrix.
    """
    if method == "sample":
        return sample_covariance(
            returns=returns,
            frequency=frequency,
            annualize=annualize,
        )

    if method == "ewma":
        return ewma_covariance(
            returns=returns,
            lambda_=lambda_,
            frequency=frequency,
            annualize=annualize,
        )

    if method == "ledoit_wolf":
        return ledoit_wolf_covariance(
            returns=returns,
            frequency=frequency,
            annualize=annualize,
        )

    raise ValueError("method must be one of: 'sample', 'ewma', 'ledoit_wolf'.")