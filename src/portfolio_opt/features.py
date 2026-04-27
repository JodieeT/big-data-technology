"""
Feature engineering functions for the minimum variance portfolio project.

This module provides reusable functions for computing asset-level returns,
volatility, drawdown, and sector-level representative indicators.

The default input frequency is daily price data.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


Frequency = Literal["daily", "weekly", "monthly"]
ReturnMethod = Literal["simple", "log"]


FREQUENCY_MAP = {
    "daily": "D",
    "weekly": "W-FRI",
    "monthly": "M",
}

PERIODS_PER_YEAR = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
}


def validate_price_data(prices: pd.DataFrame) -> None:
    """
    Validate the input price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Price DataFrame with dates as index and asset tickers as columns.

    Raises
    ------
    TypeError
        If prices is not a pandas DataFrame.
    ValueError
        If prices is empty or does not have a DatetimeIndex.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")

    if prices.empty:
        raise ValueError("prices cannot be empty.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices index must be a pandas DatetimeIndex.")


def resample_prices(
    prices: pd.DataFrame,
    frequency: Frequency = "daily",
) -> pd.DataFrame:
    """
    Resample price data to the desired frequency.

    The default assumes that input prices are already daily. For weekly and
    monthly data, the function uses the last available price in each period.

    Parameters
    ----------
    prices : pd.DataFrame
        Price DataFrame with dates as index and asset tickers as columns.
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Target price frequency.

    Returns
    -------
    pd.DataFrame
        Resampled price DataFrame.
    """
    validate_price_data(prices)

    if frequency not in FREQUENCY_MAP:
        raise ValueError("frequency must be one of: 'daily', 'weekly', 'monthly'.")

    prices = prices.sort_index()

    if frequency == "daily":
        return prices.dropna(how="all")

    rule = FREQUENCY_MAP[frequency]

    return prices.resample(rule).last().dropna(how="all")


def compute_returns(
    prices: pd.DataFrame,
    frequency: Frequency = "daily",
    method: ReturnMethod = "simple",
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Compute asset returns from price data.

    Parameters
    ----------
    prices : pd.DataFrame
        Price DataFrame with dates as index and asset tickers as columns.
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Frequency used to resample prices before computing returns.
    method : {"simple", "log"}, default "simple"
        Return calculation method.
    dropna : bool, default True
        Whether to drop rows with missing return values.

    Returns
    -------
    pd.DataFrame
        Return DataFrame.
    """
    prices_resampled = resample_prices(prices, frequency=frequency)

    if method == "simple":
        returns = prices_resampled.pct_change()
    elif method == "log":
        returns = np.log(prices_resampled / prices_resampled.shift(1))
    else:
        raise ValueError("method must be either 'simple' or 'log'.")

    if dropna:
        returns = returns.dropna(how="all")

    return returns


def compute_rolling_volatility(
    returns: pd.DataFrame,
    window: int = 21,
    frequency: Frequency = "daily",
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling volatility from returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Return DataFrame with dates as index and asset tickers as columns.
    window : int, default 21
        Rolling window length.
        For daily returns, 21 is approximately one trading month.
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Frequency of the return data.
    annualize : bool, default True
        Whether to annualize the rolling volatility.

    Returns
    -------
    pd.DataFrame
        Rolling volatility DataFrame.
    """
    if frequency not in PERIODS_PER_YEAR:
        raise ValueError("frequency must be one of: 'daily', 'weekly', 'monthly'.")

    vol = returns.rolling(window=window).std()

    if annualize:
        vol = vol * np.sqrt(PERIODS_PER_YEAR[frequency])

    return vol


def compute_drawdown(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute drawdown series from returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Return DataFrame with dates as index and asset tickers as columns.

    Returns
    -------
    pd.DataFrame
        Drawdown DataFrame.
    """
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1

    return drawdown


def compute_max_drawdown(
    returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute maximum drawdown for each asset.

    Parameters
    ----------
    returns : pd.DataFrame
        Return DataFrame with dates as index and asset tickers as columns.

    Returns
    -------
    pd.Series
        Maximum drawdown for each asset.
    """
    drawdown = compute_drawdown(returns)

    return drawdown.min()


def compute_asset_features(
    prices: pd.DataFrame,
    frequency: Frequency = "daily",
    return_method: ReturnMethod = "simple",
    vol_window: int = 21,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Compute core asset-level features.

    Parameters
    ----------
    prices : pd.DataFrame
        Price DataFrame with dates as index and asset tickers as columns.
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Frequency used to resample prices before computing returns.
    return_method : {"simple", "log"}, default "simple"
        Return calculation method.
    vol_window : int, default 21
        Rolling volatility window.

    Returns
    -------
    dict
        Dictionary containing:
        - prices
        - returns
        - rolling_volatility
        - drawdown
        - max_drawdown
    """
    prices_resampled = resample_prices(prices, frequency=frequency)

    returns = compute_returns(
        prices_resampled,
        frequency="daily",
        method=return_method,
    )

    rolling_volatility = compute_rolling_volatility(
        returns,
        window=vol_window,
        frequency=frequency,
        annualize=True,
    )

    drawdown = compute_drawdown(returns)
    max_drawdown = compute_max_drawdown(returns)

    return {
        "prices": prices_resampled,
        "returns": returns,
        "rolling_volatility": rolling_volatility,
        "drawdown": drawdown,
        "max_drawdown": max_drawdown,
    }


def compute_sector_returns(
    returns: pd.DataFrame,
    sector_map: dict[str, str],
    method: Literal["equal_weighted", "median"] = "equal_weighted",
) -> pd.DataFrame:
    """
    Compute sector-level representative return indicators.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return DataFrame.
    sector_map : dict[str, str]
        Mapping from ticker to sector.
        Example:
        {"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials"}
    method : {"equal_weighted", "median"}, default "equal_weighted"
        Method used to compute sector representative returns.

    Returns
    -------
    pd.DataFrame
        Sector-level return indicators.
    """
    if method not in {"equal_weighted", "median"}:
        raise ValueError("method must be either 'equal_weighted' or 'median'.")

    available_tickers = [ticker for ticker in returns.columns if ticker in sector_map]

    if len(available_tickers) == 0:
        raise ValueError("No tickers in returns are found in sector_map.")

    sector_returns = {}

    sectors = sorted(set(sector_map[ticker] for ticker in available_tickers))

    for sector in sectors:
        tickers_in_sector = [
            ticker
            for ticker in available_tickers
            if sector_map[ticker] == sector
        ]

        sector_data = returns[tickers_in_sector]

        if method == "equal_weighted":
            sector_returns[sector] = sector_data.mean(axis=1)
        elif method == "median":
            sector_returns[sector] = sector_data.median(axis=1)

    return pd.DataFrame(sector_returns, index=returns.index)


def compute_sector_volatility(
    sector_returns: pd.DataFrame,
    window: int = 21,
    frequency: Frequency = "daily",
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling volatility for sector representative returns.

    Parameters
    ----------
    sector_returns : pd.DataFrame
        Sector-level return DataFrame.
    window : int, default 21
        Rolling volatility window.
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Frequency of sector return data.
    annualize : bool, default True
        Whether to annualize volatility.

    Returns
    -------
    pd.DataFrame
        Sector-level rolling volatility.
    """
    return compute_rolling_volatility(
        sector_returns,
        window=window,
        frequency=frequency,
        annualize=annualize,
    )


def compute_sector_drawdown(
    sector_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute drawdown for sector representative returns.

    Parameters
    ----------
    sector_returns : pd.DataFrame
        Sector-level return DataFrame.

    Returns
    -------
    pd.DataFrame
        Sector-level drawdown.
    """
    return compute_drawdown(sector_returns)


def compute_sector_features(
    prices: pd.DataFrame,
    sector_map: dict[str, str],
    frequency: Frequency = "daily",
    return_method: ReturnMethod = "simple",
    vol_window: int = 21,
    sector_method: Literal["equal_weighted", "median"] = "equal_weighted",
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Compute sector-level representative indicators from asset prices.

    Parameters
    ----------
    prices : pd.DataFrame
        Price DataFrame with dates as index and asset tickers as columns.
    sector_map : dict[str, str]
        Mapping from ticker to sector.
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Frequency used to resample prices before computing returns.
    return_method : {"simple", "log"}, default "simple"
        Return calculation method.
    vol_window : int, default 21
        Rolling volatility window.
    sector_method : {"equal_weighted", "median"}, default "equal_weighted"
        Method used to compute sector representative returns.

    Returns
    -------
    dict
        Dictionary containing:
        - sector_returns
        - sector_rolling_volatility
        - sector_drawdown
        - sector_max_drawdown
    """
    returns = compute_returns(
        prices,
        frequency=frequency,
        method=return_method,
    )

    sector_returns = compute_sector_returns(
        returns,
        sector_map=sector_map,
        method=sector_method,
    )

    sector_rolling_volatility = compute_sector_volatility(
        sector_returns,
        window=vol_window,
        frequency=frequency,
        annualize=True,
    )

    sector_drawdown = compute_sector_drawdown(sector_returns)
    sector_max_drawdown = compute_max_drawdown(sector_returns)

    return {
        "sector_returns": sector_returns,
        "sector_rolling_volatility": sector_rolling_volatility,
        "sector_drawdown": sector_drawdown,
        "sector_max_drawdown": sector_max_drawdown,
    }


def add_sector_representative_features(
    returns: pd.DataFrame,
    sector_map: dict[str, str],
    sector_method: Literal["equal_weighted", "median"] = "equal_weighted",
    prefix: str = "sector",
) -> pd.DataFrame:
    """
    Add each asset's corresponding sector representative return as a feature.

    This is useful when building asset-level models where each stock receives
    a sector-level contextual signal.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return DataFrame.
    sector_map : dict[str, str]
        Mapping from ticker to sector.
    sector_method : {"equal_weighted", "median"}, default "equal_weighted"
        Method used to compute sector representative returns.
    prefix : str, default "sector"
        Prefix for generated feature columns.

    Returns
    -------
    pd.DataFrame
        MultiIndex-column DataFrame. The first level is ticker, and the second
        level contains:
        - asset_return
        - sector_return
    """
    sector_returns = compute_sector_returns(
        returns,
        sector_map=sector_map,
        method=sector_method,
    )

    feature_blocks = {}

    for ticker in returns.columns:
        if ticker not in sector_map:
            continue

        sector = sector_map[ticker]

        feature_blocks[ticker] = pd.DataFrame({
            "asset_return": returns[ticker],
            f"{prefix}_return": sector_returns[sector],
        })

    if len(feature_blocks) == 0:
        raise ValueError("No asset-level sector features could be created.")

    features = pd.concat(feature_blocks, axis=1)

    return features


def make_feature_set(
    prices: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
    frequency: Frequency = "daily",
    return_method: ReturnMethod = "simple",
    vol_window: int = 21,
    sector_method: Literal["equal_weighted", "median"] = "equal_weighted",
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Create a full feature set for the project.

    Parameters
    ----------
    prices : pd.DataFrame
        Price DataFrame with dates as index and asset tickers as columns.
    sector_map : dict[str, str] or None, default None
        Optional mapping from ticker to sector.
    frequency : {"daily", "weekly", "monthly"}, default "daily"
        Frequency used to resample prices before computing returns.
    return_method : {"simple", "log"}, default "simple"
        Return calculation method.
    vol_window : int, default 21
        Rolling volatility window.
    sector_method : {"equal_weighted", "median"}, default "equal_weighted"
        Method used to compute sector representative returns.

    Returns
    -------
    dict
        Dictionary containing asset-level and, if sector_map is provided,
        sector-level features.
    """
    feature_set = compute_asset_features(
        prices=prices,
        frequency=frequency,
        return_method=return_method,
        vol_window=vol_window,
    )

    if sector_map is not None:
        sector_features = compute_sector_features(
            prices=prices,
            sector_map=sector_map,
            frequency=frequency,
            return_method=return_method,
            vol_window=vol_window,
            sector_method=sector_method,
        )

        feature_set.update(sector_features)

        feature_set["asset_sector_features"] = add_sector_representative_features(
            returns=feature_set["returns"],
            sector_map=sector_map,
            sector_method=sector_method,
        )

    return feature_set
