import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from minvar_l2.data import compute_returns, download_price_data


# ── compute_returns ────────────────────────────────────────────────────────────

def test_compute_returns_shape():
    prices = pd.DataFrame({
        "A": [100, 110, 121],
        "B": [200, 220, 242],
    })
    returns = compute_returns(prices)
    assert returns.shape == (2, 2)


def test_compute_returns_values():
    prices = pd.DataFrame({
        "A": [100, 110, 121],
        "B": [200, 220, 242],
    })
    returns = compute_returns(prices)
    assert abs(returns.iloc[0, 0] - 0.10) < 1e-8
    assert abs(returns.iloc[1, 0] - 0.10) < 1e-8


def test_compute_returns_no_nan():
    prices = pd.DataFrame({
        "A": [100, 105, 110, 115],
        "B": [50, 55, 60, 65],
    })
    returns = compute_returns(prices)
    assert not returns.isnull().any().any()


def test_compute_returns_single_column():
    prices = pd.DataFrame({"A": [100, 110, 121]})
    returns = compute_returns(prices)
    assert returns.shape == (2, 1)
    assert abs(returns.iloc[0, 0] - 0.10) < 1e-8


def test_compute_returns_drops_first_row():
    # pct_change produces NaN on row 0, which dropna removes
    prices = pd.DataFrame({"A": [100, 200]})
    returns = compute_returns(prices)
    assert len(returns) == 1


# ── download_price_data (mocked) ───────────────────────────────────────────────
def _make_mock_download(tickers):
    """Return a mock prices DataFrame resembling yfinance output."""
    dates = pd.date_range("2023-01-01", periods=5, freq="W-MON")
    if len(tickers) == 1:
        # single ticker: flat columns
        df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates)
        return df
    else:
        # multi-ticker: MultiIndex columns
        arrays = [["Close"] * len(tickers), tickers]
        cols = pd.MultiIndex.from_arrays(arrays)
        data = np.tile(np.arange(100, 105).reshape(-1, 1), len(tickers))
        return pd.DataFrame(data, index=dates, columns=cols)


def test_download_price_data_multi_ticker():
    tickers = ["AAPL", "MSFT"]
    mock_df = _make_mock_download(tickers)

    with patch("minvar_l2.data.yf.download", return_value=mock_df):
        prices = download_price_data(tickers, start="2023-01-01", end="2023-03-01")

    assert isinstance(prices, pd.DataFrame)
    assert set(prices.columns) == set(tickers)
    assert not prices.isnull().any().any()


def test_download_price_data_single_ticker():
    tickers = ["AAPL"]
    mock_df = _make_mock_download(tickers)

    with patch("minvar_l2.data.yf.download", return_value=mock_df):
        prices = download_price_data(tickers, start="2023-01-01", end="2023-03-01")

    assert isinstance(prices, pd.DataFrame)
    assert list(prices.columns) == ["AAPL"]


def test_download_price_data_drops_na():
    tickers = ["AAPL", "MSFT"]
    dates = pd.date_range("2023-01-01", periods=5, freq="W-MON")
    arrays = [["Close", "Close"], tickers]
    cols = pd.MultiIndex.from_arrays(arrays)
    data = np.array([[100, 200], [np.nan, 201], [102, 202], [103, 203], [104, 204]])
    mock_df = pd.DataFrame(data, index=dates, columns=cols)

    with patch("minvar_l2.data.yf.download", return_value=mock_df):
        prices = download_price_data(tickers, start="2023-01-01", end="2023-03-01")

    assert not prices.isnull().any().any()
