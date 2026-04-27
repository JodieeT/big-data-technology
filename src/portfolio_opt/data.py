import yfinance as yf
import pandas as pd

def download_price_data(tickers, start, end, interval="1d"):
    """
    Download adjusted close price data from Yahoo Finance.

    Parameters
    ----------
    tickers : list[str]
        List of asset tickers.
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.
    interval : str
        Data frequency, default is daily.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices with dates as index and tickers as columns.
    """
    data = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    return prices.dropna()
