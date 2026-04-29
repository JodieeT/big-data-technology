from data import download_price_data, compute_returns
from backtest import run_monthly_backtest
from metrics import performance_summary


def main():
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "XOM"]

    prices = download_price_data(
        tickers=tickers,
        start="2018-01-01",
        end="2025-12-31",
        interval="1wk",
    )

    returns = compute_returns(prices)

    portfolio_returns, weights = run_monthly_backtest(
        returns=returns,
        lookback_weeks=26,
        lambda_=0.1,
        cov_method="sample",
        long_only=True,
    )

    summary = performance_summary(portfolio_returns)

    print("Performance Summary")
    print(summary)

    weights.to_csv("weights_output.csv")
    portfolio_returns.to_csv("portfolio_returns_output.csv")


if __name__ == "__main__":
    main()