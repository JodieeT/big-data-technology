[README.md](https://github.com/user-attachments/files/27183700/README.md)
# minvar-l2: Minimum Variance Portfolio with L2 Regularization

A Python package for constructing and backtesting **minimum variance portfolios** with optional **L2 (ridge) regularization**, using real market data from Yahoo Finance.

---

## Table of Contents

- [Project Purpose](#project-purpose)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Module Reference](#module-reference)
- [Running the Backtest Script](#running-the-backtest-script)
- [Running Tests](#running-tests)

---

## Project Purpose

The minimum variance portfolio selects asset weights to minimize portfolio risk (variance), regardless of expected returns. This package extends the classic formulation with an **L2 regularization term** (also known as ridge regularization):

$$\min_{w} \quad w^\top \Sigma w + \lambda \|w\|_2^2 \quad \text{subject to} \quad \mathbf{1}^\top w = 1, \quad w \geq 0$$

The regularization parameter $\lambda$ controls a tradeoff:
- **$\lambda = 0$**: Pure minimum variance — concentrates heavily in the lowest-risk assets.
- **$\lambda \to \infty$**: Forces weights toward the equal-weighted portfolio $w = \frac{1}{n}\mathbf{1}$.

This helps produce more **diversified, stable portfolios** that are less sensitive to estimation error in the covariance matrix — a well-known practical issue in portfolio optimization.

The package also provides a **monthly rebalancing backtest engine** to evaluate portfolio strategies on historical data.

---

## Dataset

Market data is sourced from **[Yahoo Finance](https://finance.yahoo.com/)** via the `yfinance` library.

- **Default assets**: A customizable list of U.S. equity tickers (e.g. `AAPL`, `MSFT`, `AMZN`, `GOOGL`, `META`, `NVDA`, `JPM`, `XOM`)
- **Frequency**: Weekly (`1wk`) adjusted closing prices
- **Default period**: 2018–2025
- **Returns**: Simple percentage returns computed from adjusted close prices

Any list of Yahoo Finance tickers can be substituted. No manual data download is required — data is fetched automatically at runtime.

---

## Installation

**Requirements:** Python ≥ 3.9

**Step 1.** Clone the repository:

```bash
git clone https://github.com/your-username/minvar_l2_project.git
cd minvar_l2_project
```

**Step 2.** Install the package in editable mode:

```bash
pip install -e .
```

**Step 3.** (Optional) Install development dependencies for running tests:

```bash
pip install -e ".[dev]"
```

---

## Project Structure

```
minvar_l2_project/
│
├── README.md
├── pyproject.toml          # Package configuration and dependencies
├── requirements.txt
│
├── data/                   # Folder for any locally cached data files
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   └── minvar_l2/          # Main package
│       ├── __init__.py
│       ├── data.py         # Data download and return computation
│       ├── covariance.py   # Covariance matrix estimators
│       ├── optimizer.py    # Core portfolio optimizer
│       ├── backtest.py     # Monthly rebalancing backtest engine
│       ├── metrics.py      # Portfolio performance metrics
│       └── config.py       # Configuration constants
│
├── scripts/
│   └── run_backtest.py     # End-to-end backtest script
│
└── tests/                  # Unit tests (pytest)
    ├── test_data.py
    ├── test_covariance.py
    ├── test_optimizer.py
    ├── test_metrics.py
    └── test_backtest.py
```

---

## Quick Start

```python
from minvar_l2.data import download_price_data, compute_returns
from minvar_l2.backtest import run_monthly_backtest
from minvar_l2.metrics import performance_summary

# 1. Download weekly price data from Yahoo Finance
prices = download_price_data(
    tickers=["AAPL", "MSFT", "AMZN", "GOOGL", "JPM"],
    start="2018-01-01",
    end="2024-12-31",
    interval="1wk",
)

# 2. Compute simple returns
returns = compute_returns(prices)

# 3. Run a monthly-rebalanced backtest
#    - lookback_weeks: number of past weeks used to estimate covariance
#    - lambda_: L2 regularization strength (0 = pure min-variance)
#    - cov_method: "sample" or "ewma"
portfolio_returns, weights = run_monthly_backtest(
    returns=returns,
    lookback_weeks=26,
    lambda_=0.1,
    cov_method="sample",
    long_only=True,
)

# 4. Evaluate performance
print(performance_summary(portfolio_returns))
```

**Example output:**

```
Total Return             0.6821
Annualized Return        0.0812
Annualized Volatility    0.1234
Sharpe Ratio             0.6581
Max Drawdown            -0.2103
dtype: float64
```

---

## Module Reference

### `minvar_l2.data`

Downloads price data and computes returns.

```python
from minvar_l2.data import download_price_data, compute_returns

# Download adjusted weekly close prices
prices = download_price_data(tickers, start, end, interval="1wk")

# Compute simple period returns from prices
returns = compute_returns(prices)
```

---

### `minvar_l2.covariance`

Estimates the covariance matrix from return data.

```python
from minvar_l2.covariance import sample_covariance, ewma_covariance

# Standard sample covariance
cov = sample_covariance(returns)

# Exponentially weighted covariance (more weight on recent observations)
cov = ewma_covariance(returns, lambda_=0.94)
```

| Method | Description | When to use |
|--------|-------------|-------------|
| `sample_covariance` | Equal-weighted historical covariance | Stable, long-horizon estimation |
| `ewma_covariance` | Exponentially decaying weights (default λ=0.94) | Volatile markets, regime-sensitive |

---

### `minvar_l2.optimizer`

Solves the regularized minimum variance optimization problem.

```python
from minvar_l2.optimizer import min_variance_l2

weights = min_variance_l2(
    cov_matrix,       # pd.DataFrame or np.ndarray covariance matrix
    lambda_=0.1,      # L2 regularization strength
    long_only=True,   # If True, enforce w_i >= 0
)
```

The optimizer uses `scipy.optimize.minimize` with the SLSQP method.

---

### `minvar_l2.backtest`

Runs a walk-forward monthly backtest with periodic rebalancing.

```python
from minvar_l2.backtest import run_monthly_backtest

portfolio_returns, weights_df = run_monthly_backtest(
    returns,                # pd.DataFrame of weekly returns
    lookback_weeks=26,      # Rolling estimation window
    lambda_=0.1,            # L2 regularization
    cov_method="sample",    # "sample" or "ewma"
    long_only=True,
)
```

**How it works:**

1. At the start of each calendar month, estimate the covariance matrix using the most recent `lookback_weeks` weeks of returns.
2. Solve the L2-regularized minimum variance problem to get new weights.
3. Apply those weights to the next month's weekly returns.
4. Repeat for each month in the sample.

---

### `minvar_l2.metrics`

Computes standard portfolio performance statistics.

```python
from minvar_l2.metrics import performance_summary, sharpe_ratio, max_drawdown

# Full summary (returns a pd.Series)
summary = performance_summary(portfolio_returns, periods_per_year=52)

# Individual metrics
sr  = sharpe_ratio(portfolio_returns, risk_free_rate=0.04, periods_per_year=52)
mdd = max_drawdown(portfolio_returns)
```

| Metric | Description |
|--------|-------------|
| `total_return` | Cumulative return over the full period |
| `annualized_return` | Geometric mean annualized return |
| `annualized_volatility` | Annualized standard deviation of returns |
| `sharpe_ratio` | Annualized excess return per unit of risk |
| `max_drawdown` | Largest peak-to-trough decline |

---

## Running the Backtest Script

A ready-to-run script is provided in `scripts/run_backtest.py`. It downloads data, runs the backtest, prints a performance summary, and saves outputs to CSV.

```bash
python scripts/run_backtest.py
```

**Output files generated:**

| File | Contents |
|------|----------|
| `weights_output.csv` | Portfolio weights at each rebalance date |
| `portfolio_returns_output.csv` | Weekly portfolio returns over the backtest period |

To customize the tickers, date range, or optimization parameters, edit the `main()` function in `scripts/run_backtest.py`.

---

## Running Tests

Make sure development dependencies are installed:

```bash
pip install -e ".[dev]"
```

Run all tests with coverage report:

```bash
pytest --cov=minvar_l2 --cov-report=term-missing
```

Run a specific test file:

```bash
pytest tests/test_optimizer.py -v
```

The test suite covers all five modules (`data`, `covariance`, `optimizer`, `metrics`, `backtest`) with over **50 unit tests** targeting correctness, edge cases, and mathematical properties of the optimization.
