import pandas as pd


def sample_covariance(returns):
    """
    Compute sample covariance matrix.
    """
    return returns.cov()


def ewma_covariance(returns, lambda_=0.94):
    """
    Compute exponentially weighted moving average covariance matrix.
    """
    return returns.ewm(alpha=1 - lambda_).cov().dropna().iloc[-returns.shape[1]:]