import numpy as np
import pandas as pd
import pytest

from src.covariance import sample_covariance, ewma_covariance


# ── helpers ────────────────────────────────────────────────────────────────────

def make_returns(n_periods=20, n_assets=3, seed=42):
    np.random.seed(seed)
    data = np.random.randn(n_periods, n_assets) * 0.01
    cols = [chr(65 + i) for i in range(n_assets)]
    return pd.DataFrame(data, columns=cols)


# ── sample_covariance ──────────────────────────────────────────────────────────

def test_sample_covariance_shape():
    returns = make_returns(n_assets=3)
    cov = sample_covariance(returns)
    assert cov.shape == (3, 3)


def test_sample_covariance_is_symmetric():
    returns = make_returns(n_assets=4)
    cov = sample_covariance(returns)
    assert np.allclose(cov.values, cov.values.T)


def test_sample_covariance_is_positive_semidefinite():
    returns = make_returns(n_assets=3)
    cov = sample_covariance(returns)
    eigenvalues = np.linalg.eigvalsh(cov.values)
    assert np.all(eigenvalues >= -1e-10)


def test_sample_covariance_diagonal_positive():
    returns = make_returns(n_assets=3)
    cov = sample_covariance(returns)
    assert np.all(np.diag(cov.values) > 0)


def test_sample_covariance_two_assets():
    returns = pd.DataFrame({
        "A": [0.01, 0.02, -0.01],
        "B": [0.03, -0.01, 0.02],
    })
    cov = sample_covariance(returns)
    assert cov.shape == (2, 2)
    assert np.isclose(cov.loc["A", "A"], returns["A"].var(ddof=1))


def test_sample_covariance_preserves_column_names():
    returns = make_returns(n_assets=3)
    cov = sample_covariance(returns)
    assert list(cov.columns) == list(returns.columns)
    assert list(cov.index) == list(returns.columns)


# ── ewma_covariance ────────────────────────────────────────────────────────────

def test_ewma_covariance_shape():
    returns = make_returns(n_assets=3)
    cov = ewma_covariance(returns)
    assert cov.shape == (3, 3)


def test_ewma_covariance_is_symmetric():
    returns = make_returns(n_assets=4)
    cov = ewma_covariance(returns)
    assert np.allclose(cov.values, cov.values.T, atol=1e-10)


def test_ewma_covariance_is_positive_semidefinite():
    returns = make_returns(n_assets=3)
    cov = ewma_covariance(returns)
    eigenvalues = np.linalg.eigvalsh(cov.values)
    assert np.all(eigenvalues >= -1e-10)


def test_ewma_covariance_diagonal_positive():
    returns = make_returns(n_assets=3)
    cov = ewma_covariance(returns)
    assert np.all(np.diag(cov.values) > 0)


def test_ewma_covariance_two_assets():
    returns = pd.DataFrame({
        "A": [0.01, 0.02, -0.01, 0.04],
        "B": [0.03, -0.01, 0.02, 0.01],
    })
    cov = ewma_covariance(returns)
    assert cov.shape == (2, 2)


def test_ewma_vs_sample_different():
    # EWMA should differ from sample covariance (weights recent observations more)
    returns = make_returns(n_periods=30, n_assets=3)
    cov_sample = sample_covariance(returns)
    cov_ewma = ewma_covariance(returns)
    assert not np.allclose(cov_sample.values, cov_ewma.values)


def test_ewma_covariance_custom_lambda():
    returns = make_returns(n_assets=3)
    cov_default = ewma_covariance(returns, lambda_=0.94)
    cov_custom = ewma_covariance(returns, lambda_=0.50)
    # Different lambda should give different results
    assert not np.allclose(cov_default.values, cov_custom.values)
