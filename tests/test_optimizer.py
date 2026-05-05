import numpy as np
import pytest

from minvar_l2.optimizer import min_variance_l2


# ── helpers ────────────────────────────────────────────────────────────────────

def make_cov(n=3, seed=42):
    np.random.seed(seed)
    A = np.random.randn(n, n)
    return A @ A.T + np.eye(n) * 0.01  # ensure positive definite


# ── basic constraints ──────────────────────────────────────────────────────────

def test_weights_sum_to_one():
    cov = make_cov(3)
    weights = min_variance_l2(cov, lambda_=0.1)
    assert np.isclose(weights.sum(), 1.0)


def test_long_only_weights_nonnegative():
    cov = make_cov(4)
    weights = min_variance_l2(cov, lambda_=0.1, long_only=True)
    assert np.all(weights >= -1e-8)


def test_long_short_allows_negative_weights():
    # With long_only=False and a covariance that encourages shorting,
    # at least check the constraint sum=1 still holds
    cov = make_cov(4)
    weights = min_variance_l2(cov, lambda_=0.0, long_only=False)
    assert np.isclose(weights.sum(), 1.0)


def test_output_shape_2_assets():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    weights = min_variance_l2(cov, lambda_=0.0)
    assert weights.shape == (2,)


def test_output_shape_n_assets():
    for n in [3, 5, 10]:
        cov = make_cov(n)
        weights = min_variance_l2(cov, lambda_=0.1)
        assert weights.shape == (n,)


# ── effect of L2 regularization ────────────────────────────────────────────────

def test_l2_identity_covariance_equal_weights():
    # With identity covariance, all assets are equivalent → equal weights
    cov = np.eye(4)
    weights = min_variance_l2(cov, lambda_=10.0)
    expected = np.ones(4) / 4
    assert np.allclose(weights, expected, atol=1e-4)


def test_l2_zero_lambda_concentrates_on_low_var():
    # Without regularization, optimizer should heavily favor lowest-variance asset
    cov = np.diag([0.01, 0.09, 0.25])
    weights_no_reg = min_variance_l2(cov, lambda_=0.0, long_only=True)
    weights_reg = min_variance_l2(cov, lambda_=1.0, long_only=True)
    # No regularization → more concentrated; regularization → more spread out
    assert weights_no_reg.max() > weights_reg.max()


def test_high_l2_approaches_equal_weights():
    # Very high lambda dominates → weights approach 1/n
    cov = make_cov(3)
    weights = min_variance_l2(cov, lambda_=1e6, long_only=True)
    expected = np.ones(3) / 3
    assert np.allclose(weights, expected, atol=1e-3)


# ── accepts pandas DataFrame as input ─────────────────────────────────────────

def test_accepts_dataframe_input():
    import pandas as pd
    cov_df = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    weights = min_variance_l2(cov_df, lambda_=0.1)
    assert np.isclose(weights.sum(), 1.0)


# ── edge cases ─────────────────────────────────────────────────────────────────

def test_single_asset():
    cov = np.array([[0.04]])
    weights = min_variance_l2(cov, lambda_=0.0)
    assert np.isclose(weights[0], 1.0)


def test_lambda_zero_long_only():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    weights = min_variance_l2(cov, lambda_=0.0, long_only=True)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= -1e-8)
