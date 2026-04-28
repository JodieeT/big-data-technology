import numpy as np
from scipy.optimize import minimize


def min_variance_l2(cov_matrix, lambda_=0.0, long_only=True):
    """
    Solve minimum variance portfolio with L2 regularization.

    Objective
    ---------
    minimize w.T @ Sigma @ w + lambda * ||w||_2^2

    subject to
    ----------
    sum(w) = 1
    w_i >= 0 if long_only is True

    Parameters
    ----------
    cov_matrix : pd.DataFrame or np.ndarray
        Covariance matrix.
    lambda_ : float
        L2 regularization strength.
    long_only : bool
        Whether to impose nonnegative weights.

    Returns
    -------
    np.ndarray
        Optimal portfolio weights.
    """
    Sigma = np.asarray(cov_matrix)
    n_assets = Sigma.shape[0]

    def objective(w):
        return w @ Sigma @ w + lambda_ * np.sum(w ** 2)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    if long_only:
        bounds = [(0, 1) for _ in range(n_assets)]
    else:
        bounds = None

    x0 = np.ones(n_assets) / n_assets

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    return result.x