from logging import info
import numpy as np

import torch
from einops import reduce, rearrange


def _get_backend(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return "torch"
    return "np"


def lwcov(X: np.ndarray | torch.Tensor, detect_orientation: bool = True) -> np.ndarray | torch.Tensor:  # type: ignore
    """
    Computes a well-conditioned (regularized) covariance matrix
    according to the method by Ledoit & Wolf.

    Parameters:
    X (numpy.ndarray or torch.Tensor): A data matrix (observations x variables)

    Returns:
    numpy.ndarray or torch.Tensor: Regularized covariance matrix (variables x variables)
    """
    xp = _get_backend(X)

    # Ensure input is 2D
    if X.ndim != 2:
        raise ValueError("Input data must be a 2D array")

    # Automatically permute if necessary (observations x variables)
    if X.shape[0] < X.shape[1] and detect_orientation:
        X = rearrange(X, "i j -> j i")
        info("Input data matrix transposed to have observations as rows.")

    # 0. Initialization
    nobs, nvar = X.shape

    # Perform mean subtraction
    mu = reduce(X, "n v -> v", "mean")
    X = X - mu

    # 1. Compute the sample covariance matrix
    assert nobs > 1, "At least two observations are needed"
    Xt = rearrange(X, "n v -> v n")
    S = (1 / (nobs - 1)) * Xt @ X

    # 2. Compute the shrinkage parameter and the weights

    if xp == "np":
        m = np.trace(S) / nvar
        d2 = (np.linalg.norm(S - m * np.eye(nvar), "fro") ** 2) / nvar
        eye = np.eye(nvar)
        b2_val = calcbbar2(X, S, xp)
        b2 = np.minimum(b2_val, d2)
    elif xp == "torch" and torch is not None:
        m = torch.trace(S) / nvar
        d2 = (torch.norm(S - m * torch.eye(nvar, device=X.device), p="fro") ** 2) / nvar
        eye = torch.eye(nvar, device=X.device)
        b2_val = calcbbar2(X, S, xp)
        b2 = torch.minimum(b2_val, d2)
    else:
        raise ValueError("Unsupported backend.")

    a2 = d2 - b2

    # 3. Compute the regularized covariance
    Sr = (b2 / d2) * m * eye + (a2 / d2) * S

    return Sr


def calcbbar2(X, S, xp):
    """
    Compute coefficient bbar^2 according to lemma 3.4 in the reference.

    Parameters:
    X: Data matrix
    S: Sample covariance matrix
    xp: Backend module (numpy or torch)

    Returns:
    float or Tensor: Coefficient bbar^2
    """
    nobs, nvar = X.shape
    rownorms = reduce(X**2, "n v -> n", "sum")

    Xt = rearrange(X, "n v -> v n")

    if xp == "np":
        term11 = Xt @ (X * rownorms[:, np.newaxis])
    elif xp == "torch":
        term11 = Xt @ (X * rownorms[:, None])
    else:
        raise ValueError("Unsupported backend.")

    term12 = -2 * S @ (Xt @ X)
    term22 = nobs * S @ S

    if xp == "np":
        bbar2 = np.trace(term11 + term12 + term22) / (nvar * nobs**2)
    elif xp == "torch":
        bbar2 = torch.trace(term11 + term12 + term22) / (nvar * nobs**2)
    else:
        raise ValueError("Unsupported backend.")

    return bbar2


def lwcov_from_cov(
    S: np.ndarray | torch.Tensor,  # type: ignore
    n: int,
) -> np.ndarray | torch.Tensor:  # type: ignore
    """
    Compute Ledoit-Wolf regularized covariance from a pre-computed
    sample covariance matrix, without requiring individual samples.

    Uses a Gaussian approximation for the bbar^2 term:
        E[||z||^4] = tr(S)^2 + 2 * tr(S^2)

    This enables online/streaming covariance estimation where only
    running sufficient statistics (sum_x, sum_xx) are maintained.

    Parameters:
    S (numpy.ndarray or torch.Tensor): Sample covariance matrix (p x p),
        i.e., (1/(n-1)) * sum_i (x_i - mu)(x_i - mu)^T.
    n (int): Number of observations used to compute S.

    Returns:
    numpy.ndarray or torch.Tensor: Regularized covariance matrix (p x p).
    """
    xp = _get_backend(S)
    p = S.shape[0]

    # 1. Grand mean of eigenvalues
    if xp == "np":
        m = np.trace(S) / p
        d2 = (np.linalg.norm(S - m * np.eye(p), "fro") ** 2) / p
        tr_S = np.trace(S)
        tr_S2 = np.trace(S @ S)
        eye = np.eye(p)
    elif xp == "torch" and torch is not None:
        m = torch.trace(S) / p
        d2 = (torch.norm(S - m * torch.eye(p, device=S.device), p="fro") ** 2) / p
        tr_S = torch.trace(S)
        tr_S2 = torch.trace(S @ S)
        eye = torch.eye(p, device=S.device)
    else:
        raise ValueError("Unsupported backend.")

    # 2. Shrinkage intensity using Gaussian approximation for bbar^2
    #    bbar^2 = (n * E[||z||^4] + (2-n) * tr(S^2)) / (p * n^2)
    #    with  E[||z||^4] ≈ tr(S)^2 + 2 * tr(S^2)
    bbar2 = (tr_S ** 2) / (p * n) + ((n + 2) * tr_S2) / (p * n ** 2)
    b2_val = bbar2
    if xp == "np":
        b2 = np.minimum(b2_val, d2)
    elif xp == "torch" and torch is not None:
        b2 = torch.minimum(b2_val, d2)
    else:
        raise ValueError("Unsupported backend.")

    a2 = d2 - b2

    # 3. Regularized covariance
    Sr = (b2 / d2) * m * eye + (a2 / d2) * S

    return Sr
