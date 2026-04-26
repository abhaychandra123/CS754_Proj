"""
Nonlocal coefficient coupling (CROWN-Inpaint section 6.5).

Independent OMP can pick different atoms for very similar patches.  Following
CSR (Dong-Zhang-Shi 2011) and GSR (Zhang-Zhao-Gao 2014), we regularise each
patch's code toward the weighted mean of its nearest-neighbour codes:

    beta_p = (1 / Z_p) sum_{q in N(p)} w_pq alpha_q,    w_pq = exp(-||x_p - x_q||^2 / h^2)

    alpha_p^* = argmin || W_p^(1/2) . (x_p - D alpha) ||^2 + lambda || alpha - beta_p ||_1

The refinement is restricted to the OMP support of patch p (Section 6.5.2),
turning a K-dimensional convex L1 problem into an |S|-dimensional one with
|S| <= s.  Solved by ISTA initialised at beta_S.

Cadence: applied every two outer iterations (Section 6.5.3); the first
iteration uses pure OMP without coupling.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Inner: L1-centralised refinement on the OMP support
# ---------------------------------------------------------------------------

def _l1_centralised_refine(
    x: np.ndarray,
    D_S: np.ndarray,
    W: np.ndarray,
    beta_S: np.ndarray,
    lam: float,
    n_ista: int = 20,
) -> np.ndarray:
    """
    Solve, for a single patch, with support S already chosen:

        min_{alpha_S}   || W^(1/2) . (x - D_S alpha_S) ||_2^2
                       + lambda || alpha_S - beta_S ||_1

    via ISTA initialised at beta_S.  Each ISTA step is

        z          = alpha - eta * 2 A^T (A alpha - b)
        alpha_new  = beta_S + soft( z - beta_S, eta * lambda )

    where  A = diag(sqrt(W)) D_S,  b = sqrt(W) . x,  and  eta = 1 / L  with
    L = 2 * lambda_max(A^T A) (the Lipschitz constant of the smooth term's
    gradient).  Convergence is linear once we are inside the active orthant,
    and the support is small (|S| <= s = 10), so n_ista = 20 is overkill.

    Parameters
    ----------
    x      : (n,)      patch values
    D_S    : (n, |S|)  dictionary restricted to active support
    W      : (n,)      pixel weights (no square root applied yet)
    beta_S : (|S|,)    centring target
    lam    : float     L1 weight
    n_ista : int       inner iterations

    Returns
    -------
    alpha_S : (|S|,)   refined coefficients
    """
    s_size = D_S.shape[1]
    if s_size == 0:
        return np.zeros(0)

    w = np.sqrt(np.clip(W, 0.0, None))
    A = w[:, None] * D_S                                 # (n, |S|)
    b = w * x                                            # (n,)

    AtA = A.T @ A                                        # (|S|, |S|)
    Atb = A.T @ b                                        # (|S|,)

    # Lipschitz constant for grad of f(alpha) = ||A alpha - b||^2 is 2 ||A^T A||_2.
    # eigvalsh returns sorted ascending; the largest is the operator norm.
    eigvals = np.linalg.eigvalsh(AtA)
    L = 2.0 * float(eigvals[-1])
    if L < 1e-10:
        # Degenerate (all-zero columns); just return the centring target.
        return beta_S.copy()
    eta = 1.0 / L

    alpha = beta_S.copy()
    for _ in range(n_ista):
        grad     = 2.0 * (AtA @ alpha - Atb)             # gradient of smooth part
        z        = alpha - eta * grad
        delta    = z - beta_S
        thresh   = eta * lam
        delta_st = np.sign(delta) * np.maximum(np.abs(delta) - thresh, 0.0)
        alpha    = beta_S + delta_st

    return alpha


# ---------------------------------------------------------------------------
# Outer: nonlocal coupling pass
# ---------------------------------------------------------------------------

def nonlocal_refine(
    X_filled: np.ndarray,
    Alpha: np.ndarray,
    W_patches: np.ndarray,
    D: np.ndarray,
    K_nl: int = 10,
    h: float = None,
    lam: float = 0.1,
    n_ista: int = 20,
) -> np.ndarray:
    """
    Apply one full pass of nonlocal coefficient coupling to all patches.

    Implementation notes
    --------------------
    * Nearest-neighbour search is brute-force Euclidean on full 64-d
      patch vectors (sklearn `NearestNeighbors` with `algorithm='brute'`).
      For ~15k patches this completes in a few seconds.  The spec
      (Section 6.5.2) suggests masked-Euclidean using only confidently
      observed coordinates; using full-Euclidean is a pragmatic v1
      simplification that becomes accurate as iterations progress (hole
      values stabilise across all patches similarly).
    * Self-match (the patch's own index, distance 0) is dropped before
      computing the group centre.
    * The bandwidth `h` defaults to the median NN distance, an adaptive
      choice that scales gracefully across images.

    Parameters
    ----------
    X_filled  : (n, N) float64
        Patch matrix at the current iteration (no NaN; hole positions
        carry their current estimate).
    Alpha : (K, N) float64
        Sparse codes from the OMP step (column p = code for patch p).
    W_patches : (n, N) float64
        Per-pixel weights used during OMP for each patch (continuous in
        [0, 1]; corresponds to W_p in the spec).
    D : (n, K) float64
        Dictionary.
    K_nl : int
        Number of nearest neighbours per patch (excluding self).
    h : float or None
        Similarity bandwidth.  If None, set to the median NN distance.
    lam : float
        L1 centring weight (lambda in the spec).
    n_ista : int
        Inner ISTA iterations per patch.

    Returns
    -------
    Alpha_new : (K, N) float64
        Refined sparse codes.  Patches with empty original support are
        left at zero.
    """
    n, N = X_filled.shape
    K    = Alpha.shape[0]

    if Alpha.shape[1] != N:
        raise ValueError(
            f"Alpha cols {Alpha.shape[1]} != N patches {N}"
        )
    if W_patches.shape != X_filled.shape:
        raise ValueError(
            f"W_patches shape {W_patches.shape} != X_filled shape {X_filled.shape}"
        )
    if D.shape[0] != n or D.shape[1] != K:
        raise ValueError(
            f"D shape {D.shape} inconsistent with patches/atoms ({n}, {K})"
        )

    # ---- 1. Nearest-neighbour search -----------------------------------
    nn = NearestNeighbors(
        n_neighbors=min(K_nl + 1, N),
        algorithm="brute",
        metric="euclidean",
    )
    nn.fit(X_filled.T)                                  # (N, n)
    distances, indices = nn.kneighbors(X_filled.T)
    # Drop the self-match (col 0; sklearn returns sorted ascending)
    distances = distances[:, 1:]                         # (N, K_nl)
    indices   = indices[:, 1:]                           # (N, K_nl)

    if h is None:
        med_d = float(np.median(distances))
        h = max(med_d, 1e-3)

    # ---- 2. Per-row similarity weights (sum to 1) ----------------------
    weights = np.exp(-(distances ** 2) / (h ** 2))       # (N, K_nl)
    Z       = weights.sum(axis=1, keepdims=True)
    safe_Z  = np.where(Z < 1e-12, 1.0, Z)
    weights = weights / safe_Z                           # row-stochastic

    # ---- 3. Group centres beta_p = sum_q w_pq alpha_q ------------------
    # Alpha[:, indices] has shape (K, N, K_nl); weight along the K_nl axis.
    alpha_nbr = Alpha[:, indices]                        # (K, N, K_nl)
    beta      = (alpha_nbr * weights[None, :, :]).sum(axis=2)   # (K, N)

    # ---- 4. Per-patch L1-centred refinement on the OMP support ---------
    Alpha_new = np.zeros_like(Alpha)
    for p in range(N):
        S = np.flatnonzero(Alpha[:, p])                  # support of patch p
        if S.size == 0:
            continue
        alpha_S = _l1_centralised_refine(
            x=X_filled[:, p],
            D_S=D[:, S],
            W=W_patches[:, p],
            beta_S=beta[S, p],
            lam=lam,
            n_ista=n_ista,
        )
        Alpha_new[S, p] = alpha_S

    return Alpha_new
