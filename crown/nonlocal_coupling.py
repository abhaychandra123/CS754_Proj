"""
Nonlocal coefficient coupling (CROWN-Inpaint section 6.5) -- corrected.

Independent OMP can pick different atoms for very similar patches.  Following
CSR (Dong-Zhang-Shi 2011) and GSR (Zhang-Zhao-Gao 2014), we regularise each
patch's code toward the weighted mean of its nearest-neighbour codes:

    beta_p   = (1 / Z_p) sum_{q in N(p)} w_pq alpha_q
    alpha_p* = argmin || W_p^(1/2) . (x_p - D alpha) ||^2 + lambda || alpha - beta_p ||_1

The refinement is restricted to the OMP support of patch p (Section 6.5.2),
turning a K-dimensional convex L1 problem into an |S|-dimensional one with
|S| <= s.  Solved by ISTA initialised at beta_S.

Distance metrics (Section 6.5.2) -- this implementation uses TWO weighted
distances faithfully derived from the spec:

  (1) NN search uses  d_nn^2(p, q) = sum_j  W_p[j] * (x_p[j] - x_q[j])^2,
      i.e. Euclidean distance restricted to the OBSERVED coordinates of the
      query patch p.  This is what spec 6.5.2 calls "Euclidean distance
      restricted to observed coordinates of p".

  (2) Similarity weights use  d_sim^2(p, q) = sum_j W_p[j] W_q[j] (x_p[j] - x_q[j])^2,
      i.e. coordinates that are confidently observed in BOTH patches contribute
      to the similarity score, as required by Section 6.5.2 ("uses W_p . W_q").

Cadence: applied every two outer iterations (Section 6.5.3); the first
iteration uses pure OMP without coupling.

The routine returns Alpha unchanged when K_nl <= 0 (treated as the explicit
"no nonlocal" ablation knob mentioned in spec Section 10.3).
"""

from __future__ import annotations

import numpy as np


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
    Solve, for a single patch with support S already chosen:

        min_{alpha_S}   || W^(1/2) . (x - D_S alpha_S) ||_2^2
                       + lambda || alpha_S - beta_S ||_1

    via ISTA initialised at beta_S.
    """
    s_size = D_S.shape[1]
    if s_size == 0:
        return np.zeros(0)

    w = np.sqrt(np.clip(W, 0.0, None))
    A = w[:, None] * D_S
    b = w * x

    AtA = A.T @ A
    Atb = A.T @ b

    eigvals = np.linalg.eigvalsh(AtA)
    L = 2.0 * float(eigvals[-1])
    if L < 1e-10:
        return beta_S.copy()
    eta = 1.0 / L

    alpha = beta_S.copy()
    for _ in range(n_ista):
        grad     = 2.0 * (AtA @ alpha - Atb)
        z        = alpha - eta * grad
        delta    = z - beta_S
        thresh   = eta * lam
        delta_st = np.sign(delta) * np.maximum(np.abs(delta) - thresh, 0.0)
        alpha    = beta_S + delta_st

    return alpha


# ---------------------------------------------------------------------------
# Per-query weighted Euclidean nearest-neighbour search
# ---------------------------------------------------------------------------

def _weighted_nn_search(
    X: np.ndarray,
    W: np.ndarray,
    K_nl: int,
) -> tuple:
    """
    For each query patch p, find K_nl nearest patches under the metric

        d_nn^2(p, q) = sum_j  W[j, p] * (X[j, p] - X[j, q])^2

    i.e. weighted Euclidean using the QUERY's per-pixel weights, with
    self excluded.  Implemented as a per-query Python loop over patches;
    each iteration is fully vectorised over the candidate set.

    Parameters
    ----------
    X : (n, N) float64    Patch matrix.
    W : (n, N) float64    Per-pixel weights for every patch.
    K_nl : int            Number of neighbours per query (excluding self).

    Returns
    -------
    indices  : (N, K_nl) int64    Neighbour indices.
    sq_dists : (N, K_nl) float64  Distances squared in the per-query metric.
    """
    n, N = X.shape
    if K_nl < 1:
        raise ValueError(f"K_nl must be >= 1, got {K_nl}")
    if K_nl > N - 1:
        K_nl = N - 1

    indices  = np.zeros((N, K_nl), dtype=np.int64)
    sq_dists = np.zeros((N, K_nl), dtype=np.float64)

    for p in range(N):
        diff_sq = (X - X[:, p:p + 1]) ** 2                # (n, N)
        dist    = (W[:, p:p + 1] * diff_sq).sum(axis=0)    # (N,)
        dist[p] = np.inf                                   # exclude self

        # argpartition gives an unsorted top-K; sort just those for
        # deterministic ordering.
        if K_nl < N:
            top_idx = np.argpartition(dist, K_nl)[:K_nl]
            order   = np.argsort(dist[top_idx])
            top_idx = top_idx[order]
        else:
            top_idx = np.argsort(dist)[:K_nl]

        indices[p]  = top_idx
        sq_dists[p] = dist[top_idx]

    return indices, sq_dists


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

    Algorithm (Section 6.5)
    -----------------------
    1. For each patch p, find K_nl nearest neighbours under the
       per-query weighted distance d_nn^2(p, q) = sum_j W_p[j] (x_p - x_q)_j^2.
    2. For each (p, q) pair, recompute the SIMILARITY weight using the
       coordinates confidently observed in BOTH patches (W_p . W_q):

           w_pq    = exp(- d_sim^2(p, q) / h^2)
           d_sim^2 = sum_j W_p[j] W_q[j] (x_p - x_q)_j^2

       This matches the spec's "uses W_p . W_q" definition.
    3. Group centre  beta_p = (1 / Z_p) sum_q w_pq alpha_q.
    4. Per-patch L1-centred refinement on the OMP support via ISTA.

    Bandwidth `h`
    -------------
    If None, h is set so that h^2 = max(median(d_sim^2), 1e-6).  This is an
    iteration-adaptive default that scales with the typical similarity
    distance among kept neighbours.

    No-op behaviour
    ---------------
    If K_nl <= 0 or N <= 1 the call is an exact identity: Alpha is returned
    unchanged.  This is the "no nonlocal" ablation knob from spec
    Section 10.3.
    """
    n, N = X_filled.shape
    K    = Alpha.shape[0]

    # ---- 0. No-op early return ----------------------------------------
    if K_nl <= 0 or N <= 1:
        return Alpha.copy()

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

    # ---- 1. Weighted-Euclidean NN search per query --------------------
    indices, _ = _weighted_nn_search(X_filled, W_patches, K_nl)
    K_nl_eff   = indices.shape[1]   # may be smaller than K_nl when N is tiny

    # ---- 2. Similarity weights using W_p . W_q ------------------------
    # nbr_X[:, p, k] = X[:, indices[p, k]]
    nbr_X = X_filled[:, indices]                      # (n, N, K_nl_eff)
    nbr_W = W_patches[:, indices]                     # (n, N, K_nl_eff)

    # diff[:, p, k]   = X_filled[:, p] - nbr_X[:, p, k]
    diff = X_filled[:, :, None] - nbr_X                # (n, N, K_nl_eff)

    # W_pq[:, p, k] = W_patches[:, p] * W_patches[:, indices[p, k]]
    W_pq = W_patches[:, :, None] * nbr_W               # (n, N, K_nl_eff)

    # d_sim^2(p, k) = sum_j W_pq[j, p, k] * diff[j, p, k]^2
    d_sim_sq = (W_pq * diff * diff).sum(axis=0)        # (N, K_nl_eff)

    # Adaptive bandwidth: h^2 = median(d_sim^2)
    if h is None:
        med = float(np.median(d_sim_sq))
        h_sq = max(med, 1e-6)
    else:
        h_sq = float(h) ** 2

    weights_pq = np.exp(-d_sim_sq / h_sq)               # (N, K_nl_eff)
    Z          = weights_pq.sum(axis=1, keepdims=True)
    safe_Z     = np.where(Z < 1e-12, 1.0, Z)
    weights_pq = weights_pq / safe_Z                    # row-stochastic

    # ---- 3. Group centres beta_p = sum_q w_pq alpha_q -----------------
    alpha_nbr = Alpha[:, indices]                      # (K, N, K_nl_eff)
    beta      = (alpha_nbr * weights_pq[None, :, :]).sum(axis=2)   # (K, N)

    # ---- 4. Per-patch L1-centred refinement on the OMP support --------
    Alpha_new = np.zeros_like(Alpha)
    for p in range(N):
        S = np.flatnonzero(Alpha[:, p])
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
