"""
Confidence-weighted masked OMP (CROWN-Inpaint section 6.4).

Generalises the binary-mask `masked_omp` from `masked_ksvd.py` to continuous
weights W in [0, 1]^n.

Per-patch optimisation problem:

    min_alpha   || W^(1/2) . (x - D alpha) ||_2^2     s.t.   ||alpha||_0 <= s

By the substitution

    w        = sqrt(W)            in R^n
    D_tilde  = diag(w) D          in R^{n x K}
    x_tilde  = w . x              in R^n

the objective becomes the standard OMP form  min ||x_tilde - D_tilde alpha||_2^2
(Section 5.2 of the spec).  The columns of D_tilde are no longer unit-norm,
so atom-selection inner products are normalised by the per-column L2 norm
of D_tilde -- exactly the same trick used in the existing binary
`masked_omp`, only now the mask is continuous.

When  W  is binary in {0, 1}^n  this reduces  bit-exactly  to the
existing `masked_omp`, so the change is a strict generalisation.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Building per-patch weight maps from M and c
# ---------------------------------------------------------------------------

def build_weight_image(M: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Combine the binary observation mask M and the hole confidence map c into
    a single continuous-weight image  W_img in [0, 1]^{H x W}.

        W_img(i) =  1            if i in Omega           (M_i = 1)
        W_img(i) =  c(i)         if i in Omega^c         (M_i = 0)

    Equivalently  W_img = M + (1 - M) * c.

    Patches W_p extracted from W_img have observed positions weighted at
    full trust (1.0) and missing positions weighted by their hole-pixel
    confidence (Section 6.1.2).

    Parameters
    ----------
    M : (H, W) float64        Binary mask (1 = observed, 0 = missing)
    c : (H, W) float64        Hole-pixel confidence in [0, 1]

    Returns
    -------
    W_img : (H, W) float64    Continuous weights in [0, 1]
    """
    if M.shape != c.shape:
        raise ValueError(f"shape mismatch: M={M.shape}, c={c.shape}")
    return np.clip(M + (1.0 - M) * c, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Weighted OMP (per-patch)
# ---------------------------------------------------------------------------

def weighted_omp(
    x: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    s: int,
    eps_norm: float = 1e-10,
) -> np.ndarray:
    """
    Confidence-weighted OMP for a single patch.

    Solves:
        min_{alpha in R^K}   || sqrt(W) . (x - D alpha) ||_2^2
                              s.t.   ||alpha||_0 <= s

    This is the equivalent of weighted least squares on the active support;
    OMP is a greedy approximation that adds one atom per iteration.

    Parameters
    ----------
    x : (n,) float64
        Patch vector (no NaN; missing entries should already carry their
        current estimate).
    D : (n, K) float64
        Dictionary with unit-norm columns (the un-weighted dictionary).
    W : (n,) float64
        Per-pixel non-negative weights.  Typical range is [0, 1] but any
        non-negative values are accepted.
    s : int
        Sparsity (max number of nonzeros in the returned code).
    eps_norm : float
        Threshold below which a column of D_tilde is treated as zero
        (e.g. when its corresponding pixel weights are all zero).

    Returns
    -------
    alpha : (K,) float64
        Sparse code with at most s nonzeros.
    """
    n, K = D.shape
    if x.shape != (n,):
        raise ValueError(f"x shape {x.shape} != ({n},)")
    if W.shape != (n,):
        raise ValueError(f"W shape {W.shape} != ({n},)")
    if s < 0:
        raise ValueError(f"sparsity s must be >= 0, got {s}")
    if s == 0:
        return np.zeros(K)

    # Substitution from spec section 5.2
    w        = np.sqrt(np.clip(W, 0.0, None))           # (n,)
    x_tilde  = w * x                                     # (n,)
    D_tilde  = w[:, None] * D                            # (n, K)

    # L2-norms of weighted dictionary columns; used to normalise atom-
    # selection inner products so that atoms whose pixels are weighted
    # heavily are not artificially favoured.
    norms      = np.linalg.norm(D_tilde, axis=0)         # (K,)
    safe_norms = np.where(norms < eps_norm, 1.0, norms)
    dead_atom  = norms < eps_norm                        # treat as -inf below

    # OMP state
    residual = x_tilde.copy()                            # (n,)
    support: list[int] = []
    alpha    = np.zeros(K)
    alpha_s  = np.zeros(0)

    for _ in range(s):
        # Normalised correlations: (D_tilde / ||D_tilde||).T @ residual
        corr = np.abs(D_tilde.T @ residual) / safe_norms

        # Disqualify already-chosen atoms and zero-norm atoms.
        if support:
            corr[support] = -np.inf
        corr[dead_atom] = -np.inf

        j_star = int(np.argmax(corr))
        if not np.isfinite(corr[j_star]) or corr[j_star] <= 0:
            # No usable atom remains (all eligible correlations are <=0
            # or all atoms have been used).  Stop early; OMP is allowed
            # to terminate with fewer than s nonzeros if no atom can
            # further reduce the residual.
            break
        support.append(j_star)

        # Solve weighted least-squares on the current support.
        # min ||x_tilde - D_tilde[:, S] alpha_s||^2  is identical to
        # min ||W^(1/2) (x - D[:, S] alpha_s)||^2.
        D_s = D_tilde[:, support]
        alpha_s, _, _, _ = np.linalg.lstsq(D_s, x_tilde, rcond=None)

        residual = x_tilde - D_s @ alpha_s

    for local_idx, atom_idx in enumerate(support):
        alpha[atom_idx] = alpha_s[local_idx]

    return alpha


def weighted_omp_batch(
    X: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    s: int,
    eps_norm: float = 1e-10,
) -> np.ndarray:
    """
    Vectorised loop over patches.

    Parameters
    ----------
    X : (n, N) float64    Patch matrix, columns are patches.
    D : (n, K) float64    Dictionary (unit-norm columns).
    W : (n, N) float64    Per-pixel weights for every patch.
    s : int               Sparsity.

    Returns
    -------
    Alpha : (K, N) float64    Sparse codes for every patch.

    NOTE: A truly batched OMP requires sharing supports across patches,
    which CROWN-Inpaint does not assume.  This is a Python-level loop
    over patches that calls `weighted_omp` for each.  It is the bottleneck
    of the outer iteration; for production use one would write a
    Numba/C++ kernel.  For research-grade evaluation on 128 x 128 images
    this loop runs in a few seconds per outer iteration.
    """
    if X.shape != W.shape:
        raise ValueError(f"X / W shape mismatch: {X.shape} vs {W.shape}")
    if X.shape[0] != D.shape[0]:
        raise ValueError(
            f"patch dim {X.shape[0]} does not match dictionary {D.shape}"
        )
    K = D.shape[1]
    N = X.shape[1]
    Alpha = np.zeros((K, N), dtype=np.float64)
    for i in range(N):
        Alpha[:, i] = weighted_omp(X[:, i], D, W[:, i], s, eps_norm=eps_norm)
    return Alpha
