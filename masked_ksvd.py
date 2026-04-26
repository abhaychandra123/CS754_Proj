"""
=============================================================================
Masked Dictionary Learning for Image Inpainting — Phase 1 & 2 (1D Proof)
=============================================================================
Novel contribution:
  Classical K-SVD assumes fully observed patches. When pixels are missing,
  the SVD step on the error matrix E_k is undefined (missing entries ≠ zeros).

  This file proves the core algorithmic novelty on 1D synthetic data:

  (1) Masked OMP  — Greedy sparse coding that ignores missing entries:
        min  ||M(x - Dα)||₂²   s.t.  ||α||₀ ≤ s
      Achieved by reducing to standard OMP on the masked pair (Mx, MD).

  (2) Masked K-SVD — Dictionary update via ALS Rank-1 Approximation:
        min_{d_k, h_k}  Σ_{i∈S_k} ||m_i ⊙ (e_{k,i} - d_k h_k[i])||₂²
      ALS alternates two closed-form subproblems, side-stepping SVD entirely.
      Post-ALS normalisation: d_k ← d_k/||d_k||, h_k ← h_k·||d_k||
      (mandatory so OMP inner products remain correctly scaled).

  Baseline for comparison:
  Standard K-SVD where missing entries are IMPUTED with zeros — no masking.
  This is the naive approach the novel method improves upon.

Author : CS754 Project Group
License: MIT
=============================================================================
"""

import numpy as np
import os
import csv
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time


# =============================================================================
# PHASE 1 — Synthetic Data Generation
# =============================================================================

def generate_ground_truth_dictionary(n, k, rng):
    """
    Random overcomplete dictionary with unit-norm columns.

    Parameters
    ----------
    n : signal dimension   (e.g. 20)
    k : number of atoms    (e.g. 50, overcomplete since k > n)

    Returns
    -------
    D : (n, k)  each column satisfies ||d_j||_2 = 1
    """
    D = rng.standard_normal((n, k))
    # Divide every column by its L2-norm.
    # keepdims=True -> shape (1,k) -> broadcasts over n rows.
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    return D


def generate_sparse_signals(D, N, s, noise_std, rng):
    """
    Generate N noisy sparse signals  x = D alpha + eps,  ||alpha||_0 = s.

    Additive noise eps ~ N(0, noise_std^2) makes the problem realistic:
    without noise, a random dictionary can trivially over-fit observed entries
    in one iteration, masking any convergence signal.

    Parameters
    ----------
    D         : (n, k)  ground-truth dictionary
    N         : int     number of signals
    s         : int     sparsity per code vector
    noise_std : float   standard deviation of additive Gaussian noise

    Returns
    -------
    X          : (n, N)  noisy signals
    Alpha_true : (k, N)  true sparse codes (exactly s nonzeros per column)
    """
    n, k = D.shape
    Alpha_true = np.zeros((k, N))

    for i in range(N):
        support = rng.choice(k, size=s, replace=False)   # s distinct atom indices
        Alpha_true[support, i] = rng.standard_normal(s)  # N(0,1) nonzero coefficients

    X = D @ Alpha_true + noise_std * rng.standard_normal((n, N))
    return X, Alpha_true


def corrupt_signals(X, missing_fraction, rng):
    """
    Randomly drop each pixel independently with probability = missing_fraction.

    The algorithm NEVER imputes missing values. Corruption is represented
    by a binary mask only.  NaN is placed for human-readable display; all
    arithmetic operates exclusively through the mask matrix.

    Returns
    -------
    X_corrupt : (n, N)  copy with NaN at missing positions
    Masks     : (n, N)  binary;  Masks[j,i] = 1 iff entry (j,i) is observed
    """
    n, N = X.shape
    Masks = (rng.uniform(size=(n, N)) >= missing_fraction).astype(np.float64)

    # Guarantee at least 1 observed entry per signal (avoids rank-0 patch)
    for i in range(N):
        if Masks[:, i].sum() == 0:
            Masks[rng.integers(0, n), i] = 1.0

    X_corrupt = X.copy()
    X_corrupt[Masks == 0] = np.nan
    return X_corrupt, Masks


def init_dictionary_from_data(X_corrupt, Masks, K, rng):
    """
    Initialise dictionary by sampling observed patches from training data.

    For each atom slot: pick a random signal, fill unobserved entries with
    that signal's observed mean, then normalise to unit norm.
    This is a much better starting point than pure random Gaussian atoms,
    and ensures the convergence curve starts from a meaningful baseline.
    """
    n, N = X_corrupt.shape
    D = np.zeros((n, K))
    indices = rng.choice(N, size=K, replace=(K > N))

    for j, idx in enumerate(indices):
        patch    = X_corrupt[:, idx].copy()
        mask_col = Masks[:, idx]
        obs_mean = np.nanmean(patch) if np.any(~np.isnan(patch)) else 0.0
        # Fill unobserved entries with the observed mean of that signal
        patch    = np.where(mask_col == 1, patch, obs_mean)
        norm_p   = np.linalg.norm(patch)
        if norm_p > 1e-10:
            D[:, j] = patch / norm_p
        else:
            v = rng.standard_normal(n)
            D[:, j] = v / np.linalg.norm(v)

    # Safety normalisation pass
    col_norms = np.linalg.norm(D, axis=0, keepdims=True)
    col_norms = np.where(col_norms < 1e-10, 1.0, col_norms)
    D /= col_norms
    return D


# =============================================================================
# PHASE 2A — Masked Orthogonal Matching Pursuit
# =============================================================================

def masked_omp(x, D, mask, s):
    """
    Masked OMP: greedy sparse coding over partially observed signals.

    Objective
    ---------
        min  ||M(x - D alpha)||_2^2    subject to    ||alpha||_0 <= s

    Key Reduction
    -------------
    Define:
        x_tilde = mask * x              (observed entries; 0 elsewhere)
        D_tilde = mask[:,None] * D      (zero out rows for missing entries)

    The objective becomes standard OMP on the observed-only system (x_tilde, D_tilde).
    Missing entries contribute exactly zero to every inner product — they are
    effectively ignored with NO imputation or interpolation performed.

    Normalisation note
    ------------------
    D has unit-norm columns but D_tilde = MD does NOT (rows zeroed).
    We divide each inner product by ||D_tilde[:,j]||_2 so that atoms with
    fewer observed entries are not artificially disfavoured.

    Parameters
    ----------
    x    : (n,)   signal; NaN at missing positions
    D    : (n, k) dictionary with unit-norm columns
    mask : (n,)   binary mask: 1=observed, 0=missing
    s    : int    target sparsity

    Returns
    -------
    alpha : (k,)  sparse coefficient vector, at most s nonzeros
    """
    n, k = D.shape

    # --- Reduce to observed subspace ------------------------------------------
    # nan_to_num converts NaN -> 0; mask then zeros any residual missing entries.
    # x_tilde shape: (n,)
    x_tilde = mask * np.nan_to_num(x, nan=0.0)

    # D_tilde[j, col] = mask[j] * D[j, col]
    # Broadcasting: mask[:,None] has shape (n,1) -> applied over all k columns.
    # D_tilde shape: (n, k)
    D_tilde = mask[:, None] * D

    # Precompute L2-norms of masked atoms (may be < 1 since rows are zeroed)
    # D_tilde_norms shape: (k,)
    D_tilde_norms = np.linalg.norm(D_tilde, axis=0)
    # Suppress atoms with near-zero observable support
    safe_norms = np.where(D_tilde_norms < 1e-10, 1.0, D_tilde_norms)

    # --- OMP state ------------------------------------------------------------
    residual = x_tilde.copy()   # shape: (n,)
    support  = []               # list of selected atom indices
    alpha    = np.zeros(k)
    alpha_s  = np.zeros(0)      # current LS solution over support

    for _ in range(s):
        # Step 1: normalised inner products of residual with all masked atoms
        # (D_tilde.T @ residual) computes all k inner products at once
        # shape: (k,)
        correlations = (D_tilde.T @ residual) / safe_norms

        # Step 2: greedy selection — atom with highest absolute correlation
        j_star = int(np.argmax(np.abs(correlations)))
        if j_star in support:
            break   # duplicate (linear dependence); terminate early
        support.append(j_star)

        # Step 3: orthogonal projection (least squares) over current support
        # D_s shape: (n, t)  where t = |support|
        D_s = D_tilde[:, support]
        # Solve min ||x_tilde - D_s alpha_s||_2^2 (overdetermined, n > t)
        alpha_s, _, _, _ = np.linalg.lstsq(D_s, x_tilde, rcond=None)
        # alpha_s shape: (t,)

        # Step 4: update residual
        residual = x_tilde - D_s @ alpha_s   # shape: (n,)

    # Write solution back to full-length sparse vector
    for local_idx, atom_idx in enumerate(support):
        alpha[atom_idx] = alpha_s[local_idx] if len(support) > 0 else 0.0

    return alpha   # shape: (k,)


# =============================================================================
# PHASE 2B — Masked K-SVD: ALS Rank-1 Dictionary Update
# =============================================================================

def masked_ksvd_update_atom(D, Alpha, X, Masks, k_idx, als_iters=15, rng=None):
    """
    Update one dictionary atom via Masked Rank-1 ALS Approximation.

    Why ALS instead of SVD?
    -----------------------
    Standard K-SVD uses SVD(E_k) where E_k is the leave-one-out error matrix.
    With missing pixels, E_k has unobserved entries and numpy.linalg.svd cannot
    handle NaN.  ALS solves the SAME rank-1 factorisation but restricts every
    subproblem to observed entries only:

        min_{d_k, h_k}  sum_{i in S_k}  || m_i * (e_{k,i} - d_k * h_k[i]) ||_2^2

    where:
        e_{k,i} = leave-one-out error vector for signal i
        m_i     = Masks[:,i]  binary observation mask
        S_k     = { i : Alpha[k_idx, i] != 0 }  active set for atom k

    ALS Step A — Fix d_k, update h_k[i] independently for each i in S_k:
    (closed-form scalar weighted regression)

        h_k[i] = sum_j m_i[j] * d_k[j] * e_{k,i}[j]
                 ─────────────────────────────────────
                 sum_j m_i[j] * d_k[j]^2

    Vectorised over i: form d_k_masked of shape (n, |S_k|):
        d_k_masked[j,i] = Masks[j, S_k[i]] * d_k[j]
    Broadcasting: d_k[:,None] has shape (n,1) -- expands over |S_k| columns.
        numerator[i]   = sum_j d_k_masked[j,i] * E_k[j,i]   (axis=0 sum)
        denominator[i] = sum_j d_k_masked[j,i]^2             (axis=0 sum)

    ALS Step B — Fix h_k, update d_k[j] independently for each entry j:
    (closed-form scalar weighted regression per dimension)

        d_k[j] = sum_i m_i[j] * h_k[i] * e_{k,i}[j]
                 ─────────────────────────────────────
                 sum_i m_i[j] * h_k[i]^2

    Vectorised over j: form h_k_masked of shape (n, |S_k|):
        h_k_masked[j,i] = Masks[j, S_k[i]] * h_k[i]
    Broadcasting: h_k[None,:] has shape (1,|S_k|) -- expands over n rows.
        numerator_d[j]   = sum_i h_k_masked[j,i] * E_k[j,i]     (axis=1 sum)
        denominator_d[j] = sum_i Masks[j,S_k[i]] * h_k[i]^2     (axis=1 sum)

    Normalisation (MANDATORY for OMP correctness):
        scale = ||d_k||_2
        d_k  <- d_k / scale       (unit-norm atom)
        h_k  <- h_k * scale       (absorb scale: product d_k h_k^T unchanged)

    Parameters
    ----------
    D         : (n, K)   current dictionary (unit-norm columns)
    Alpha     : (K, N)   current sparse codes
    X         : (n, N)   signals with NaN at missing positions
    Masks     : (n, N)   binary masks (1=observed, 0=missing)
    k_idx     : int      atom index to update
    als_iters : int      number of ALS alternation steps

    Returns
    -------
    d_k_new : (n,)  updated unit-norm atom
    h_k_new : (N,)  updated coefficient row (nonzero only at S_k positions)
    """
    n, K = D.shape
    N    = Alpha.shape[1]

    # Reproducible RNG: if no rng is provided, fall back to a fresh default
    # generator (NOT the legacy global np.random state, which is process-wide
    # and would leak randomness across calls).  Callers seeking strict
    # determinism should pass a seeded `np.random.Generator` from the outer
    # K-SVD loop.
    if rng is None:
        rng = np.random.default_rng()

    # ---- Active set S_k: signals whose code has a nonzero at atom k_idx ----
    S_k = np.where(Alpha[k_idx, :] != 0)[0]   # shape: (|S_k|,)

    if len(S_k) == 0:
        # Atom unused -> reinitialise to a random unit vector via the seeded rng
        d_k = rng.standard_normal(n)
        d_k /= np.linalg.norm(d_k)
        return d_k, np.zeros(N)

    # ---- Compute leave-one-out error matrix E_k on columns S_k -------------
    #
    # E_k[:,i] = X[:,i] - sum_{j != k} D[:,j] * Alpha[j,i]
    #          = X[:,i] - D @ Alpha[:,i]  +  D[:,k] * Alpha[k,i]
    #
    # The (+) term "adds back" atom k's contribution, giving us the error
    # that atom k alone is responsible for approximating.
    #
    X_clean_Sk  = np.nan_to_num(X[:, S_k], nan=0.0)           # (n, |S_k|)
    X_hat_Sk    = D @ Alpha[:, S_k]                            # (n, |S_k|)
    d_k_current = D[:, k_idx]                                  # (n,)
    alpha_k_Sk  = Alpha[k_idx, S_k]                            # (|S_k|,)

    # np.outer(d_k_current, alpha_k_Sk): shape (n, |S_k|)
    # outer[j,i] = d_k_current[j] * alpha_k_Sk[i]
    E_k_Sk = X_clean_Sk - X_hat_Sk + np.outer(d_k_current, alpha_k_Sk)

    # Observation masks for the S_k signals
    M_Sk = Masks[:, S_k]   # (n, |S_k|);  M_Sk[j,i]=1 iff entry j observed

    # ---- ALS Initialisation -------------------------------------------------
    d_k = d_k_current.copy()   # (n,)     start from current atom
    h_k = alpha_k_Sk.copy()    # (|S_k|,) start from current coefficients

    for _ in range(als_iters):

        # =====================================================================
        # ALS Step A: Fix d_k -> solve for h_k[i] independently for each i
        # =====================================================================
        # Vectorise: d_k_masked[j,i] = M_Sk[j,i] * d_k[j]
        # Broadcasting: d_k[:,None] shape (n,1) expands over |S_k| columns
        d_k_masked    = M_Sk * d_k[:, None]                    # (n, |S_k|)

        # Sum over axis=0 (the n spatial dimension) -> reduce to (|S_k|,)
        numerator_h   = np.sum(d_k_masked * E_k_Sk, axis=0)   # (|S_k|,)
        denominator_h = np.sum(d_k_masked ** 2,     axis=0)   # (|S_k|,)

        safe_denom_h  = np.where(denominator_h < 1e-10, 1.0, denominator_h)
        h_k           = numerator_h / safe_denom_h             # (|S_k|,)

        # =====================================================================
        # ALS Step B: Fix h_k -> solve for d_k[j] independently for each j
        # =====================================================================
        # Vectorise: h_k_masked[j,i] = M_Sk[j,i] * h_k[i]
        # Broadcasting: h_k[None,:] shape (1,|S_k|) expands over n rows
        h_k_masked    = M_Sk * h_k[None, :]                    # (n, |S_k|)

        # Sum over axis=1 (the |S_k| signal dimension) -> reduce to (n,)
        numerator_d   = np.sum(h_k_masked * E_k_Sk,          axis=1)  # (n,)
        # M_Sk[j,i] * h_k[i] * h_k[i] = h_k_masked[j,i] * h_k[i]
        denominator_d = np.sum(h_k_masked * h_k[None, :],    axis=1)  # (n,)

        safe_denom_d  = np.where(np.abs(denominator_d) < 1e-10, 1.0, denominator_d)
        d_k           = numerator_d / safe_denom_d             # (n,)

    # ---- Normalise d_k and absorb scale into h_k ----------------------------
    # MANDATORY:
    #   OMP selects atoms via ||(D_tilde[:,j])^T r|| / ||D_tilde[:,j]||.
    #   If d_k is not unit-norm, atom k's correlation is mis-scaled relative
    #   to all other atoms, breaking OMP's greedy selection in the next iter.
    #   Absorbing the scale into h_k preserves the rank-1 product d_k h_k^T.
    norm_d = np.linalg.norm(d_k)
    if norm_d > 1e-10:
        h_k = h_k * norm_d        # h_k[i] <- h_k[i] * ||d_k||_2
        d_k = d_k / norm_d        # d_k <- d_k / ||d_k||_2
    else:
        # Degenerate atom -> reinitialise randomly via the seeded rng
        d_k = rng.standard_normal(n)
        d_k /= np.linalg.norm(d_k)
        h_k = np.zeros(len(S_k))

    # Build full-length coefficient row; only S_k positions are nonzero
    h_k_full       = np.zeros(N)
    h_k_full[S_k]  = h_k   # (N,)

    return d_k, h_k_full   # (n,), (N,)


# =============================================================================
# PHASE 2C — Masked K-SVD Outer Loop
# =============================================================================

def masked_ksvd(X, Masks, K, s, n_iter=20, als_iters=15, seed=42, label="Masked K-SVD"):
    """
    Masked K-SVD: dictionary learning directly from incomplete observations.

    Outer loop alternates:
        (1) Masked Sparse Coding   — masked_omp on each of N signals
        (2) Masked Dictionary Update — ALS rank-1 update on each of K atoms
        (3) Log masked RMSE (observed entries ONLY)

    Parameters
    ----------
    X        : (n, N)  corrupted signals, NaN at missing positions
    Masks    : (n, N)  binary masks (1=observed, 0=missing)
    K        : int     number of dictionary atoms to learn
    s        : int     sparsity per signal
    n_iter   : int     outer K-SVD iterations
    als_iters: int     ALS steps per atom update
    seed     : int     RNG seed for dictionary initialisation

    Returns
    -------
    D      : (n, K)      learned dictionary (unit-norm columns)
    Alpha  : (K, N)      learned sparse codes
    errors : (n_iter,)   masked RMSE per outer iteration
    """
    rng = np.random.default_rng(seed)
    n, N = X.shape

    # Initialise from data patches (much better than pure random Gaussian)
    D = init_dictionary_from_data(X, Masks, K, rng)   # (n, K)

    Alpha  = np.zeros((K, N))
    errors = []

    print(f"\n  [{label}]")
    for iteration in range(n_iter):
        t0 = time.time()

        # ------------------------------------------------------------------
        # Step 1: Masked Sparse Coding  (N independent problems)
        # ------------------------------------------------------------------
        for i in range(N):
            Alpha[:, i] = masked_omp(X[:, i], D, Masks[:, i], s)

        # ------------------------------------------------------------------
        # Step 2: Masked Dictionary Update  (K independent ALS problems)
        # ------------------------------------------------------------------
        for k_idx in range(K):
            d_k_new, h_k_new = masked_ksvd_update_atom(
                D, Alpha, X, Masks, k_idx, als_iters=als_iters, rng=rng
            )
            D[:, k_idx]     = d_k_new
            Alpha[k_idx, :] = h_k_new

        # ------------------------------------------------------------------
        # Step 3: Masked RMSE — evaluate ONLY on observed entries
        # ------------------------------------------------------------------
        # The mask zeros out missing positions so they don't contribute to error.
        # This is the only fair metric: we never "know" the missing values.
        X_hat    = D @ Alpha                              # (n, N)
        X_obs    = np.nan_to_num(X, nan=0.0)             # NaN -> 0 (masked anyway)
        residual = Masks * (X_obs - X_hat)               # zero-out missing
        rmse     = np.sqrt(np.sum(residual ** 2) / max(Masks.sum(), 1))
        errors.append(rmse)

        dt = time.time() - t0
        print(f"    Iter {iteration+1:02d}/{n_iter}  |  masked RMSE = {rmse:.6f}  |  {dt:.2f}s")

    return D, Alpha, np.array(errors)


def baseline_ksvd(X, Masks, K, s, n_iter=20, als_iters=15, seed=42):
    """
    Baseline K-SVD: naively replaces missing entries with zeros (zero imputation).

    No masking in OMP or dictionary update. The algorithm treats imputed zeros
    as real observed data, which corrupts the error signal and biases the
    learned dictionary toward explaining the artificial zeros.

    Evaluated on the SAME masked RMSE metric as the novel algorithm, for
    a fair apples-to-apples comparison.
    """
    rng = np.random.default_rng(seed)
    n, N = X.shape

    X_imputed = np.nan_to_num(X, nan=0.0)         # missing -> 0
    all_ones  = np.ones_like(Masks)               # pretend everything is observed

    D = init_dictionary_from_data(X, Masks, K, rng)   # same init as masked

    Alpha  = np.zeros((K, N))
    errors = []

    print(f"\n  [Baseline K-SVD — zero imputation]")
    for iteration in range(n_iter):
        t0 = time.time()

        for i in range(N):
            Alpha[:, i] = masked_omp(X_imputed[:, i], D, all_ones[:, i], s)

        for k_idx in range(K):
            d_k_new, h_k_new = masked_ksvd_update_atom(
                D, Alpha, X_imputed, all_ones, k_idx, als_iters=als_iters,
                rng=rng,
            )
            D[:, k_idx]     = d_k_new
            Alpha[k_idx, :] = h_k_new

        # Evaluate on OBSERVED entries (same metric for fair comparison)
        X_hat    = D @ Alpha
        residual = Masks * (X_imputed - X_hat)
        rmse     = np.sqrt(np.sum(residual ** 2) / max(Masks.sum(), 1))
        errors.append(rmse)

        dt = time.time() - t0
        print(f"    Iter {iteration+1:02d}/{n_iter}  |  masked RMSE = {rmse:.6f}  |  {dt:.2f}s")

    return D, Alpha, np.array(errors)


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def compute_full_rmse(X_clean, D, Alpha):
    """RMSE over ALL signal entries including held-out missing ones."""
    return float(np.sqrt(np.mean((X_clean - D @ Alpha) ** 2)))


def compute_dict_coherence(D):
    """Mutual coherence: max |d_i^T d_j| for i != j.  Lower = more incoherent."""
    G = D.T @ D
    np.fill_diagonal(G, 0)
    return float(np.max(np.abs(G)))


def _format_value(value):
    """Compact formatting for plot labels and CSV-friendly strings."""
    if isinstance(value, float):
        if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-3):
            return f"{value:.2e}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _sanitize_token(token):
    """Make a file-name-safe token from an arbitrary value."""
    text = _format_value(token)
    text = text.replace(".", "p")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(ch if ch in allowed else "-" for ch in text)


def _param_tag(params, ordered_keys):
    """Build compact run identifier text for file names."""
    chunks = []
    for key in ordered_keys:
        if key in params:
            chunks.append(f"{key}-{_sanitize_token(params[key])}")
    return "__".join(chunks)


def _dict_rows_to_csv(rows, save_path):
    """Write list-of-dicts to CSV with stable column order."""
    if not rows:
        return

    preferred = [
        "combo_id", "seed", "n", "K_true", "K_learn", "s_true", "s",
        "N", "noise_std", "missing_frac", "n_iter", "als_iters",
        "masked_rmse_initial", "masked_rmse_final",
        "baseline_rmse_initial", "baseline_rmse_final",
        "full_rmse_masked", "full_rmse_baseline",
        "coherence_masked", "coherence_baseline",
        "avg_nnz_masked", "avg_nnz_baseline",
        "improvement_masked_pct", "improvement_baseline_pct", "final_gap_pct",
        "plot_path"
    ]
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = [key for key in preferred if key in all_keys]
    fieldnames += sorted(k for k in all_keys if k not in fieldnames)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_results(errors_masked, errors_baseline,
                 D_true, D_masked, D_baseline,
                 X_clean, Alpha_masked, Alpha_baseline,
                 save_path, run_params=None):
    """Five-panel diagnostic figure."""

    fig = plt.figure(figsize=(20, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    iters = np.arange(1, len(errors_masked) + 1)
    BLUE  = '#2563EB'
    RED   = '#DC2626'

    # ---- Panel A: Convergence comparison ------------------------------------
    ax = fig.add_subplot(gs[:, 0])
    ax.semilogy(iters, errors_masked,   'o-', color=BLUE, lw=2.2, ms=6,
                mfc='white', mew=1.8, label='Masked K-SVD  (novel — ALS update)')
    ax.semilogy(iters, errors_baseline, 's--', color=RED, lw=2.0, ms=6,
                mfc='white', mew=1.8, label='Baseline  (zero imputation)')
    ax.set_xlabel("K-SVD Outer Iteration", fontsize=12)
    ax.set_ylabel("Masked RMSE — observed entries (log scale)", fontsize=11)
    ax.set_title(
        "Convergence: Masked K-SVD vs Baseline\n"
        "(ALS Rank-1 atom update vs zero-imputation K-SVD)",
        fontsize=12, fontweight='bold'
    )
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(fontsize=10, loc='upper right')
    # Annotate final RMSE values
    # ax.annotate(f"Final: {errors_masked[-1]:.4f}",
    #             xy=(len(iters), errors_masked[-1]),
    #             xytext=(0.08, 0.14),
    #             textcoords='axes fraction',
    #             fontsize=9, color=BLUE,
    #             arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))
    # ax.annotate(f"Final: {errors_baseline[-1]:.4f}",
    #             xy=(len(iters), errors_baseline[-1]),
    #             xytext=(0.08, 0.90),
    #             textcoords='axes fraction',
    #             fontsize=9, color=RED,
    #             arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))

    # ---- Panel B: Learned dictionary atoms (masked K-SVD) -------------------
    K_show = min(20, D_masked.shape[1])
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(D_masked[:, :K_show].T, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1)
    ax.set_xlabel("Signal dimension  j", fontsize=11)
    ax.set_ylabel("Atom index  k",       fontsize=11)
    ax.set_title(f"Masked K-SVD — Learned Dictionary\n(first {K_show} atoms)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # ---- Panel C: Ground-truth dictionary atoms -----------------------------
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(D_true[:, :K_show].T, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1)
    ax.set_xlabel("Signal dimension  j", fontsize=11)
    ax.set_ylabel("Atom index  k",       fontsize=11)
    ax.set_title(f"Ground-Truth Dictionary\n(first {K_show} atoms)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # ---- Panel D: Signal reconstruction comparison --------------------------
    ax    = fig.add_subplot(gs[0, 2])
    n_sig = 5
    gap   = 4.0
    cmap  = plt.cm.tab10
    X_hat_m = D_masked   @ Alpha_masked
    X_hat_b = D_baseline @ Alpha_baseline
    offset  = 0
    for idx in range(n_sig):
        c = cmap(idx / 10)
        ax.plot(X_clean[:, idx]  + offset, '-',  color=c,    lw=1.8)
        ax.plot(X_hat_m[:, idx]  + offset, '--', color=BLUE, lw=1.3, alpha=0.85)
        ax.plot(X_hat_b[:, idx]  + offset, ':',  color=RED,  lw=1.3, alpha=0.85)
        offset += gap
    ax.plot([], [], 'k-',  lw=1.8, label='Ground truth')
    ax.plot([], [], '--', color=BLUE, lw=1.3, label='Masked K-SVD')
    ax.plot([], [], ':',  color=RED,  lw=1.3, label='Baseline')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title("Signal Reconstruction\n(5 random test signals)", fontsize=11)
    ax.set_xlabel("Dimension  j", fontsize=11)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)

    # ---- Panel E: Sparsity pattern of learned codes -------------------------
    ax     = fig.add_subplot(gs[1, 2])
    n_show = min(80, Alpha_masked.shape[1])
    ax.imshow(Alpha_masked[:, :n_show] != 0, aspect='auto',
              cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlabel(f"Signal index (first {n_show})", fontsize=11)
    ax.set_ylabel("Atom index  k", fontsize=11)
    ax.set_title("Sparsity Pattern — Masked K-SVD Codes\n(dark = nonzero)", fontsize=11)

    fig.suptitle(
        "Masked Dictionary Learning — Phase 1 & 2 Proof-of-Concept\n"
        "Novel ALS Rank-1 atom update vs naive zero-imputation baseline",
        fontsize=14, fontweight='bold', y=1.01
    )

    if run_params is not None:
        ordered_keys = [
            "combo_id", "seed", "n", "K_true", "K_learn", "s_true", "s",
            "N", "noise_std", "missing_frac", "n_iter", "als_iters",
            "masked_rmse_final", "baseline_rmse_final",
            "full_rmse_masked", "full_rmse_baseline"
        ]
        lines = []
        for key in ordered_keys:
            if key in run_params:
                lines.append(f"{key}={_format_value(run_params[key])}")
        for key in sorted(k for k in run_params.keys() if k not in ordered_keys):
            lines.append(f"{key}={_format_value(run_params[key])}")

        fig.text(
            0.012, 0.012,
            "Run parameters\n" + "\n".join(lines),
            ha="left", va="bottom", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.3, edgecolor="#BFBFBF")
        )

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved -> {save_path}")


def _run_single_config(config, seed, output_dir, combo_id="single"):
    """Run one configuration end-to-end and persist its diagnostic figure."""
    rng = np.random.default_rng(seed)

    n            = int(config["n"])
    K_true       = int(config["K_true"])
    K_learn      = int(config["K_learn"])
    s_true       = int(config.get("s_true", config["s"]))
    s            = int(config["s"])
    N            = int(config["N"])
    noise_std    = float(config["noise_std"])
    missing_frac = float(config["missing_frac"])
    n_iter       = int(config["n_iter"])
    als_iters    = int(config["als_iters"])

    print("=" * 80)
    print(f"  [{combo_id}] seed={seed}  n={n} K_true={K_true} K_learn={K_learn} s_true={s_true} s={s}")
    print(f"  N={N}  noise={noise_std}  missing={missing_frac}  n_iter={n_iter}  als_iters={als_iters}")

    D_true           = generate_ground_truth_dictionary(n, K_true, rng)
    X_clean, _       = generate_sparse_signals(D_true, N, s_true, noise_std, rng)
    X_corrupt, Masks = corrupt_signals(X_clean, missing_frac, rng)

    D_masked, Alpha_masked, err_masked = masked_ksvd(
        X_corrupt, Masks,
        K=K_learn, s=s, n_iter=n_iter, als_iters=als_iters,
        seed=seed, label=f"Masked K-SVD ({combo_id})"
    )

    D_base, Alpha_base, err_base = baseline_ksvd(
        X_corrupt, Masks,
        K=K_learn, s=s, n_iter=n_iter, als_iters=als_iters, seed=seed
    )

    fr_masked  = compute_full_rmse(X_clean, D_masked, Alpha_masked)
    fr_base    = compute_full_rmse(X_clean, D_base,   Alpha_base)
    coh_masked = compute_dict_coherence(D_masked)
    coh_base   = compute_dict_coherence(D_base)
    nnz_m      = float((Alpha_masked != 0).sum(axis=0).mean())
    nnz_b      = float((Alpha_base   != 0).sum(axis=0).mean())

    impr_m = (1.0 - err_masked[-1] / err_masked[0]) * 100.0
    impr_b = (1.0 - err_base[-1]   / err_base[0])   * 100.0
    gap    = (1.0 - err_masked[-1] / err_base[-1])  * 100.0

    row = {
        "combo_id": combo_id,
        "seed": seed,
        "n": n,
        "K_true": K_true,
        "K_learn": K_learn,
        "s_true": s_true,
        "s": s,
        "N": N,
        "noise_std": noise_std,
        "missing_frac": missing_frac,
        "n_iter": n_iter,
        "als_iters": als_iters,
        "masked_rmse_initial": float(err_masked[0]),
        "masked_rmse_final": float(err_masked[-1]),
        "baseline_rmse_initial": float(err_base[0]),
        "baseline_rmse_final": float(err_base[-1]),
        "full_rmse_masked": float(fr_masked),
        "full_rmse_baseline": float(fr_base),
        "coherence_masked": float(coh_masked),
        "coherence_baseline": float(coh_base),
        "avg_nnz_masked": nnz_m,
        "avg_nnz_baseline": nnz_b,
        "improvement_masked_pct": float(impr_m),
        "improvement_baseline_pct": float(impr_b),
        "final_gap_pct": float(gap),
    }

    tag_keys = [
        "n", "K_true", "K_learn", "s_true", "s", "N",
        "noise_std", "missing_frac", "n_iter", "als_iters"
    ]
    run_tag = _param_tag(row, tag_keys)
    plot_path = os.path.join(output_dir, f"{combo_id}__seed-{seed}__{run_tag}.png")
    # row["plot_path"] = plot_path

    plot_results(
        err_masked, err_base,
        D_true, D_masked, D_base,
        X_clean, Alpha_masked, Alpha_baseline=Alpha_base,
        save_path=plot_path, run_params=row
    )

    print(
        "  Final RMSE  |  "
        f"Masked={row['masked_rmse_final']:.6f}  "
        f"Baseline={row['baseline_rmse_final']:.6f}  "
        f"Gap={row['final_gap_pct']:+.2f}%"
    )

    return row


def _aggregate_rows(rows, group_keys):
    """Aggregate sweep rows over seeds for each hyperparameter configuration."""
    metric_keys = [
        "masked_rmse_final", "baseline_rmse_final",
        "full_rmse_masked", "full_rmse_baseline",
        "coherence_masked", "coherence_baseline",
        "final_gap_pct"
    ]
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    agg_rows = []
    for key_values, group in grouped.items():
        out = {k: v for k, v in zip(group_keys, key_values)}
        out["num_seeds"] = len(group)
        for metric in metric_keys:
            values = np.array([float(item[metric]) for item in group], dtype=float)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std(ddof=0))
        agg_rows.append(out)

    agg_rows.sort(key=lambda item: item["masked_rmse_final_mean"])
    return agg_rows


def _plot_sweep_summary(agg_rows, group_keys, save_path, top_k=12):
    """Save a ranking plot of the best sweep configurations."""
    if not agg_rows:
        return

    top = agg_rows[:min(top_k, len(agg_rows))]
    labels = [", ".join(f"{k}={_format_value(row[k])}" for k in group_keys) for row in top]
    y = np.arange(len(top))

    masked_means = np.array([row["masked_rmse_final_mean"] for row in top], dtype=float)
    masked_stds  = np.array([row["masked_rmse_final_std"]  for row in top], dtype=float)
    base_means   = np.array([row["baseline_rmse_final_mean"] for row in top], dtype=float)
    base_stds    = np.array([row["baseline_rmse_final_std"]  for row in top], dtype=float)

    fig_h = max(6.0, 0.5 * len(top) + 2.5)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    bar_h = 0.36

    ax.barh(y + bar_h / 2, base_means,   height=bar_h, color="#DC2626", alpha=0.85, label="Baseline")
    ax.barh(y - bar_h / 2, masked_means, height=bar_h, color="#2563EB", alpha=0.85, label="Masked K-SVD")

    ax.errorbar(base_means,   y + bar_h / 2, xerr=base_stds,   fmt="none", ecolor="#7F1D1D", elinewidth=1.0)
    ax.errorbar(masked_means, y - bar_h / 2, xerr=masked_stds, fmt="none", ecolor="#1E3A8A", elinewidth=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Final masked RMSE (lower is better)", fontsize=11)
    ax.set_title("Hyperparameter Sweep Summary (Top Configurations)", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right")

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary figure saved -> {save_path}")


def run_hyperparameter_sweep(base_config, sweep_grid, seeds, output_dir):
    """
    Run grid search and export:
      - One diagnostic plot per (configuration, seed)
      - Per-run CSV
      - Aggregated CSV over seeds
      - Summary ranking plot
    """
    group_keys = list(sweep_grid.keys())
    grid_values = [sweep_grid[key] for key in group_keys]
    combos = [dict(zip(group_keys, values)) for values in itertools.product(*grid_values)]

    run_plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(run_plot_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("  SWEEP START")
    print("=" * 80)
    print(f"  Configurations : {len(combos)}")
    print(f"  Seeds          : {len(seeds)}")
    print(f"  Total runs     : {len(combos) * len(seeds)}")

    rows = []
    run_idx = 0
    total_runs = len(combos) * len(seeds)
    for combo_i, combo in enumerate(combos, start=1):
        cfg = dict(base_config)
        cfg.update(combo)
        combo_id = f"cfg{combo_i:03d}"
        print("\n" + "-" * 80)
        print(f"Configuration {combo_i:03d}/{len(combos)} -> {combo}")
        for seed in seeds:
            run_idx += 1
            print(f"Run {run_idx:03d}/{total_runs}  (combo={combo_id}, seed={seed})")
            row = _run_single_config(cfg, seed=seed, output_dir=run_plot_dir, combo_id=combo_id)
            rows.append(row)

    run_csv = os.path.join(output_dir, "sweep_runs.csv")
    _dict_rows_to_csv(rows, run_csv)
    print(f"Per-run metrics saved -> {run_csv}")

    agg_rows = _aggregate_rows(rows, group_keys=group_keys)
    agg_csv = os.path.join(output_dir, "sweep_aggregate.csv")
    _dict_rows_to_csv(agg_rows, agg_csv)
    print(f"Aggregated metrics saved -> {agg_csv}")

    summary_plot = os.path.join(output_dir, "sweep_summary.png")
    _plot_sweep_summary(agg_rows, group_keys=group_keys, save_path=summary_plot, top_k=12)

    if agg_rows:
        best = agg_rows[0]
        best_desc = ", ".join(f"{k}={_format_value(best[k])}" for k in group_keys)
        print("\n" + "=" * 80)
        print("  BEST CONFIGURATION")
        print("=" * 80)
        print(f"  {best_desc}")
        print(f"  mean final masked RMSE : {best['masked_rmse_final_mean']:.6f}")
        print(f"  mean final baseline RMSE: {best['baseline_rmse_final_mean']:.6f}")
        print(f"  mean final gap (%)       : {best['final_gap_pct_mean']:+.2f}")

    return rows, agg_rows


# =============================================================================
# MAIN
# =============================================================================

def main():
    base_config = {
        "n": 20,
        "K_true": 50,
        "K_learn": 50,
        "s_true": 3,      # data-generation sparsity (kept fixed unless explicitly swept)
        "s": 3,           # coding sparsity used by OMP
        "N": 500,
        "noise_std": 0.15,
        "missing_frac": 0.30,
        "n_iter": 20,
        "als_iters": 15,
    }

    # Toggle between one run and full sweep.
    RUN_SWEEP = False

    out_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs"
    )

    if RUN_SWEEP:
        sweep_grid = {
            "K_learn": [40, 50, 60],
            "s": [2, 3, 4],
            "n_iter": [15, 30],
            "als_iters": [8, 15],
        }
        sweep_seeds = [69]
        sweep_out = os.path.join(out_root, "sweep")
        run_hyperparameter_sweep(base_config, sweep_grid, sweep_seeds, sweep_out)
        return

    single_row = _run_single_config(
        base_config,
        seed=42,
        output_dir=out_root,
        combo_id="single"
    )
    print("\nSingle run complete.")
    
    # print(f"Plot saved -> {single_row['plot_path']}")


if __name__ == "__main__":
    main()
