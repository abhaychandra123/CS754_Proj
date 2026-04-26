# CROWN-Inpaint
## Confidence-weighted Regime-aware Overlap-Nonlocal Inpainting
### Full Method Specification (the project bible)

Document version: 1.0
Date: 2026-04-25
Scope: complete mathematical specification, justification, implementation plan, and validation protocol for a publishable upgrade to iterative masked K-SVD inpainting.

---

## Table of Contents

1. Executive Summary
2. Problem Statement and Notation
3. Why This Method Exists (motivation and inspirations)
4. The Three Components (high-level)
5. Mathematical Foundations
6. Detailed Component Specifications
   - 6.1 Confidence Map
   - 6.2 Regime Map
   - 6.3 Smooth-Prior Branch
   - 6.4 Confidence-Weighted Masked OMP
   - 6.5 Nonlocal Coefficient Coupling
   - 6.6 Overlap Averaging Reconstruction
   - 6.7 Regime-Aware Fusion and Hard Constraint
   - 6.8 Optional Stochastic Manifold Correction
7. The Full Algorithm
8. Theoretical Properties and Sanity Guarantees
9. Implementation Guide
10. Hyperparameters and Schedules
11. Experimental Protocol
12. Diagnostics and Failure Modes
13. Connection to Prior Work
14. Validation Checklist
15. Glossary

---

## 1. Executive Summary

CROWN-Inpaint is an iterative inpainting method that fuses three priors inside one solver:

1. A **smooth-prior branch** that is provably good in low-texture regions.
2. A **nonlocal grouped sparse-coding branch** that propagates structure and texture from observed regions into the hole.
3. An **optional stochastic manifold-correction step** that prevents drift and escapes local minima.

The fusion is driven by two pixel-wise scalar maps:

- A **regime map** $r \in [0,1]$ that says how textured each location is.
- A **confidence map** $c \in [0,1]$ that says how reliable each currently estimated hole pixel is.

The optimization always preserves observed pixels exactly via a hard Dirichlet constraint, so the method cannot make observed regions worse. Improvements come from how the hole is filled.

This is intentionally a **non-deep, optimization-first** method. It is implementable directly inside your existing K-SVD codebase, and it has clear ablation handles for a paper.

---

## 2. Problem Statement and Notation

We work on grayscale or single-channel images of size $H \times W$. Extension to color is via per-channel processing or joint patch dictionaries.

| Symbol | Meaning |
|---|---|
| $y \in \mathbb{R}^{H \times W}$ | Observed image, hole pixels filled with arbitrary placeholder |
| $M \in \{0,1\}^{H \times W}$ | Mask, $M_i = 1$ if pixel $i$ is observed, $0$ if missing |
| $\Omega = \{i : M_i = 1\}$ | Observed set |
| $\Omega^c$ | Hole set |
| $u^t \in \mathbb{R}^{H \times W}$ | Current reconstruction at iteration $t$ |
| $D \in \mathbb{R}^{n \times K}$ | Dictionary, $n = p^2$ for patch size $p$, $K$ atoms |
| $R_p$ | Patch extraction operator at location $p$, returns vector of length $n$ |
| $\alpha_p \in \mathbb{R}^K$ | Sparse code for patch at $p$ |
| $r^t \in [0,1]^{H \times W}$ | Regime map, $0 = $ smooth, $1 = $ textured |
| $c^t \in [0,1]^{H \times W}$ | Confidence map for hole pixels |
| $\Pi_M(u, y)$ | Projection enforcing $u_i = y_i$ for $i \in \Omega$ |

Patch model: each patch is approximately representable as a sparse combination of dictionary atoms,

$$
R_p u \approx D \alpha_p, \quad \|\alpha_p\|_0 \le s.
$$

---

## 3. Why This Method Exists (motivation and inspirations)

### 3.1 The empirical problem
Iterative masked K-SVD inpainting works well on textured regions but fails to beat biharmonic on smooth regions. Biharmonic in turn smears textures.

### 3.2 The structural reason
- **Smooth regions** are governed by elliptic and biharmonic interpolation theory. Biharmonic completion is the unique minimizer of bending energy with Dirichlet and Neumann boundary conditions, so for smooth holes it is essentially optimal.
- **Textured regions** require atom-level transfer of patterns from the boundary into the hole. Sparse-coding overlap averaging is exactly the right mechanism.
- **No single prior dominates both regimes.**

### 3.3 Inspirations from the literature
- **K-SVD and patch sparse coding** (Aharon-Elad-Bruckstein 2006; Elad-Aharon 2006): sparse representations over learned dictionaries.
- **Mairal et al. 2008**: sparse coding handles missing data well when the patch model is reliable.
- **Centralized Sparse Representation** (Dong-Zhang-Shi 2011) and **Group-Based Sparse Representation** (Zhang-Zhao-Gao 2014): nonlocal coefficient coupling improves both stability and quality.
- **Chan-Shen 2002**: variational and biharmonic models for nontexture inpainting.
- **Manifold-constraint diffusion solvers** (Chung et al. 2022) and **Stochastic generative PnP** (Park et al. 2026): off-manifold drift in iterative inverse-problem solvers can be partially fixed by light stochastic correction.
- **Bertalmio et al. 2000**: structure transport from boundary inward.

CROWN-Inpaint takes the strongest single idea from each line of work and unifies them inside one optimization-first iterative solver.

### 3.4 What is genuinely new
- The combined use of:
  - a per-pixel **regime map** that gates two priors,
  - a per-pixel **confidence map** that adaptively weights the influence of estimated hole pixels in the sparse-coding objective,
  - **nonlocal coefficient coupling** with **hard observation constraints**, and
  - an **optional stochastic correction** that does not require any pretrained network.

The novelty is the unified optimization design and the confidence schedule. No single component is itself new in isolation, but no published method, to my knowledge, fuses them in this exact way for inpainting.

---

## 4. The Three Components (high-level)

### 4.1 Smooth-prior branch
Performs a small number of harmonic or biharmonic relaxation steps inside the hole, holding observed pixels fixed.

### 4.2 Nonlocal grouped sparse branch
Solves a confidence-weighted patch sparse coding problem with optional nonlocal coefficient coupling over similar patches, then reconstructs the image by overlap averaging.

### 4.3 Optional manifold-correction branch
Injects small noise inside the hole and applies a light denoising step (BM3D, non-local means, or TV) followed by reprojection to observed data. Used every $k$ iterations.

---

## 5. Mathematical Foundations

### 5.1 Variational view
We can view the entire algorithm as alternating minimization of an energy of the form

$$
E(u, \{\alpha_p\}) = \sum_p \big\| W_p \odot (R_p u - D\alpha_p) \big\|_2^2
+ \lambda \sum_{p, q \in \mathcal{N}(p)} w_{pq} \|\alpha_p - \alpha_q\|_1
+ \mu \, J_{\text{smooth}}(u; r)
+ \iota_{\Omega}(u; y)
$$

with constraints $\|\alpha_p\|_0 \le s$ for all $p$, where:

- $W_p \in [0,1]^n$ is the per-patch confidence-weighted mask.
- $\mathcal{N}(p)$ is a small nearest-neighbor set in patch space.
- $J_{\text{smooth}}(u; r)$ is a smoothness penalty whose strength is locally modulated by the regime map.
- $\iota_{\Omega}(u; y)$ is the indicator function enforcing $u_i = y_i$ for all $i \in \Omega$.

Each branch in the algorithm corresponds to one block-coordinate descent step on this energy, with closed-form or near-closed-form updates.

### 5.2 Weighted patch sparse coding subproblem
For each patch $p$, holding $u$ fixed:

$$
\min_{\alpha_p} \big\| W_p^{1/2} \odot (R_p u - D\alpha_p) \big\|_2^2
\quad \text{s.t.} \quad \|\alpha_p\|_0 \le s.
$$

Letting $\widetilde{D}_p = \mathrm{diag}(W_p^{1/2}) D$ and $\widetilde{x}_p = W_p^{1/2} \odot R_p u$, this is a standard OMP problem on $(\widetilde{D}_p, \widetilde{x}_p)$. **Confidence weighting is therefore a clean extension of masked OMP**, requiring only diagonal scaling of the dictionary and target per patch.

### 5.3 Image update subproblem
Holding $\{\alpha_p\}$ fixed, the image update minimizes

$$
\sum_p \| W_p \odot (R_p u - D\alpha_p) \|_2^2 + \mu J_{\text{smooth}}(u; r),
$$

subject to $u_i = y_i$ for $i \in \Omega$. We approximate this by:

1. Overlap-averaging the patch reconstructions to get $u_\text{sparse}$.
2. Computing a smoothing update $u_\text{smooth}$ with $\mu$ effectively localized via $r$.
3. Fusing using the regime map and applying $\Pi_M$.

### 5.4 Hard observation constraint $\Pi_M$
For any candidate $\bar{u}$,

$$
\Pi_M(\bar{u}, y) = M \odot y + (1 - M) \odot \bar{u}.
$$

This guarantees $\Pi_M(\bar{u}, y)$ exactly matches observed pixels.

### 5.5 Energy descent guarantee (informal)
Each subproblem is solved to (approximate) optimality given the others fixed, and the projection $\Pi_M$ does not increase the data-fidelity energy on $\Omega$ since it sets observed pixels to the target value $y$. Therefore the algorithm exhibits monotone descent on the energy components it can directly minimize. We do not claim global optimality because $\ell_0$ sparse coding is NP-hard, but stationary-point convergence under standard assumptions follows the same structure as classical K-SVD inpainting.

---

## 6. Detailed Component Specifications

### 6.1 Confidence Map $c^t$

#### 6.1.1 Goal
Tell the sparse coder how reliable each missing pixel’s current estimate is.

#### 6.1.2 Construction (per missing pixel $i \in \Omega^c$)
We combine three signals:

1. **Overlap variance** $v_i$: across all patches that cover pixel $i$, compute the variance of their reconstructions at $i$. High variance $\Rightarrow$ low confidence.
2. **Boundary distance** $d_i$: shortest distance from $i$ to $\Omega$. Far inside the hole $\Rightarrow$ low confidence.
3. **Iteration progress** $t / T$. We allow trust in hole estimates to grow over iterations.

We define:

$$
c_i^{\text{var}}    = \exp\!\big(-v_i / \tau_v^2\big), \qquad
c_i^{\text{geom}}   = \exp\!\big(-d_i / \rho\big), \qquad
c_i^{\text{sched}}  = \min\!\big(1, c_0 + \beta\, t\big).
$$

Combined confidence:

$$
c_i^t = c_i^{\text{var}} \cdot c_i^{\text{geom}} \cdot c_i^{\text{sched}}, \qquad i \in \Omega^c.
$$

For observed pixels we set $c_i^t = 1$, since they are kept fixed by $\Pi_M$ regardless.

Default scalars:
- $c_0 = 0.05$, $\beta = 0.15$
- $\tau_v$ adapts as the running median of $\sqrt{v_i}$ over the hole
- $\rho$ ~ patch radius (for $p = 8$, use $\rho = 4$)

#### 6.1.3 Why this shape
- Multiplicative form: any single signal having low confidence collapses the product, which is the desired conservative behavior early in the iterations.
- The schedule term is what makes confidence grow over time so that later iterations can integrate hole information without re-fitting biharmonic smearing.

### 6.2 Regime Map $r^t$

#### 6.2.1 Goal
Decide how textured each pixel’s neighborhood is, so the fusion can blend smoothly between the smooth and sparse branches.

#### 6.2.2 Cues (computed on observed image only, then propagated)

For each observed pixel $i \in \Omega$, compute three local features in a small window $\mathcal{W}_i$ (typical size $9 \times 9$):

1. **Gradient energy** $g_i = \frac{1}{|\mathcal{W}_i|} \sum_{j \in \mathcal{W}_i} \|\nabla y_j\|_2$.
2. **Structure-tensor anisotropy**: from $J = \sum_j \nabla y_j (\nabla y_j)^\top$, eigenvalues $\lambda_1 \ge \lambda_2 \ge 0$, define
   $$
   a_i = \frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2 + \epsilon}.
   $$
   This isolates oriented edges from isotropic texture.
3. **Spectral entropy** $h_i$ of the windowed DCT magnitude spectrum, normalized to $[0,1]$. Smooth $\Rightarrow$ low entropy.

Define the raw regime score:

$$
\tilde{r}_i = \omega_g\, \mathrm{norm}(g_i) + \omega_a\, a_i + \omega_h\, h_i.
$$

Default $\omega_g = 0.4, \omega_a = 0.3, \omega_h = 0.3$, all renormalized to sum to one.

#### 6.2.3 Propagation into the hole
For $i \in \Omega^c$, we cannot trust hole intensities. Define $r$ on the hole by **boundary-conditional propagation**:

$$
r^t_i = \frac{\sum_{j \in \Omega} \kappa(\|i - j\|; \sigma) \, \tilde{r}_j}
            {\sum_{j \in \Omega} \kappa(\|i - j\|; \sigma)},
$$

where $\kappa$ is an isotropic Gaussian kernel with a hole-aware truncation (we only sum over observed pixels within a fixed radius). For efficiency, use a fast marching extrapolation or an inpaint-style harmonic extension of $\tilde{r}$ from $\Omega$ into $\Omega^c$.

#### 6.2.4 Smoothing
Pass $r^t$ through a small Gaussian blur (sigma ~ patch radius) to avoid sharp spatial fusion seams.

### 6.3 Smooth-Prior Branch

#### 6.3.1 Update rule
We run $K_s$ Jacobi iterations of harmonic relaxation on the hole, holding observed pixels fixed. For pixel $i \in \Omega^c$:

$$
u_{\text{smooth}}^{(\ell+1)}(i) = \frac{1}{|\mathcal{N}_4(i)|}
\sum_{j \in \mathcal{N}_4(i)} u_{\text{smooth}}^{(\ell)}(j),
$$

initialized from $u^{t-1}$. Observed pixels keep $y_i$ values throughout. Typical $K_s = 5$ to $20$.

For initialization at iteration 0 we use a full biharmonic inpaint as in Chan-Shen 2002, since it is the optimal smooth interpolant in this setting. That heavier solve only happens once.

#### 6.3.2 Why harmonic for inner iterations
Each outer iteration does not need a fresh full biharmonic solve. A few harmonic Jacobi steps suffice because the global smooth structure has already been set at $t = 0$ and only local adjustments are needed.

### 6.4 Confidence-Weighted Masked OMP

#### 6.4.1 Per-patch problem
For patch position $p$, given $W_p \in [0,1]^n$, current estimate $x_p = R_p u$, and dictionary $D$, solve:

$$
\min_{\alpha_p} \big\| W_p^{1/2} \odot (x_p - D\alpha_p) \big\|_2^2
\quad \text{s.t.} \quad \|\alpha_p\|_0 \le s.
$$

#### 6.4.2 Implementation
Define $w = W_p^{1/2}$. Replace OMP inputs:

$$
\widetilde{D} = \mathrm{diag}(w)\, D, \qquad \widetilde{x} = w \odot x_p.
$$

Run standard OMP on $(\widetilde{D}, \widetilde{x})$. Recovery is identical to the existing masked OMP code path, with the binary mask replaced by continuous weights. For pixel-wise efficient evaluation, normalize columns of $\widetilde{D}$ inside OMP (they are no longer unit-norm after diagonal scaling). This is the only nontrivial change required in your current `masked_omp` routine.

#### 6.4.3 Equivalence to weighted least squares
This is exactly weighted least squares restricted to the active support $\mathcal{S}$:

$$
\alpha_{\mathcal{S}} = (D_{\mathcal{S}}^\top \mathrm{diag}(W_p) D_{\mathcal{S}})^{-1} D_{\mathcal{S}}^\top \mathrm{diag}(W_p) x_p.
$$

OMP is a greedy approximation to this with $|\mathcal{S}| \le s$.

### 6.5 Nonlocal Coefficient Coupling

#### 6.5.1 Why
Independent OMP can pick different atoms for very similar patches. CSR/GSR-style coupling regularizes this, leading to better and more stable sparse codes.

#### 6.5.2 Construction
For each patch $p$, find the $K_{nl}$ nearest neighbor patches $\mathcal{N}(p)$ in patch-vector space using either:
- Euclidean distance restricted to observed coordinates of $p$, or
- a fast approximate nearest-neighbor index over the patch set.

Define weights $w_{pq} = \exp(-\|x_p - x_q\|_2^2 / h^2)$ over coordinates that are confidently observed in both patches (uses $W_p \cdot W_q$).

Compute the **group center**

$$
\beta_p = \frac{1}{Z_p} \sum_{q \in \mathcal{N}(p)} w_{pq}\, \alpha_q, \qquad
Z_p = \sum_q w_{pq}.
$$

Then re-estimate $\alpha_p$ via the centralized refinement:

$$
\alpha_p^\star = \arg\min_{\alpha} \big\| W_p^{1/2} \odot (x_p - D\alpha) \big\|_2^2 + \lambda \|\alpha - \beta_p\|_1.
$$

This is now a **convex** problem and admits ISTA/FISTA with closed-form soft-thresholding around $\beta_p$. Because we already have an OMP support estimate, we restrict the refinement to that support, making it inexpensive.

#### 6.5.3 Frequency
We do not run nonlocal coupling every iteration. Once every two outer iterations is enough to stabilize without significantly increasing runtime. The first iteration uses pure OMP.

### 6.6 Overlap Averaging Reconstruction

#### 6.6.1 Standard form
$$
\widehat{u}_{\text{sparse}}(i) = \frac{\sum_{p:\, i \in p} (D \alpha_p)_{i - p}}{\sum_{p:\, i \in p} 1}.
$$

#### 6.6.2 Confidence-weighted form (optional)
$$
\widehat{u}_{\text{sparse}}(i) = \frac{\sum_{p:\, i \in p} q_p (D \alpha_p)_{i - p}}{\sum_{p:\, i \in p} q_p},
$$

where $q_p$ is a per-patch reliability score, e.g. fraction of observed pixels in patch $p$ or inverse residual $\|W_p \odot (x_p - D\alpha_p)\|_2^2$. For simplicity we recommend uniform averaging in v1 and adding $q_p$ in ablations.

### 6.7 Regime-Aware Fusion and Hard Constraint

Compose the two branches using the regime map and project:

$$
\bar{u}^t = (1 - r^t) \odot u_{\text{smooth}}^t + r^t \odot u_{\text{sparse}}^t,
$$
$$
u^{t+1} = M \odot y + (1 - M) \odot \bar{u}^t.
$$

**Properties**:

1. Pixel-wise convex combination, so values stay in image range if both branches are clipped to $[0,1]$.
2. Hard constraint preserves all observed pixels exactly, every iteration.
3. Smooth and sparse never have to “fight” over observed pixels.

### 6.8 Optional Stochastic Manifold Correction

#### 6.8.1 Motivation
Iterative methods can lock in mistakes. A small stochastic perturbation followed by a denoising step can:

- escape local minima,
- average out per-iteration bias,
- re-project onto the natural-image manifold.

This is inspired by manifold-constraint diffusion solvers and stochastic plug-and-play, but does not require any pretrained deep model.

#### 6.8.2 Update
Every $k$ outer iterations:

1. Sample $\eta \sim \mathcal{N}(0, \sigma_t^2 I)$ on hole pixels only.
2. Form $u^t_{\text{noisy}} = u^t + (1 - M) \odot \eta$.
3. Apply a denoising operator $\mathcal{T}_{\sigma_t}$ to $u^t_{\text{noisy}}$. Choices:
   - BM3D with noise level $\sigma_t$,
   - non-local means,
   - small TV-prox step,
   - or any other plug-and-play denoiser.
4. Re-apply the hard constraint:
   $$
   u^t \leftarrow M \odot y + (1 - M) \odot \mathcal{T}_{\sigma_t}(u^t_{\text{noisy}}).
   $$

#### 6.8.3 Schedule
Decreasing schedule, e.g. $\sigma_t = \sigma_0 \cdot \gamma^{t-1}$ with $\sigma_0 = 0.04$, $\gamma = 0.7$. Apply every 2 outer iterations after iteration 1.

#### 6.8.4 Why this is safe
Because of the hard constraint, the correction never modifies observed pixels. Because the denoiser is a contraction in many natural image priors, repeated applications do not drift. The decreasing $\sigma_t$ enforces convergence-like behavior.

---

## 7. The Full Algorithm

### 7.1 Inputs and outputs
- Inputs: $y$, $M$, dictionary $D$ (pretrained or trained on observed patches), patch size $p$, sparsity $s$, iteration count $T$, optional manifold flag and schedule, fusion hyperparameters.
- Outputs: final reconstruction $u^T$, optionally diagnostics including per-iteration confidence and regime maps.

### 7.2 Initialization
1. Train or load dictionary $D$ on observed patches via masked K-SVD. (You already have this.)
2. Compute initial fill $u^0 = \text{biharmonic}(y, M)$.
3. Compute regime map $r$ on $\Omega$ from $y$, propagate to $\Omega^c$, then smooth.

### 7.3 Outer loop (for $t = 1, \dots, T$)
1. Compute confidence map $c^t$ from current $u^{t-1}$, $M$, and the schedule.
2. **Sparse branch**:
   1. For each patch position $p$, build $W_p$ from $M$ and $c^t$.
   2. Run confidence-weighted OMP to get $\alpha_p$.
   3. If $t$ matches the nonlocal cadence, run nonlocal coefficient coupling to refine $\alpha_p$.
   4. Reconstruct $u_{\text{sparse}}^t$ by overlap averaging.
3. **Smooth branch**: run $K_s$ harmonic Jacobi steps from $u^{t-1}$ to get $u_{\text{smooth}}^t$.
4. **Fuse**: $\bar{u}^t = (1 - r) \odot u_{\text{smooth}}^t + r \odot u_{\text{sparse}}^t$.
5. **Project**: $u^t = M \odot y + (1 - M) \odot \bar{u}^t$.
6. **Optional manifold correction** (if enabled and on schedule):
   $u^t \leftarrow \Pi_M\big(\mathcal{T}_{\sigma_t}(u^t + (1-M)\odot \eta), y\big)$.
7. Optionally log diagnostics: PSNR, SSIM, hole-PSNR, boundary MAE, regime and confidence histograms.

### 7.4 Termination
Stop after $T$ iterations or when relative change $\|u^t - u^{t-1}\|_2 / \|u^{t-1}\|_2 < \epsilon_\text{stop}$ is below a tolerance (e.g. $10^{-4}$).

### 7.5 Pseudocode
```
Inputs: y, M, D, p, s, T, K_s, K_nl, fusion params, manifold params

# Phase A: setup
u <- biharmonic_inpaint(y, M)
r <- compute_regime_map(y, M)        # observed cues + propagation + smoothing

# Phase B: outer iterations
for t in 1..T:
    c <- compute_confidence_map(u, M, t)        # 6.1

    # Sparse branch
    alpha = []
    for each patch p:
        W_p <- build_patch_weights(M, c, p)     # 6.4
        alpha_p <- weighted_OMP(R_p u, D, W_p, s)
        alpha.append(alpha_p)

    if (t mod nonlocal_cadence == 0):
        alpha <- nonlocal_refine(alpha, patches, D, W_p, lambda)  # 6.5

    u_sparse <- overlap_average(D, alpha, image_shape)            # 6.6

    # Smooth branch
    u_smooth <- harmonic_relax(u, M, K_s)                        # 6.3

    # Fuse and constrain
    u_bar <- (1 - r) * u_smooth + r * u_sparse                   # 6.7
    u <- M * y + (1 - M) * u_bar

    # Optional manifold correction
    if manifold_enabled and (t mod manifold_cadence == 0):
        eta <- gaussian_noise(sigma_t, shape=image_shape)
        u_noisy <- u + (1 - M) * eta
        u <- M * y + (1 - M) * denoise(u_noisy, sigma_t)

    if relative_change_below_eps(u): break

return u
```

---

## 8. Theoretical Properties and Sanity Guarantees

### 8.1 Hard observation guarantee
For all $t$, $u^t_i = y_i$ for $i \in \Omega$. Direct from $\Pi_M$.

### 8.2 No-regression property in smooth limit
If $r \equiv 0$, the algorithm reduces to repeated harmonic relaxation initialized from biharmonic, plus manifold correction. In the smooth-only setting, the harmonic limit on the hole equals the harmonic completion of $y|_\Omega$. With biharmonic init and Dirichlet matching $y$, the iterate stays in the family of harmonic-like extensions, so we cannot strictly worse than biharmonic by more than the fusion-induced perturbation.

### 8.3 Texture transport in textured regime
If $r \equiv 1$, the algorithm reduces to confidence-weighted iterative masked K-SVD with optional nonlocal coupling. By the standard sparse-dictionary patch model, atoms straddling the hole boundary transport observed structure into the hole each iteration, and overlap averaging smooths the transitions.

### 8.4 Energy descent (informal)
Each subproblem (OMP, smooth Jacobi, projection, fusion) is either a local minimization or a non-expansive projection, so the energy in Section 5.1 is non-increasing on its individually addressed components. Caveat: $\ell_0$ sparse coding is non-convex, so global descent is only guaranteed up to OMP approximation error.

### 8.5 Stability of confidence schedule
Because $c^t$ is bounded in $[0,1]$ and updated using bounded statistics (variance, distance, schedule), the weighting cannot cause unbounded amplification of any pixel. The weighted OMP problem is well-posed for all $t$.

### 8.6 Boundedness of iterates
If we clip both branches to $[0,1]$ before fusion and after projection, $u^t \in [0,1]^{H \times W}$ for all $t$. This is a benign constraint for natural images.

### 8.7 Manifold-correction safety
Because the denoiser is applied only inside the hole and then projected back via $\Pi_M$, observed data remains exact. For decreasing $\sigma_t \to 0$ and a contractive denoiser on the natural-image prior, the correction step becomes increasingly mild, preserving final-iterate stability.

---

## 9. Implementation Guide

### 9.1 Code structure (suggested)
- `crown/regime.py` — regime map estimator and boundary propagation
- `crown/confidence.py` — confidence map estimator
- `crown/weighted_omp.py` — extension of `masked_omp` with continuous weights
- `crown/nonlocal.py` — nearest-neighbor search and centralized refinement
- `crown/smooth.py` — biharmonic init and harmonic Jacobi relaxer
- `crown/manifold.py` — optional stochastic correction
- `crown/fuse.py` — fusion and projection
- `crown/run.py` — driver implementing the outer loop

This sits alongside your existing `masked_ksvd.py`, `inpainting_masked_ksvd.py`, and `phase5_iterative_ksvd.py` files. Reuse `masked_ksvd` for dictionary training.

### 9.2 Key functions and signatures (Python-style)

```python
def compute_regime_map(y: np.ndarray, M: np.ndarray,
                       window: int = 9, weights=(0.4, 0.3, 0.3),
                       sigma_smooth: float = 1.0) -> np.ndarray: ...

def compute_confidence_map(u: np.ndarray, M: np.ndarray, t: int,
                           c0: float, beta: float,
                           tau_v: float, rho: float) -> np.ndarray: ...

def weighted_omp(x: np.ndarray, D: np.ndarray,
                 W: np.ndarray, s: int) -> np.ndarray: ...

def nonlocal_refine(alpha: np.ndarray, patches: np.ndarray,
                    W_patches: np.ndarray, D: np.ndarray,
                    K_nl: int, h: float, lam: float) -> np.ndarray: ...

def harmonic_relax(u: np.ndarray, M: np.ndarray, K_s: int) -> np.ndarray: ...

def manifold_correct(u: np.ndarray, M: np.ndarray,
                     sigma_t: float, denoiser=...) -> np.ndarray: ...

def fuse_and_project(u_sparse, u_smooth, r, M, y) -> np.ndarray: ...
```

### 9.3 Critical implementation notes
1. **Weighted OMP**: scale $D$ columns by $\sqrt{W_p}$ per patch, normalize them inside OMP, and undo the scaling when reconstructing the patch.
2. **Boundary propagation of regime map**: easiest correct implementation is `scipy.ndimage.distance_transform_edt` plus Gaussian-weighted lookup, or a `inpaint_biharmonic` call on the regime map itself (treating $\Omega^c$ as missing).
3. **Overlap averaging**: precompute a normalization map equal to the count of patches covering each pixel.
4. **Patch extraction**: use sklearn's patch extractor for consistency with your existing code.
5. **Memory**: with $128 \times 128$ images, $8 \times 8$ patches, ~14400 patches, all intermediate tensors fit easily in memory. Avoid storing per-iteration full patch sets; recompute from $u$.
6. **Determinism**: seed all RNGs in `crown.run.run_crown_inpaint` and propagate the seed to manifold noise injection.

### 9.4 Reuse of existing code
- Reuse `masked_ksvd.masked_omp` as starting point for `weighted_omp` (add the diagonal-scaling preprocessing and OMP column normalization).
- Reuse `inpainting_multiscale_masked_ksvd.extract_and_translate_patches` and the dictionary training pipeline.
- Reuse `phase5_iterative_ksvd` for the iteration scaffolding; CROWN-Inpaint replaces the inner sparse-coding-only step with the full fused step.

---

## 10. Hyperparameters and Schedules

### 10.1 Defaults
- Patch size $p = 8$, atoms $K = 256$, sparsity $s = 8$.
- Outer iterations $T = 5$.
- Smooth inner iterations $K_s = 10$.
- Nonlocal neighbors $K_{nl} = 10$; cadence every 2 outer iterations.
- Confidence schedule: $c_0 = 0.05$, $\beta = 0.15$, $\tau_v$ adapted to median, $\rho = 4$.
- Regime weights $\omega_g = 0.4, \omega_a = 0.3, \omega_h = 0.3$, smoothing $\sigma_r = 1.0$.
- Manifold correction: $\sigma_0 = 0.04$, $\gamma = 0.7$, cadence every 2 outer iterations.

### 10.2 Stability advice
- If the regime map looks too binary (sharp seams), increase $\sigma_r$ or use a smaller window.
- If the confidence map saturates to 1 too early, decrease $\beta$ to slow trust growth.
- If the manifold step degrades quality, decrease $\sigma_0$ or disable; it is optional.

### 10.3 Sweep recommendations for the paper
Sweep one variable at a time and report PSNR/SSIM, hole-PSNR, and runtime:
- $\beta \in \{0.05, 0.10, 0.15, 0.25\}$
- $K_{nl} \in \{0, 5, 10, 20\}$ (0 = no nonlocal)
- $\sigma_0 \in \{0.0, 0.02, 0.04, 0.08\}$ (0 = manifold off)
- Smooth weights vs flat $r = 0.5$ baseline.

---

## 11. Experimental Protocol

### 11.1 Datasets
- **Places2** for natural-scene generalization.
- **Paris StreetView** for textured urban content.
- **CelebA-HQ** for face-centric tests.
- **DTD-style texture sets** for stress on textured holes.
- Your **controlled smooth-vs-textured hole protocol** for clean per-regime analysis.

### 11.2 Mask families
- Rectangular blocks (varying size).
- Free-form irregular brush strokes.
- Thin scratches.
- Mixed-mask stress tests.
- Extreme mask ratios (e.g. 30%-50%).

### 11.3 Baselines
- Biharmonic and Telea classical baselines.
- Your current iterative masked K-SVD baseline.
- A nonlocal sparse representation baseline (CSR/GSR-style where feasible).
- A modern deep baseline (LaMa).
- A diffusion baseline (RePaint or a fast inversion variant).

### 11.4 Metrics
- PSNR and SSIM.
- LPIPS and DISTS.
- Hole-only PSNR.
- Boundary-band MAE in a 4-pixel ring around the hole perimeter.
- Runtime and memory.

### 11.5 Ablations
- No confidence weighting (all hole pixels weight 1).
- No regime fusion (sparse only or smooth only).
- No nonlocal coupling.
- No manifold correction.
- Single dictionary vs adaptive dictionary.

### 11.6 Statistical protocol
- Multiple mask seeds per image (e.g. 5).
- Paired significance tests (Wilcoxon signed-rank) on hole-PSNR.
- Report median, mean, and 95% confidence interval.

---

## 12. Diagnostics and Failure Modes

### 12.1 Diagnostics to track per iteration
- PSNR / SSIM / hole-PSNR.
- Mean and std of $r^t$ inside the hole.
- Mean and std of $c^t$ inside the hole.
- Mean OMP residual.
- Number of nonzero entries per code (sparsity sanity check).
- Boundary-band MAE.

### 12.2 Likely failure modes and mitigations
1. **Texture leak across regime boundary**: caused by sharp $r^t$. Mitigation: Gaussian-smooth $r^t$.
2. **Self-reinforcing artifacts**: caused by confidence growing too fast. Mitigation: lower $\beta$.
3. **Over-smoothing in textured regime**: caused by misclassified texture as smooth. Mitigation: increase $\omega_a$ and $\omega_h$.
4. **Manifold step blurring details**: caused by overly aggressive denoiser. Mitigation: smaller $\sigma_0$ or stronger $\gamma$ decay.
5. **Slow runtime from nonlocal coupling**: mitigation: reduce $K_{nl}$, run nonlocal less often, or use ANN search.

---

## 13. Connection to Prior Work

| Component in CROWN-Inpaint | Closest prior idea | What is different here |
|---|---|---|
| Sparse coding over learned dictionary | K-SVD + masked OMP (Aharon-Elad-Bruckstein 2006; Elad-Aharon 2006) | Confidence-weighted OMP and integration with regime fusion |
| Smooth prior | Chan-Shen 2002 PDE inpainting | Used as one branch of a fusion, not as a standalone method |
| Nonlocal coefficient coupling | CSR (Dong et al. 2011), GSR (Zhang et al. 2014) | Combined with hard observation constraints inside an iterative inpainting pipeline |
| Iterative refinement with hard constraints | Mairal-Sapiro-Elad-style sparse restoration | Confidence schedule and regime-aware fusion are new |
| Stochastic correction | Manifold-constraint diffusion (Chung et al. 2022), Stochastic PnP (Park et al. 2026) | Plug-in-style without retraining; optional and lightweight |

The novelty is the unified design: a single iterative solver that is regime-aware, confidence-aware, nonlocal, and optionally noise-perturbed — for inpainting specifically.

---

## 14. Validation Checklist

Before claiming any reportable improvement:

- [ ] Hard constraint preserved every iteration (numerical check on $\|u^t \odot M - y \odot M\|$).
- [ ] $u^t \in [0,1]$ at all iterations.
- [ ] Sparsity respected per patch.
- [ ] Manifold correction toggle reproduces non-manifold case bit-exactly when off.
- [ ] Regime map equals near-binary on synthetic test images with one smooth side and one striped side.
- [ ] Confidence map is monotone-increasing on average across iterations.
- [ ] Runtime is within an acceptable constant factor of the baseline iterative K-SVD (target $\le 3\times$).
- [ ] PSNR / SSIM / hole-PSNR, LPIPS, and boundary MAE all logged to disk.
- [ ] Seeds fixed and reproducibility verified across two runs.
- [ ] Ablation matrix complete.
- [ ] Paired significance tests done on at least one dataset/mask pair.

---

## 15. Glossary

- **Hole**: set of pixels with $M_i = 0$.
- **Observation set $\Omega$**: pixels with $M_i = 1$.
- **Regime map $r$**: scalar in $[0,1]$ describing local texture richness.
- **Confidence map $c$**: scalar in $[0,1]$ describing reliability of current hole estimate.
- **Weighted OMP**: OMP applied to weighted residuals; reduces to standard masked OMP when weights are 0/1.
- **Hard constraint $\Pi_M$**: replaces observed pixels with $y$ exactly.
- **Manifold correction**: stochastic perturbation followed by denoising and projection.

---

## End of specification

This document is intended to be the single source of truth for CROWN-Inpaint. Any implementation, experiment, or paper writing should reference this file for math, defaults, and validation criteria.
