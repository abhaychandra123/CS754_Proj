# Internet Literature Review and Novel Method Proposal for Image Inpainting

Date: 2026-04-25
Prepared for: CS754 project workflow

## 1. Request and Scope

You requested a wide internet search on image inpainting, then asked for:

- broad paper coverage
- gap identification
- relevant idea borrowing
- one concrete novel improvement with paper potential

This report consolidates the full synthesis from that search process.

## 2. Search Strategy and Coverage

### 2.1 Search surfaces used

- arXiv query pages for broad discovery
- arXiv paper pages for method-level abstracts and claims
- Crossref metadata endpoints for DOI-grounded title and venue confirmation
- OpenAlex metadata endpoints for citation context and abstract snippets
- Curated Github list for diffusion-editing benchmarks and method families

### 2.2 Query families executed

- image inpainting broad query
- sparse coding inpainting query
- diffusion image inpainting query
- image inpainting survey query
- K-SVD inpainting query

### 2.3 Access limitations encountered

Some publisher pages blocked direct extraction in this environment:

- IEEE pages occasionally returned scraping errors
- ACM pages returned 403 in some attempts
- Github repository search returned 502 in one attempt

Mitigation:

- switched to Crossref API metadata for DOI records
- switched to OpenAlex API for title/year/topic confirmation and abstract snippets

## 3. Consolidated Findings by Method Era

## 3.1 Foundational inpainting priors (PDE and exemplar)

| Paper | Year | Core idea | Why it matters here |
|---|---:|---|---|
| Image inpainting (Bertalmio et al., SIGGRAPH) https://doi.org/10.1145/344779.344972 | 2000 | propagates structure/isophotes from boundary inward | baseline principle behind structure continuity |
| Mathematical Models for Local Nontexture Inpaintings (Shen and Chan, SIAM) https://doi.org/10.1137/S0036139900368844 | 2002 | variational and PDE inpainting; harmonic and biharmonic behavior in smooth regions | explains why smooth-hole interpolation can be near-optimal |
| Object removal by exemplar-based inpainting (Criminisi et al., CVPR) https://doi.org/10.1109/CVPR.2003.1211538 | 2003 | patch copying from known regions | precursor to modern patch-based texture transport |

Key observation:

- smooth areas often favor PDE priors
- textured regions require patch/structure transfer mechanisms

## 3.2 Sparse coding and dictionary-learning lineage

| Paper | Year | Core idea | Relevance to your pipeline |
|---|---:|---|---|
| K-SVD (Aharon, Elad, Bruckstein) https://doi.org/10.1109/TSP.2006.881199 | 2006 | learns overcomplete dictionary with alternating sparse coding and atom updates | backbone of masked K-SVD style inpainting |
| Image Denoising via Sparse and Redundant Representations Over Learned Dictionaries (Elad, Aharon) https://doi.org/10.1109/TIP.2006.881969 | 2006 | sparse patch coding over learned dictionary for restoration | canonical sparse reconstruction prior |
| Image Denoising Via Learned Dictionaries and Sparse representation (CVPR) https://doi.org/10.1109/CVPR.2006.142 | 2006 | practical sparse dictionary denoising variant | early proof of restoration quality from learned dictionaries |
| Sparse Representation for Color Image Restoration (Mairal, Elad, Sapiro) https://doi.org/10.1109/TIP.2007.911828 | 2008 | sparse coding for color restoration under corruption | important bridge to inpainting/restoration tasks |
| Centralized sparse representation for image restoration (Dong et al.) https://doi.org/10.1109/ICCV.2011.6126377 | 2011 | combines sparse coding with nonlocal coefficient consistency | key signal that grouped/nonlocal coupling improves quality |
| Group-Based Sparse Representation for Image Restoration (Zhang et al.) https://doi.org/10.1109/TIP.2014.2323127 | 2014 | group sparse representation over patch groups | strong evidence for nonlocal self-similarity gains |
| Image Deblurring and Super-Resolution by Adaptive Sparse Domain Selection and Adaptive Regularization (Dong et al.) https://doi.org/10.1109/TIP.2011.2108306 | 2011 | adaptive sparse domain plus adaptive regularization | relevant for region-adaptive prior selection |

Key observation:

- sparse coding is strongest when coupled with nonlocal/group regularization
- independent patch coding tends to miss long-range consistency

## 3.3 Deep CNN/GAN inpainting progression

| Paper | Year | Core idea | Claimed issue addressed |
|---|---:|---|---|
| Generative Image Inpainting with Contextual Attention https://arxiv.org/abs/1801.07892 | 2018 | explicit feature borrowing from distant regions | fixes blurry/distorted results from plain convs |
| Partial Convolutions https://arxiv.org/abs/1804.07723 | 2018 | masked and renormalized convolution over valid pixels | reduces artifacts from invalid-hole convolutions |
| Free-Form Image Inpainting with Gated Convolution https://arxiv.org/abs/1806.03589 | 2019 | learnable gating per location/channel | better free-form mask handling and flexibility |
| EdgeConnect https://arxiv.org/abs/1901.00212 | 2019 | edge hallucination then image completion | stronger structural fidelity |
| LaMa (Fourier convolutions) https://arxiv.org/abs/2109.07161 | 2021 | image-wide receptive fields with FFC, large-mask training | better large hole and periodic texture completion |

Key observation:

- stronger receptive field and structural guidance are recurring winners
- methods still trade off seam consistency and hole realism depending on mask and texture regime

## 3.4 Diffusion and inverse-problem methods

| Paper | Year | Core idea | Claimed limitation addressed |
|---|---:|---|---|
| Palette https://arxiv.org/abs/2111.05826 | 2021/2022 | unified diffusion image-to-image framework | reduces task-specific tuning burden |
| RePaint https://arxiv.org/abs/2201.09865 | 2022 | uses pretrained unconditional DDPM with mask-conditioned reverse updates | mask-generalization and semantic quality vs GANs |
| Improving Diffusion Models for Inverse Problems using Manifold Constraints https://arxiv.org/abs/2206.00941 | 2022 | manifold correction term in iterative diffusion inverse solving | off-manifold drift and accumulated error |
| Structure Matters (StrDiffusion) https://arxiv.org/abs/2403.19898 | 2024 | structure-guided denoising and adaptive resampling | semantic discrepancy between masked and unmasked regions |
| InverFill https://arxiv.org/abs/2603.23463 | 2026 | one-step inversion to improve few-step diffusion inpainting | harmonization artifacts from random noise init |
| Stochastic Generative Plug-and-Play Priors https://arxiv.org/abs/2604.03603 | 2026 | stochastic PnP with score priors and smoothed objective interpretation | poor robustness in severely ill-posed settings |

Key observation:

- manifold consistency and initialization quality are central to modern diffusion inpainting quality
- optimization-aware sampling modifications are now a major trend

## 3.5 Surveys and benchmark trends

| Survey | Year | Scope | Open directions highlighted |
|---|---:|---|---|
| Deep Learning-based Image and Video Inpainting: A Survey https://arxiv.org/abs/2401.03395 | 2024 | CNN, VAE, GAN, diffusion, losses, datasets, metrics | open challenges in realism, consistency, and evaluation |
| Diffusion Models for Image Restoration and Enhancement: A Comprehensive Survey https://arxiv.org/abs/2308.09388 | 2023/2025 | diffusion IR pipelines across restoration tasks | sampling efficiency, compression, distortion modeling, robust framework design |
| Diffusion Model-Based Image Editing: A Survey https://arxiv.org/abs/2402.17525 | 2024/2025 | broad editing taxonomy, includes inpainting/outpainting | benchmark standardization and robust evaluation needed |

Benchmark ecosystem reference:

- Awesome diffusion editing list and EditEval tracking: https://github.com/SiatMMLab/Awesome-Diffusion-Model-Based-Image-Editing-Methods

## 4. High-Confidence Gap Analysis for Your Problem Statement

Your project is iterative masked K-SVD inpainting with explicit smooth-vs-textured regime behavior. Based on the above literature, the strongest unfilled gaps are:

1. Regime mismatch is under-modeled
- most methods use one dominant prior family globally
- smooth and textured hole regions require different priors inside one solver

2. Confidence-unaware iterative sparse reconstruction
- hole estimates are often treated as equally reliable early on
- this can trap iterative methods in self-reinforcing artifacts

3. Weak nonlocal coupling in patch-sparse inpainting implementations
- nonlocal/group sparse coupling is known to help but is often omitted in practical iterative inpainting code

4. Missing manifold correction in classical sparse loops
- modern inverse-problem diffusion work shows off-manifold drift is harmful
- sparse-only loops typically lack explicit manifold re-projection

5. Evaluation objective mismatch
- full-image PSNR/SSIM can hide boundary and hole-interior failures
- hole-specific and boundary-band metrics are not always primary

## 5. Proposed Novel Method (Publishable Candidate)

Method name:

CROWN-Inpaint
Confidence-weighted Regime-aware Overlap Nonlocal inpainting

Core principle:

Combine three components in one iterative framework:

- smooth-prior branch for low-texture zones
- nonlocal grouped sparse branch for textured zones
- optional stochastic manifold correction step for stability and realism

## 5.1 Mathematical form

Let:

- y be observed image
- M be binary mask (1 observed, 0 missing)
- D be learned dictionary
- u^t be current reconstruction at iteration t
- r^t in [0,1] be texture regime map (0 smooth, 1 textured)
- c^t in [0,1] be confidence map on missing pixels

Confidence-weighted sparse coding per patch p:

$$
\min_{\alpha_p}\; \|W_p^t \odot (x_p^t - D\alpha_p)\|_2^2 \quad \text{s.t.}\; \|\alpha_p\|_0 \le s
$$

with

$$
W_p^t = M_p + (1-M_p)\odot c_p^t
$$

Nonlocal coefficient coupling:

$$
\lambda \sum_{q\in\mathcal{N}(p)} \|\alpha_p - \alpha_q\|_1
$$

Regime-aware fusion with hard observation constraint:

$$
u^{t+1} = M\odot y + (1-M)\odot \left[(1-r^t)u_{smooth}^t + r^t u_{sparse}^t\right]
$$

Optional stochastic manifold correction every k steps:

- inject small controlled noise inside hole only
- perform one correction denoising/prox step with data consistency projection

## 5.2 Algorithm sketch

1. Initialize u^0 with biharmonic fill.
2. For each iteration t:
   - compute regime map r^t from local gradient energy, structure coherence, and spectral entropy
   - compute confidence map c^t from overlap disagreement and patch reconstruction residual
   - run confidence-weighted sparse coding with nonlocal patch groups
   - reconstruct u_sparse^t by overlap averaging
   - compute u_smooth^t with smooth prior update
   - fuse by regime map and enforce hard constraint on observed pixels
   - optional manifold correction step
3. Output final u^T and confidence diagnostics.

## 5.3 Why this is novel enough to investigate for publication

1. Explicit regime-adaptive prior fusion inside one iterative sparse framework.
2. Confidence scheduling for missing-pixel contributions in sparse coding.
3. Integration of nonlocal group consistency with hard-constraint iterative inpainting.
4. Optional diffusion-inspired manifold correction without requiring full retraining.

Novelty is in the unified optimization design, not in any single borrowed module.

## 6. Paper-Ready Experimental Design

## 6.1 Datasets

- Places2
- Paris StreetView
- CelebA-HQ
- texture-focused sets such as DTD-like subsets
- your controlled smooth-hole and textured-hole protocol

## 6.2 Mask families

- rectangle blocks
- irregular free-form brush masks
- thin scratches
- mixed-mask stress protocol
- extreme mask ratio protocol

## 6.3 Baselines

- biharmonic and Telea classical baselines
- your current iterative masked K-SVD baseline
- representative nonlocal sparse baselines (CSR/GSR style where implementable)
- one fast modern deep baseline (for example LaMa)
- one diffusion inpainting baseline (for example RePaint or few-step inversion variant)

## 6.4 Metrics

- PSNR and SSIM
- LPIPS and DISTS
- hole-only PSNR
- boundary-band MAE around hole perimeter
- runtime and memory

## 6.5 Required ablations

- remove confidence weighting
- remove regime map
- remove nonlocal coupling
- remove manifold correction
- compare single dictionary versus dual-branch/adaptive behavior

## 6.6 Statistical protocol

- paired significance testing across mask seeds
- confidence intervals for key metrics
- report robust central tendency (median and mean)

## 7. Practical Implementation Plan for Current Codebase

Target file focus in your repo:

- phase5_iterative_ksvd.py
- masked_ksvd.py
- inpainting_multiscale_masked_ksvd.py

Suggested implementation steps:

1. Add regime-map estimator function.
2. Add confidence-map estimator from overlap variance and residuals.
3. Extend masked OMP to support confidence weights.
4. Add nonlocal patch grouping and coefficient-coupling penalty.
5. Add smooth branch update and regime fusion step.
6. Add optional correction hook for stochastic manifold step.
7. Add evaluation utilities for hole-only and boundary metrics.

## 8. Risks and Mitigations

1. Risk: high runtime from nonlocal grouping and repeated sparse solves.
- Mitigation: approximate nearest neighbor search and patch subset scheduling.

2. Risk: confidence map instability early in iterations.
- Mitigation: monotone confidence schedule and clipping bounds.

3. Risk: overfitting method behavior to one mask style.
- Mitigation: multi-family mask training/evaluation protocol.

4. Risk: novelty overlap with prior adaptive sparse models.
- Mitigation: emphasize confidence-weighted regime fusion plus manifold correction and prove each block with ablations.

## 9. Publication Feasibility Assessment

Conservative assessment:

- promising and plausible if empirical gains are consistent and ablations are clean
- not guaranteed without strong cross-dataset evidence

Minimum evidence needed:

- consistent gains on textured-hole and mixed-hole settings
- no regressions in smooth-hole settings relative to biharmonic baseline
- strong seam quality metrics and visual examples
- meaningful runtime versus quality tradeoff

## 10. Source Index (Search Trace)

## 10.1 Broad query and survey pages

- https://arxiv.org/search/?query=image+inpainting&searchtype=all&abstracts=show&order=-announced_date_first&size=50
- https://arxiv.org/search/?query=sparse+coding+inpainting&searchtype=all&abstracts=show&order=-announced_date_first&size=50
- https://arxiv.org/search/?query=image+inpainting+survey&searchtype=all&abstracts=show&order=-announced_date_first&size=25
- https://arxiv.org/search/?query=K-SVD+inpainting&searchtype=all&abstracts=show&order=-announced_date_first&size=25
- https://arxiv.org/search/?query=diffusion+image+inpainting&searchtype=all&abstracts=show&order=-announced_date_first&size=25

## 10.2 Canonical inpainting and sparse papers

- https://doi.org/10.1145/344779.344972
- https://doi.org/10.1137/S0036139900368844
- https://doi.org/10.1109/CVPR.2003.1211538
- https://doi.org/10.1109/TSP.2006.881199
- https://doi.org/10.1109/TIP.2006.881969
- https://doi.org/10.1109/CVPR.2006.142
- https://doi.org/10.1109/TIP.2007.911828
- https://doi.org/10.1109/ICCV.2011.6126377
- https://doi.org/10.1109/TIP.2014.2323127

## 10.3 Deep and diffusion papers used in synthesis

- https://arxiv.org/abs/1801.07892
- https://arxiv.org/abs/1804.07723
- https://arxiv.org/abs/1806.03589
- https://arxiv.org/abs/1901.00212
- https://arxiv.org/abs/2109.07161
- https://arxiv.org/abs/2111.05826
- https://arxiv.org/abs/2201.09865
- https://arxiv.org/abs/2206.00941
- https://arxiv.org/abs/2403.19898
- https://arxiv.org/abs/2603.23463
- https://arxiv.org/abs/2604.03603

## 10.4 Survey and benchmark references

- https://arxiv.org/abs/2401.03395
- https://arxiv.org/abs/2308.09388
- https://arxiv.org/abs/2402.17525
- https://github.com/SiatMMLab/Awesome-Diffusion-Model-Based-Image-Editing-Methods

## 10.5 Metadata endpoints used for validation

- https://api.crossref.org/works/10.1109/TIP.2006.881969
- https://api.crossref.org/works/10.1137/S0036139900368844
- https://api.crossref.org/works/10.1145/344779.344972
- https://api.crossref.org/works/10.1109/TSP.2006.881199
- https://api.crossref.org/works/10.1109/TIP.2007.911828
- https://api.crossref.org/works/10.1109/CVPR.2006.142
- https://api.crossref.org/works/10.1109/TIP.2014.2323127
- https://api.crossref.org/works/10.1109/TIP.2011.2108306
- https://api.openalex.org/works/https://doi.org/10.1145/344779.344972
- https://api.openalex.org/works/https://doi.org/10.1109/CVPR.2003.1211538
- https://api.openalex.org/works/https://doi.org/10.1109/TSP.2006.881199
- https://api.openalex.org/works/https://doi.org/10.1109/ICCV.2011.6126377

## 11. Final Summary

The strongest practical path for your project is not to replace sparse inpainting, but to upgrade it into a regime-aware, confidence-weighted, nonlocal-coupled iterative framework, with optional stochastic manifold correction.

This directly aligns with:

- classical PDE strength on smooth regions
- sparse/nonlocal strength on textured regions
- modern inverse-problem insight about manifold drift

It is technically implementable in your current codebase and has a credible paper narrative if validated with strong ablations and robust mask protocols.
