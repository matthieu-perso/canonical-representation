# Reconstructing the geometry of the information manifold

This document explains **what is computed**, **how to use it**, and **how to detect geometric changes** in representation and parameter space (e.g. at the grokking transition).

---

## 1. What is now computed

### Parameter space (Fisher–Rao metric)

| Quantity | Meaning | Where |
|----------|---------|--------|
| **Fisher top-k eigenvalues** | Principal “stiffnesses” of the Fisher metric at θ | In-training + offline CSV |
| **fisher_max_eig, fisher_cond** | Largest eigenvalue; condition number (anisotropy) | In-training + offline CSV |
| **curvature_proxy** | Scalar \(R_{\mathrm{proxy}} = (\lambda_{\max} - \lambda_{\min})/(\lambda_{\max} + \lambda_{\min})\); local anisotropy | In-training + offline CSV |
| **sharpness_hessian_trace** | Hessian trace at θ (loss-landscape sharpness) | In-training + offline CSV |
| **segment_natural_length** | Fisher–Rao length of the chord between two consecutive checkpoints | Offline CSV only |
| **cumulative_natural_path_length** | Sum of segment lengths from step 0 | Offline CSV only |

- **Path length** uses a low-rank Fisher (top-10 eigenpairs) at the *current* checkpoint:  
  \(\mathrm{length}^2 \approx \sum_i \lambda_i (v_i \cdot \Delta\theta)^2\), \(\Delta\theta = \theta_{\mathrm{curr}} - \theta_{\mathrm{prev}}\).  
  So you get an **information-geometric distance** along the training path (no full \(d\times d\) matrix).

### Representation space (hidden states)

| Quantity | Meaning | Where |
|----------|---------|--------|
| **representation_spectrum** | Top eigenvalues of the covariance of last-layer hidden states (last token) | In-training + offline CSV |

- Describes the **shape of the representation cloud** (effective dimension, anisotropy).  
- Changes in this spectrum over training indicate geometric changes in **representation space**.

---

## 2. What you still do *not* have (and would need extra work)

- **Full metric \(g(\theta)\)**: Only top-k eigenvalues (and, internally, eigenvectors for path length) are used; no full \(d\times d\) FIM.
- **True scalar curvature \(R(\theta)\)**: That would need derivatives of the metric; we only have the scalar **curvature_proxy** from the eigenvalue spectrum.
- **Curvature along the path**: We have **curvature_proxy(step)** (a curve of a scalar proxy), not \(R(\theta(t))\) from differential geometry.
- **Geodesic distance** between non-consecutive checkpoints: Only **consecutive** segments are summed for cumulative path length; no geodesic solver.

So: you have a **low-rank, scalar-summary view** of the Fisher metric, a **path length** along the training trajectory, and the **representation spectrum**. That is enough to detect **geometric changes** (see below).

---

## 3. How to get the data

### In training (every `compute_geometry_every` steps)

- Logged to **wandb**: `geometry/fisher_max_eig`, `geometry/fisher_cond`, `geometry/curvature_proxy`, `geometry/sharpness_hessian_trace`, `geometry/fisher_eig_0`, …, `geometry/representation_eig_0`, …
- Logged to **console**: full geometry dict (including lists).

### Offline (after a run)

```bash
uv run python grokking/scripts/run_geometry_analysis.py RUN_DIR -o RUN_DIR/geometry_metrics.csv --step-interval 500
```

- Produces **geometry_metrics.csv** with: step, Fisher eigenvalues, curvature_proxy, sharpness, representation_spectrum, **segment_natural_length**, **cumulative_natural_path_length**.
- Path length is computed between **consecutive** checkpoints (e.g. 500→1000, 1000→1500). First row has no segment; cumulative starts at 0 and increases.

---

## 4. How to detect geometric changes (e.g. at grokking)

1. **Align with the transition**  
   - From your eval logs, get **train** and **val accuracy** (or loss) vs step.  
   - Mark the step \(t^*\) where **val accuracy** crosses your grokking threshold (e.g. 0.9).

2. **Plot geometry vs step**  
   - **Parameter space:**  
     - `curvature_proxy`, `fisher_max_eig`, `fisher_cond`, `sharpness_hessian_trace` vs step.  
     - `cumulative_natural_path_length` vs step (from offline CSV).  
   - **Representation space:**  
     - `representation_spectrum` (e.g. top 3–5 eigenvalues) vs step.  
   - Overlay vertical line at \(t^*\).

3. **What to look for**  
   - **Slope or level change** in curvature_proxy, Fisher eigenvalues, or sharpness near \(t^*\).  
   - **Inflection** or **acceleration** in cumulative natural path length (e.g. path “speeds up” or “slows down” in information distance).  
   - **Change in representation spectrum** (e.g. effective dimension or anisotropy) around \(t^*\).

4. **Quantitative**  
   - Correlate curvature_proxy (or Fisher max eig, or sharpness) with val accuracy over time.  
   - Or: fit a simple change-point model for curvature_proxy vs step and compare the estimated change point to \(t^*\).

---

## 5. Summary

- **Parameter space:** You get Fisher eigenvalues, a scalar curvature proxy, sharpness, and **natural path length** between consecutive checkpoints (and its cumulative sum).  
- **Representation space:** You get the **covariance spectrum** of last-layer hidden states.  
- You do **not** get the full Riemannian curvature tensor or full metric; you get **enough to detect geometric and representation-space changes** around phase transitions (e.g. grokking) by plotting and correlating these quantities with validation accuracy.
