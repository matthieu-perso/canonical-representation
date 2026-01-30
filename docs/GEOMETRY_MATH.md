# Geometry of the parameter space: math walkthrough

This document walks through the **math** behind the geometry we compute: what each object is, why it matters, and how the numbers you see (Fisher eigenvalues, curvature proxy, sharpness, path length) relate to the "shape" of the space.

---

## 1. Parameter space and loss

- **Parameters:** \(\theta \in \mathbb{R}^d\) (all trainable weights of the model flattened; \(d \approx 300\text{k}\)).
- **Loss:** \(\mathcal{L}(\theta) = \mathbb{E}_{(x,y)}[\ell(f_\theta(x), y)]\) (expected loss over data). In practice we use a **sample** (e.g. 20 batches) to approximate this.
- **Training:** We move \(\theta\) along a trajectory \(\theta(t)\) (step \(t\)) by gradient descent. So we have a **path** in \(\mathbb{R}^d\).

Euclidean geometry in \(\mathbb{R}^d\) (plain distances and angles) does not reflect how hard it is to change the **predictions** when we move \(\theta\). Information geometry gives a better notion of "distance" and "curvature" for that.

---

## 2. Fisher information as a Riemannian metric

**Idea:** Treat the parameter space as a **Riemannian manifold**: at each \(\theta\) we define an inner product (a "metric") that measures how much the **model distribution** changes when we move \(\theta\). Small change in predictions \(\Rightarrow\) short distance; large change \(\Rightarrow\) long distance.

**Model as a conditional distribution:** For a classification task, \(f_\theta(x)\) gives logits; we get a probability distribution \(p_\theta(y|x)\) (e.g. softmax). So the model defines a family of distributions \(\{p_\theta\}\) indexed by \(\theta\).

**Fisher information matrix (FIM):**  
At a point \(\theta\), the FIM is the \(d\times d\) symmetric positive-semidefinite matrix
\[
g_{ij}(\theta) = \mathbb{E}_{x,y\sim p_\theta}\left[ \frac{\partial \log p_\theta(y|x)}{\partial \theta_i} \frac{\partial \log p_\theta(y|x)}{\partial \theta_j} \right].
\]
For the **empirical** version we average over **training data** (or a batch) instead of the model distribution:
\[
\widehat{g}_{ij}(\theta) = \frac{1}{n}\sum_{k=1}^n \frac{\partial \log p_\theta(y_k|x_k)}{\partial \theta_i} \frac{\partial \log p_\theta(y_k|x_k)}{\partial \theta_j}.
\]
In other words: \(\widehat{g} = \frac{1}{n}\sum_k \nabla_\theta \log p_\theta(y_k|x_k) \big(\nabla_\theta \log p_\theta(y_k|x_k)\big)^\top\). For cross-entropy loss, \(\nabla_\theta \log p_\theta(y|x)\) is the same as the gradient of the loss on that example (up to sign). So **empirical Fisher** = average of outer products of per-example gradients. That is what we compute in `fisher.py`.

**What \(g\) does:**  
- **Length of a small step \(v\):** \(\|v\|_g^2 = v^\top g(\theta) v\).  
- **Eigenvalues of \(g\):** If \(\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d\) are eigenvalues, then in direction of eigenvector \(v_i\), a unit step in parameter space has length \(\sqrt{\lambda_i}\). So **large \(\lambda\)** \(\Rightarrow\) that direction is "stiff" (small parameter change = big change in the metric); **small \(\lambda\)** \(\Rightarrow\) "flat" (you can move a lot in parameters without changing the model much in that direction).

We only compute the **top few eigenvalues** (and eigenvectors) of \(g\) by power iteration + deflation, without forming the full \(d\times d\) matrix.

---

## 3. Quantities we compute from the Fisher

**Fisher top eigenvalues \(\lambda_1,\ldots,\lambda_k\):**  
- \(\lambda_1\) = **fisher_max_eig**: stiffness in the "stiffest" direction.  
- **fisher_cond** = \(\lambda_{\max}/\lambda_{\min}\) (over the top \(k\) we have): if large, the metric is very **anisotropic** (some directions much stiffer than others).

**curvature_proxy:**  
We define
\[
R_{\mathrm{proxy}} = \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}} \in [0,1].
\]
This is **not** the true Riemannian scalar curvature \(R(\theta)\) (which would need derivatives of \(g\)). It is a **scalar summary of anisotropy**: \(R_{\mathrm{proxy}}=0\) means all considered eigenvalues equal (isotropic); \(R_{\mathrm{proxy}}\) near 1 means one direction dominates. We use it as a simple "curvature-like" indicator that can be plotted vs step.

---

## 4. Sharpness (Hessian trace)

**Hessian:** \(H_{ij}(\theta) = \frac{\partial^2 \mathcal{L}}{\partial \theta_i \partial \theta_j}\). It measures **curvature of the loss** (second derivatives).

**Trace:** \(\mathrm{tr}(H) = \sum_i H_{ii} = \sum_i \lambda_i^{(H)}\) (sum of all eigenvalues of \(H\)).  
- **Large trace** \(\Rightarrow\) loss is "sharp" in many directions (steep curvature).  
- **Small trace** \(\Rightarrow\) loss is "flat" (flat minimum).

We **never form \(H\)** (it would be \(d\times d\)). We estimate \(\mathrm{tr}(H)\) by **Hutchinson's method**: \(\mathrm{tr}(H) = \mathbb{E}_v[v^\top H v]\) where \(v\) is random (e.g. Rademacher). So we take random vectors \(v\), compute \(Hv\) via one Hessian-vector product (autodiff), and average \(v^\top (Hv)\) to get the trace. That is `sharpness.py`.

So **sharpness_hessian_trace** = estimated \(\mathrm{tr}(H)\) at the current \(\theta\). It describes the **loss landscape** (how curved the loss is), not the Fisher metric; but sharp minima often correlate with generalization, and we track it alongside the Fisher.

---

## 5. Path length (natural distance along training)

**Euclidean distance** between two checkpoints \(\theta_1\) and \(\theta_2\) is \(\|\theta_2 - \theta_1\|\). It ignores how much the **model** changed.

**Natural (Fisher–Rao) distance** along a path \(\theta(t)\) is
\[
\int \sqrt{ \dot\theta(t)^\top g(\theta(t)) \dot\theta(t) } \, dt.
\]
So we weight each infinitesimal step by the metric at that point. We **don't have the full path** \(\theta(t)\), only checkpoints at steps \(t_1, t_2, \ldots\). So we approximate:

**Segment from \(\theta_{\mathrm{prev}}\) to \(\theta_{\mathrm{curr}}\):**  
Set \(\Delta\theta = \theta_{\mathrm{curr}} - \theta_{\mathrm{prev}}\). We approximate the length of this chord in the metric at \(\theta_{\mathrm{curr}}\) (or at one of the endpoints):
\[
\mathrm{length}^2 \approx \Delta\theta^\top g(\theta_{\mathrm{curr}}) \Delta\theta.
\]
We don't have full \(g\); we have top-\(k\) eigenpairs \((\lambda_i, v_i)\). So we use the low-rank approximation \(g \approx \sum_i \lambda_i v_i v_i^\top\):
\[
\mathrm{length}^2 \approx \sum_{i=1}^k \lambda_i \, (v_i^\top \Delta\theta)^2.
\]
That is **segment_natural_length**. We sum over consecutive checkpoint pairs to get **cumulative_natural_path_length**. So you see "how far" the training has traveled in **information distance** (distance in terms of the Fisher metric), not in raw parameter space.

---

## 6. Representation space (hidden states)

So far everything was in **parameter space** \(\theta\). The **representation space** is different: at each input \(x\), the model produces a hidden state \(h_\theta(x) \in \mathbb{R}^D\) (e.g. last layer, last token, \(D=128\)). So we get a **cloud of points** \(\{h_\theta(x_i)\}\) in \(\mathbb{R}^D\).

**Covariance of hidden states:**  
Over a batch (or many batches), compute \(\Sigma = \frac{1}{n}\sum_i (h_i - \bar{h})(h_i - \bar{h})^\top\). The **eigenvalues** of \(\Sigma\) tell you how variance is distributed across directions: a few large eigenvalues \(\Rightarrow\) points lie roughly in a low-dimensional subspace (low "effective dimension"); many similar eigenvalues \(\Rightarrow\) more spread out.

We compute the **top eigenvalues of \(\Sigma\)** (or of \(H^\top H/n\) where \(H\) is the matrix of centered hidden states) and log them as **representation_spectrum**. So you track how the **shape** of the representation cloud (effective dimension, anisotropy) changes with training. That is `representation.py`.

---

## 7. How this fits together (summary table)

| Object | Math | What we compute | Interpretation |
|--------|------|------------------|----------------|
| **Metric \(g(\theta)\)** | FIM = \(\mathbb{E}[\nabla\log p \, (\nabla\log p)^\top]\) | Top-\(k\) eigenvalues (and eigenvectors for path length) | Local "stiffness" of the model in parameter space |
| **curvature_proxy** | \((\lambda_{\max}-\lambda_{\min})/(\lambda_{\max}+\lambda_{\min})\) | Scalar from those eigenvalues | Anisotropy of the metric (not true curvature \(R\)) |
| **Sharpness** | \(\mathrm{tr}(H)\) | Hutchinson estimate of Hessian trace | Curvature of the **loss** (sharp vs flat minimum) |
| **Path length** | \(\int \sqrt{\dot\theta^\top g \dot\theta}\,dt\) | Sum of \(\sqrt{\Delta\theta^\top g \Delta\theta}\) over segments | Information-geometric distance traveled along training |
| **Representation spectrum** | Eigenvalues of cov of \(h_\theta(x)\) | Top eigenvalues of hidden-state covariance | Shape / effective dimension of the representation cloud |

---

## 8. What "geometry of the space" means here

- **Parameter space:** We give it a **Riemannian metric** \(g(\theta)\) (the Fisher). Then "geometry" means: lengths of vectors (eigenvalues of \(g\)), anisotropy (curvature_proxy), and distances along the training path (path length). We do **not** compute the full curvature tensor or scalar curvature \(R(\theta)\); we only have scalar summaries (curvature_proxy, Fisher cond, sharpness) and path length.
- **Representation space:** We don't put a metric on it; we only look at the **covariance spectrum** of the hidden states. So "geometry" there means the **shape** of the cloud (effective dimension, dominant directions).

So: **Fisher** = metric in parameter space; **curvature_proxy** = anisotropy of that metric; **sharpness** = curvature of the loss; **path length** = distance along the training path in that metric; **representation_spectrum** = shape of the representation cloud. Together they let you ask whether the **information-geometric** picture (and the representation) changes around the grokking transition.
