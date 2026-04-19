"""Alignment of the last linear map with the canonical Fourier subspace for Z_p (modular arithmetic).

The readout weight W has shape (p, d): each column lives in R^p and can be compared to the
2-dimensional irrep subspace for frequency k (cos/sin modes on class indices 0..p-1).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    pass


def fourier_mode_subspace(
    p: int,
    k: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Orthonormal basis Q in R^{p x 2} for the k-th non-trivial Fourier mode.

    Rows are indexed by class label y in {0, ..., p-1}.
    """
    if p < 3:
        msg = f"p must be >= 3, got {p}"
        raise ValueError(msg)
    if k < 1 or k > (p - 1) // 2:
        msg = f"k must be in [1, (p-1)/2], got k={k} for p={p}"
        raise ValueError(msg)

    y = torch.arange(p, device=device, dtype=dtype)
    theta = 2.0 * math.pi * k * y / p
    scale = math.sqrt(2.0 / p)
    q0 = scale * torch.cos(theta)
    q1 = scale * torch.sin(theta)
    return torch.stack([q0, q1], dim=1)


def alignment_from_output_weight(
    weight: torch.Tensor,
    p: int,
    k: int = 1,
) -> dict[str, float]:
    """Compute scalar alignment metrics from output layer weight W of shape (p, d).

    Uses orthogonal projection onto the span of the k-th Fourier mode on class indices.
    """
    if weight.shape[0] != p:
        msg = f"weight must have shape (p, d) with p={p}, got {tuple(weight.shape)}"
        raise ValueError(msg)

    device = weight.device
    dtype = weight.dtype
    q = fourier_mode_subspace(p, k, device=device, dtype=dtype)
    # P = Q Q^T; orthogonal complement (I-P) W
    qt_w = q.T @ weight  # (2, d)
    proj = q @ qt_w  # (p, d)
    orth = weight - proj

    fro_w = torch.linalg.matrix_norm(weight, ord="fro")
    fro_orth = torch.linalg.matrix_norm(orth, ord="fro")
    fro_in = torch.linalg.matrix_norm(proj, ord="fro")

    eps = torch.finfo(dtype).eps * 128
    ratio_in = (fro_in / (fro_w + eps)).item()
    delta = (fro_orth / (fro_w + eps)).item()

    # Per-column angle between column and its projection (subspace angle proxy)
    norm_w = torch.linalg.vector_norm(weight, dim=0).clamp_min(eps)
    norm_proj = torch.linalg.vector_norm(proj, dim=0).clamp_min(eps)
    cos_align = (norm_proj / norm_w).clamp(0.0, 1.0)
    angles = torch.acos(cos_align)
    implemented_mean = angles.mean().item()
    implemented_max = angles.max().item()

    # Magnitude: relative gap between full norm and in-subspace energy
    magnitude_error = (1.0 - ratio_in) if ratio_in <= 1.0 else 0.0

    target_angle = 2.0 * math.pi * k / p

    return {
        "ideal_space_delta_to_ideal": float(delta),
        "ideal_distance_magnitude_error": float(magnitude_error),
        "ideal_2d_target_angle_rad": float(target_angle),
        "ideal_implemented_angle_rad_mean": float(implemented_mean),
        "ideal_implemented_angle_rad_max": float(implemented_max),
        "ideal_in_subspace_energy_ratio": float(ratio_in),
    }


# ---------------------------------------------------------------------------
# Extended geometry functions for canonical-geometry experiments
# ---------------------------------------------------------------------------


def per_irrep_energy_spectrum(weight: torch.Tensor, p: int) -> dict[str, float]:
    """Decompose W (shape p×d) into per-Fourier-mode energy fractions for Z_p.

    For each Fourier mode k ∈ {0, 1, …, (p-1)//2} computes
        alpha_k = ‖Π_{V_k} W‖²_F / ‖W‖²_F
    where k=0 is the DC (trivial) mode and k≥1 are the non-trivial irreps of Z_p.
    Fractions sum to 1 by construction.

    Returns a dict with:
      alpha_0 … alpha_{(p-1)//2}  — per-mode energy fractions
      dominant_k                  — argmax alpha_k for k ≥ 1
      top1_energy                 — max_{k≥1} alpha_k
      top3_energy                 — sum of top-3 alpha_k for k ≥ 1
      epsilon                     — 1 − top1_energy  (deviation from dominant mode)
      epsilon_dc                  — alpha_0  (trivial/DC energy fraction)
      spectrum_entropy            — Shannon entropy −∑ alpha_k log alpha_k
    """
    device = weight.device
    dtype = weight.dtype
    eps = torch.finfo(dtype).eps * 128

    fro_sq_val = torch.linalg.matrix_norm(weight, ord="fro").pow(2).item()

    result: dict[str, float] = {}

    # k = 0: DC / trivial mode, v0 = (1/√p) · 1_p
    v0 = torch.ones(p, device=device, dtype=dtype) / math.sqrt(p)
    alpha_0 = (v0 @ weight).pow(2).sum().item() / (fro_sq_val + eps)
    result["alpha_0"] = alpha_0

    nontrivial: list[tuple[int, float]] = []
    for k in range(1, (p - 1) // 2 + 1):
        q = fourier_mode_subspace(p, k, device=device, dtype=dtype)  # (p, 2)
        alpha_k = (q.T @ weight).pow(2).sum().item() / (fro_sq_val + eps)
        result[f"alpha_{k}"] = alpha_k
        nontrivial.append((k, alpha_k))

    sorted_nt = sorted(nontrivial, key=lambda t: t[1], reverse=True)
    dominant_k = sorted_nt[0][0] if sorted_nt else 1
    top1 = sorted_nt[0][1] if sorted_nt else 0.0
    top3 = sum(a for _, a in sorted_nt[:3])

    result["dominant_k"] = float(dominant_k)
    result["top1_energy"] = top1
    result["top3_energy"] = top3
    result["epsilon"] = 1.0 - top1
    result["epsilon_dc"] = alpha_0

    all_alphas = [alpha_0] + [a for _, a in nontrivial]
    entropy = -sum(a * math.log(a + 1e-12) for a in all_alphas)
    result["spectrum_entropy"] = entropy

    return result


def reference_k_theory(
    p: int,
    *,
    alpha: float = 1.0,
    sigma: float = 0.0,
    epsilon_target: float = 0.01,
) -> tuple[int, ...]:
    """Greedy irrep set K from the paper's Eq. 8 construction (compute_K_min).

    With fixed (alpha, sigma, epsilon_target) this is a function of p only.
    Default alpha=1, sigma=0 matches the symmetric analytic margin model used in
    irrep_ce_analysis / compute_K_min.
    """
    result = compute_K_min(p=p, alpha=alpha, sigma=sigma, epsilon_target=epsilon_target)
    return tuple(int(k) for k in result["active_ks"])


def spectrum_energy_on_ks(
    spectrum: dict[str, float],
    ks: tuple[int, ...],
) -> dict[str, float]:
    """How much of a per_irrep_energy_spectrum lies in ⊕_{k ∈ ks} V_k.

    spectrum keys are alpha_0, alpha_1, … as returned by per_irrep_energy_spectrum.

    Returns:
      energy_K_total_frac     — ∑_{k∈ks} alpha_k  (fraction of total ‖·‖_F²)
      energy_K_nontrivial_frac — ∑_{k∈ks} alpha_k / (1 − alpha_0); energy among
                                non-trivial modes that lies on the chosen K
      epsilon_off_K_nt        — 1 − energy_K_nontrivial_frac (small ⇒ aligned with K)
    """
    energy_K = sum(float(spectrum.get(f"alpha_{k}", 0.0)) for k in ks)
    alpha_0 = float(spectrum.get("alpha_0", 0.0))
    energy_nontrivial = max(1.0 - alpha_0, 1e-12)
    frac_nt = energy_K / energy_nontrivial
    return {
        "energy_K_total_frac": energy_K,
        "energy_K_nontrivial_frac": frac_nt,
        "epsilon_off_K_nt": 1.0 - frac_nt,
    }


def project_onto_canonical(
    weight: torch.Tensor,
    p: int,
    ks: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Project W (shape p×d) onto the Fourier subspace spanned by modes ks.

    If ks is None, projects onto ALL non-trivial modes (equivalent to removing
    the DC component). If ks is a tuple of integers, projects onto the union
    ⊕_{k ∈ ks} V_k.
    """
    device = weight.device
    dtype = weight.dtype

    if ks is None:
        # Remove DC: Π_{V*} W = W − v0 (v0ᵀ W)
        v0 = torch.ones(p, device=device, dtype=dtype) / math.sqrt(p)
        return weight - v0.unsqueeze(1) * (v0 @ weight).unsqueeze(0)

    proj = torch.zeros_like(weight)
    for k in ks:
        q = fourier_mode_subspace(p, k, device=device, dtype=dtype)  # (p, 2)
        proj = proj + q @ (q.T @ weight)
    return proj


def frob_sq_off_canonical(
    weight: torch.Tensor,
    p: int,
    ks: tuple[int, ...] = (1,),
) -> torch.Tensor:
    """Differentiable ‖Π_{(V_ks)⊥} W‖²_F  — the canonical regulariser term.

    Penalises energy in W outside the Fourier subspace spanned by modes ks.
    Default ks=(1,) enforces the *minimal* canonical representation (k=1 only).
    Use as: loss = ce_loss + lambda_canonical * frob_sq_off_canonical(W, p, ks).
    """
    proj = project_onto_canonical(weight, p, ks)
    return (weight - proj).pow(2).sum()


def build_delta_permutation(
    p: int,
    class_to_element: dict[int, int],
    element_to_class: dict[int, int],
) -> torch.Tensor:
    """Build an integer tensor perm of shape (p-1, p).

    perm[delta-1, y_class] = class index of the competitor at group offset delta,
    given that the correct class is y_class.  Used to vectorise margin computation.
    """
    perm = torch.zeros(p - 1, p, dtype=torch.long)
    for delta in range(1, p):
        for y_class in range(p):
            c_star = class_to_element[y_class]
            c_comp = (c_star + delta) % p
            perm[delta - 1, y_class] = element_to_class[c_comp]
    return perm


def empirical_margins(
    model: nn.Module,
    full_loader: DataLoader,
    p: int,
    device: torch.device,
    perm: torch.Tensor,
    max_samples: int = 512,
) -> dict[str, object]:
    """Estimate empirical margins γ_Δ from a random sample of examples.

    γ_Δ = mean_{(a,b)} [ logit_{c*}(a,b) − logit_{c*⊕Δ}(a,b) ]

    The margin curve is smooth in Δ, so max_samples ≈ 512 gives accurate
    estimates of γ_min and the curve shape without a full-dataset pass.

    Args:
        model: The model (set to eval mode internally).
        full_loader: DataLoader (finite) used to draw samples.
        p: Prime modulus.
        device: Torch device.
        perm: Integer tensor of shape (p-1, p) from build_delta_permutation.
        max_samples: Maximum number of examples to use (default 512).

    Returns dict with:
      margin_curve          — np.ndarray shape (p-1,), γ_1 … γ_{p-1}
      margin_min            — float, min_Δ γ_Δ
      margin_mean           — float, mean_Δ γ_Δ
      margin_std            — float, std_Δ γ_Δ
      fourier_amplitudes    — np.ndarray shape (p//2,), |DFT of margin curve|
    """
    was_training = model.training
    model.eval()

    perm_dev = perm.to(device)
    margin_sums = torch.zeros(p - 1, device=device, dtype=torch.float32)
    count = 0

    with torch.no_grad():
        for x, y in full_loader:
            if count >= max_samples:
                break
            x_dev = x.to(device)
            y_dev = y.to(device)
            logits, _, _ = model(x_dev)
            logits = logits[:, -1, :].float()  # (batch, p)
            correct_logits = logits[torch.arange(len(y_dev), device=device), y_dev]  # (batch,)

            for delta_idx in range(p - 1):
                comp_classes = perm_dev[delta_idx, y_dev]  # (batch,)
                comp_logits = logits[torch.arange(len(y_dev), device=device), comp_classes]
                margin_sums[delta_idx] += (correct_logits - comp_logits).sum()
            count += len(y_dev)

    if was_training:
        model.train()

    curve = (margin_sums / max(count, 1)).cpu().numpy().astype(np.float32)

    # Fourier decomposition over Z_p (NOT over the p-1 length signal).
    # γ_Δ is defined for Δ ∈ {1,...,p-1}; γ_0 = 0 by convention (correct class beats itself).
    # Prepend 0 to get a length-p signal over {0,...,p-1}, then rfft gives
    # coefficients at frequencies k/p — these are exactly the Z_p irrep indices k.
    # fft_amps[k] = |DFT(γ)[k]| for k=0,...,p//2
    # k=0 is the trivial (DC) mode; k=1,...,(p-1)//2 are the non-trivial irreps.
    curve_zp = np.concatenate([[0.0], curve])  # length p, Δ=0 prepended
    fft_amps = np.abs(np.fft.rfft(curve_zp))  # shape (p//2 + 1,)

    return {
        "margin_curve": curve,
        "margin_min": float(curve.min()),
        "margin_mean": float(curve.mean()),
        "margin_std": float(curve.std()),
        "fourier_amplitudes": fft_amps,
    }


def mean_logit_canonical_deviation(
    model: nn.Module,
    full_loader: DataLoader,
    p: int,
    ks: tuple[int, ...],
    device: torch.device,
    *,
    max_examples: int | None = None,
) -> dict[str, float]:
    """Representational deviation in logit space (paper Definition, Sec. 3.2).

    For each input, logits :math:`z = W_L\\phi(a,b) \\in \\mathbb{R}^p` (last-position logits).
    Let :math:`V^* = \\bigoplus_{k \\in K} V_k` be the span of the real Fourier mode subspaces
    on class coordinates. Returns the pooled ratio

    .. math::

        \\frac{\\sum_{(a,b)} \\|\\Pi_{(V^*)^\\perp} z\\|^2}{\\sum_{(a,b)} \\|z\\|^2}.

    When ``max_examples`` is ``None``, every batch from ``full_loader`` is used (typically all
    :math:`p^2` pairs for modular addition).
    """
    if not ks:
        msg = "ks must be non-empty; use e.g. reference_k_theory(p, ...) for K."
        raise ValueError(msg)

    was_training = model.training
    model.eval()

    tot_orth_sq = 0.0
    tot_norm_sq = 0.0
    n_ex = 0

    with torch.no_grad():
        for x, _y in full_loader:
            if max_examples is not None and n_ex >= max_examples:
                break
            x_dev = x.to(device)
            logits, _, _ = model(x_dev)
            z = logits[:, -1, :].float()
            if max_examples is not None:
                remaining = max_examples - n_ex
                if z.shape[0] > remaining:
                    z = z[:remaining]

            proj = torch.zeros_like(z)
            for k in ks:
                q = fourier_mode_subspace(p, k, device=device, dtype=z.dtype)
                proj = proj + z @ q @ q.T

            orth = z - proj
            tot_orth_sq += orth.pow(2).sum().item()
            tot_norm_sq += z.pow(2).sum().item()
            n_ex += z.shape[0]

    if was_training:
        model.train()

    denom = max(tot_norm_sq, 1e-12)
    return {
        "epsilon_logits_off_Vstar": float(tot_orth_sq / denom),
        "n_logit_examples": float(n_ex),
        "total_logit_norm_sq": float(tot_norm_sq),
    }


def per_sample_excess_loss(
    model: nn.Module,
    full_loader: DataLoader,
    p: int,
    ks: tuple[int, ...],
    device: torch.device,
    *,
    max_examples: int | None = None,
) -> dict[str, np.ndarray]:
    """Per-sample predicted vs observed excess CE loss (paper Theorem 2).

    For each input i with last-position logits :math:`z_i \\in \\mathbb{R}^p`:
        :math:`z^*_i = \\Pi_{V^*} z_i`         (canonical projection, :math:`V^* = \\bigoplus_{k\\in ks} V_k`)
        :math:`H_i  = z_i - z^*_i`             (deviation in :math:`(V^*)^\\perp`)

    Predicted excess loss (Theorem 2):
        :math:`d_L(H_i) = \\log \\sum_c e^{z^*_{ic} + H_{ic}} - \\log \\sum_c e^{z^*_{ic}} - H_{i, c^*_i}`

    Observed excess loss:
        :math:`L(z_i) - L(z^*_i)`
        :math:`= [\\mathrm{lse}(z_i) - z_{i,c^*_i}] - [\\mathrm{lse}(z^*_i) - z^*_{i,c^*_i}]`

    These two are algebraically identical (since :math:`z = z^* + H`); the comparison
    serves as a numerical sanity check.  The empirically meaningful quantity is
    ``predicted`` vs ``ce_loss`` — i.e. how well the deviation alone predicts the
    full per-sample CE.

    Returns dict with numpy arrays of length n:
        predicted        — :math:`d_L(H_i)`
        observed         — :math:`L(z_i) - L(z^*_i)`
        h_norm_sq        — :math:`\\|H_i\\|^2`
        ce_loss          — :math:`L(z_i)` (full CE on sample i)
        canonical_loss   — :math:`L(z^*_i)`
    """
    if not ks:
        msg = "ks must be non-empty"
        raise ValueError(msg)

    was_training = model.training
    model.eval()

    qs = [fourier_mode_subspace(p, k, device=device, dtype=torch.float32) for k in ks]

    pred_list: list[np.ndarray] = []
    obs_list: list[np.ndarray] = []
    hsq_list: list[np.ndarray] = []
    ce_list: list[np.ndarray] = []
    canon_list: list[np.ndarray] = []
    n_seen = 0

    with torch.no_grad():
        for x, y in full_loader:
            if max_examples is not None and n_seen >= max_examples:
                break
            x_dev = x.to(device)
            y_dev = y.to(device)
            logits, _, _ = model(x_dev)
            z = logits[:, -1, :].float()  # (B, p)
            if max_examples is not None:
                rem = max_examples - n_seen
                if z.shape[0] > rem:
                    z = z[:rem]
                    y_dev = y_dev[:rem]

            z_star = torch.zeros_like(z)
            for q in qs:
                z_star = z_star + z @ q @ q.T
            h = z - z_star

            idx = torch.arange(z.shape[0], device=device)
            lse_z = torch.logsumexp(z, dim=-1)
            lse_zstar = torch.logsumexp(z_star, dim=-1)

            predicted = lse_z - lse_zstar - h[idx, y_dev]
            ce = lse_z - z[idx, y_dev]
            canon = lse_zstar - z_star[idx, y_dev]
            observed = ce - canon

            pred_list.append(predicted.cpu().numpy())
            obs_list.append(observed.cpu().numpy())
            hsq_list.append(h.pow(2).sum(dim=-1).cpu().numpy())
            ce_list.append(ce.cpu().numpy())
            canon_list.append(canon.cpu().numpy())
            n_seen += z.shape[0]

    if was_training:
        model.train()

    return {
        "predicted": np.concatenate(pred_list),
        "observed": np.concatenate(obs_list),
        "h_norm_sq": np.concatenate(hsq_list),
        "ce_loss": np.concatenate(ce_list),
        "canonical_loss": np.concatenate(canon_list),
    }


def class_mean_activations(
    model: nn.Module,
    full_loader: DataLoader,
    p: int,
    device: torch.device,
    max_samples: int = 2048,
) -> torch.Tensor:
    """Collect last hidden states grouped by output class, return class-mean matrix H ∈ R^{p×d}.

    For each input (a, b), collects the hidden state at the last sequence position
    (the "=" token, position index -1 in the last transformer layer), which is the
    representation the readout acts on.  Groups by label c = (a+b) mod p and averages.

    H[c, :] = mean_{(a,b): a+b≡c} h_{last}(a, b)

    H has the same shape as W_L (p × d).  Applying per_irrep_energy_spectrum(H, p)
    gives the proper activation-space canonical geometry, not just the readout geometry.
    """
    was_training = model.training
    model.eval()

    hidden_dim: int | None = None
    sums: torch.Tensor | None = None
    counts = torch.zeros(p, dtype=torch.long)
    n_seen = 0

    with torch.no_grad():
        for x, y in full_loader:
            if n_seen >= max_samples:
                break
            x_dev = x.to(device)
            _, _, hidden_states_list = model(x_dev)
            # Last layer hidden states: shape (batch, seq_len, d)
            h_last_layer = hidden_states_list[-1]
            # Take last sequence position (the "=" token)
            h = h_last_layer[:, -1, :].float().cpu()  # (batch, d)

            if hidden_dim is None:
                hidden_dim = h.shape[-1]
                sums = torch.zeros(p, hidden_dim)

            y_cpu = y.cpu()
            for c in range(p):
                mask = y_cpu == c
                if mask.any():
                    sums[c] += h[mask].sum(dim=0)
                    counts[c] += mask.sum()
            n_seen += len(y_cpu)

    if was_training:
        model.train()

    if sums is None:
        msg = "No samples collected — full_loader may be empty."
        raise RuntimeError(msg)

    # Avoid division by zero for unseen classes
    safe_counts = counts.float().clamp_min(1.0).unsqueeze(1)
    H = sums / safe_counts  # (p, d)

    # Center H: subtract the global mean across classes.
    # The DC component (alpha_0) is just the global mean activation and is
    # uninformative about how the model represents different classes.
    # The canonical geometry lives in the VARIATION of H across classes.
    H = H - H.mean(dim=0, keepdim=True)

    return H


def compute_K_min(
    p: int,
    alpha: float = 1.0,
    sigma: float = 0.0,
    epsilon_target: float = 0.01,
    max_modes: int | None = None,
) -> dict[str, object]:
    """Compute the theoretically minimal active irrep set K for Z_p (Equation 8 of the paper).

    Two stopping criteria (whichever is reached first):
      (a) Loss bound:  L(K, alpha) ≤ epsilon_target
      (b) Noise bound: min_Δ γ_Δ > 2σ  (if sigma > 0)

    Uses greedy mode selection — at each step adds the k that maximally reduces the
    cross-entropy loss, which matches the analysis in irrep_ce_analysis.py.

    Args:
        p:              Prime modulus.
        alpha:          Per-mode amplitude (assumed equal for all modes).
                        Estimated from the model as mean logit Fourier amplitude.
        sigma:          Noise floor (logit perturbation bound, Section 3.1).
                        Set to 0 to use loss criterion only.
        epsilon_target: Loss target ε (default 0.01 ≈ near-perfect CE loss).
        max_modes:      Maximum number of modes to consider (default (p-1)//2).

    Returns dict with:
      K_min          — int, minimum number of non-trivial irreps needed
      active_ks      — list[int], the greedy-optimal frequency indices
      loss_per_step  — list[float], CE loss after adding each mode
      min_margin_per_step — list[float], min_Δ γ_Δ after adding each mode
      margin_curve   — np.ndarray shape (p-1,), final γ_Δ curve
      loss_baseline  — float, log(p) (random-guess CE loss)
      loss_target    — float, epsilon_target
      noise_threshold — float, 2*sigma
    """
    deltas = np.arange(1, p, dtype=np.float64)
    n_max = max_modes if max_modes is not None else (p - 1) // 2
    all_ks = list(range(1, (p - 1) // 2 + 1))

    def _gammas(ks: list[int]) -> np.ndarray:
        g = np.zeros(p - 1, dtype=np.float64)
        for k in ks:
            g += alpha * (1.0 - np.cos(2.0 * np.pi * k * deltas / p))
        return g

    def _ce_loss(ks: list[int]) -> float:
        g = _gammas(ks)
        z = np.sum(np.exp(-np.clip(g, 0, 500)))
        return float(np.log(1.0 + z))

    active: list[int] = []
    loss_per_step: list[float] = [float(np.log(p))]  # baseline: log(p)
    min_margin_per_step: list[float] = [0.0]          # no modes: min margin = 0

    for _ in range(n_max):
        # Greedy: pick the k not yet in active that most reduces CE loss
        best_k, best_loss = None, float("inf")
        for k in all_ks:
            if k in active:
                continue
            candidate_loss = _ce_loss(active + [k])
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_k = k

        if best_k is None:
            break

        active.append(best_k)
        current_loss = _ce_loss(active)
        current_gammas = _gammas(active)
        current_min_margin = float(current_gammas.min())

        loss_per_step.append(current_loss)
        min_margin_per_step.append(current_min_margin)

        loss_ok = current_loss <= epsilon_target
        margin_ok = sigma <= 0.0 or current_min_margin > 2.0 * sigma
        if loss_ok and margin_ok:
            break

    final_gammas = _gammas(active)

    return {
        "K_min": len(active),
        "active_ks": active,
        "loss_per_step": loss_per_step,
        "min_margin_per_step": min_margin_per_step,
        "margin_curve": final_gammas,
        "loss_baseline": float(np.log(p)),
        "loss_target": float(epsilon_target),
        "noise_threshold": float(2.0 * sigma),
    }


def embedding_geometry(
    embed_weight: torch.Tensor,
    p: int,
    elem_token_offset: int = 2,
) -> dict[str, float]:
    """Measure Fourier structure of token embeddings for Z_p elements.

    For modular arithmetic the canonical mechanism (Nanda et al., 2023) works by
    encoding each element a ∈ Z_p as an irrep circle:
        embed(a) ≈ ∑_k [ cos(2πka/p) · u_k  +  sin(2πka/p) · v_k ]
    where u_k, v_k ∈ ℝ^d span the k-th Fourier subspace.

    This is the *most fundamental* test of the canonical representation.
    It checks whether the *input* embedding encodes Z_p group structure as
    Fourier irreps — before any transformer processing.

    Args:
        embed_weight:      nn.Embedding weight matrix, shape (vocab_size, d).
                           Token IDs for Z_p elements occupy positions
                           [elem_token_offset, elem_token_offset + p).
        p:                 Prime modulus — number of group elements.
        elem_token_offset: First token ID for Z_p elements (default 2, since
                           token 0='op' and token 1='=' are reserved vocabulary slots).

    Returns:
        Same dict structure as per_irrep_energy_spectrum: alpha_0, alpha_k,
        dominant_k, top1_energy, top3_energy, epsilon, epsilon_dc,
        spectrum_entropy.
    """
    E = embed_weight[elem_token_offset : elem_token_offset + p].float()  # (p, d)
    # Center: canonical geometry is in the VARIATION across elements;
    # the global mean is an uninformative constant offset common to all inputs.
    E = E - E.mean(dim=0, keepdim=True)
    return per_irrep_energy_spectrum(E, p)


def predicted_margin_curve(weight: torch.Tensor, p: int) -> np.ndarray:
    """Predicted γ_Δ curve from W_L via the Fourier decomposition formula.

    Uses ‖Π_{V_k} W‖_F as a proxy for the per-mode amplitude α_k:
        γ_Δ ≈ ∑_{k=1}^{(p-1)/2} ‖Π_{V_k} W‖_F · (1 − cos(2πkΔ/p))

    This gives the correct *shape* of the margin curve.  Absolute scale differs
    from the empirical curve because it ignores the hidden representation amplitude.
    """
    device = weight.device
    dtype = weight.dtype
    deltas = np.arange(1, p, dtype=np.float64)
    curve = np.zeros(p - 1, dtype=np.float64)
    for k in range(1, (p - 1) // 2 + 1):
        q = fourier_mode_subspace(p, k, device=device, dtype=dtype)
        alpha_k = (q.T @ weight).pow(2).sum().sqrt().item()
        curve += alpha_k * (1.0 - np.cos(2.0 * math.pi * k * deltas / p))
    return curve
