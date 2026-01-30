"""Path geometry: natural (Fisher) length along the training trajectory."""

from __future__ import annotations

import math

import torch

from grokking.geometry.fisher import empirical_fisher_top_eigenpairs


def get_flat_params(model: torch.nn.Module, device: torch.device | None = None) -> torch.Tensor:
    """Flatten trainable parameters into a single vector. On model's device or given device."""
    params = [p.detach().flatten() for p in model.parameters() if p.requires_grad]
    if not params:
        return torch.tensor([], device=next(model.parameters()).device)
    out = torch.cat(params)
    if device is not None:
        out = out.to(device)
    return out


def segment_natural_length(
    theta_prev: torch.Tensor,
    theta_curr: torch.Tensor,
    model: torch.nn.Module,
    train_dataloader,
    device: torch.device,
    *,
    n_fisher_batches: int = 20,
    top_k: int = 10,
) -> float:
    """
    Approximate natural (Fisher-Rao) length of the segment from theta_prev to theta_curr.
    Uses low-rank Fisher at current point: length^2 ≈ sum_i λ_i (v_i · Δθ)^2, Δθ = theta_curr - theta_prev.
    """
    delta = (theta_curr - theta_prev.to(device)).to(theta_curr.dtype)
    eigs, evecs = empirical_fisher_top_eigenpairs(
        model=model,
        dataloader=train_dataloader,
        device=device,
        n_batches=n_fisher_batches,
        top_k=top_k,
    )
    length_sq = 0.0
    for lam, v in zip(eigs, evecs):
        if lam <= 0:
            continue
        v = v.to(device).to(delta.dtype)
        dot = (v * delta).sum().item()
        length_sq += lam * (dot * dot)
    return math.sqrt(max(0.0, length_sq))
