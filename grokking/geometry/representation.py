"""Representation-space geometry: spectrum of the covariance of hidden states."""

from __future__ import annotations

import torch


def representation_spectrum(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    *,
    n_batches: int = 20,
    top_k: int = 10,
    layer_index: int = -1,
) -> list[float]:
    """
    Top eigenvalues of the covariance of last-layer hidden states (last token).
    Characterizes the shape of the representation cloud: effective dimension, anisotropy.
    layer_index: which layer's hidden state (-1 = last layer).
    """
    model.eval()
    hidden_list: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            if len(hidden_list) >= n_batches:
                break
            x = batch[0].to(device)
            _, _, hidden_states_over_layers = model(x)
            # hidden_states_over_layers[layer_index] shape (batch, time, hidden_dim)
            h = hidden_states_over_layers[layer_index][:, -1, :]  # last token, (batch, hidden_dim)
            hidden_list.append(h)
    if not hidden_list:
        return [0.0] * top_k
    H = torch.cat(hidden_list, dim=0)  # (N, hidden_dim)
    H = H - H.mean(dim=0)
    # Covariance = H^T H / N; top eigenvalues via power iteration on H^T H
    n, d = H.shape
    if n < 2 or d < 1:
        return [0.0] * top_k
    H = H.to(device)
    remaining = H.clone()
    eigenvalues: list[float] = []
    for _ in range(min(top_k, d, n)):
        v = torch.randn(d, device=device, dtype=H.dtype)
        v = v / (v.norm() + 1e-12)
        for _ in range(15):
            # (H^T H) v = H^T (H v)
            u = remaining @ v
            v = (remaining.T @ u) / (u.norm() + 1e-12)
            v = v / (v.norm() + 1e-12)
        lam = (v * (remaining.T @ (remaining @ v))).sum().item() / n
        lam = max(0.0, lam)
        eigenvalues.append(lam)
        # Deflate: remove component along v from each row of remaining
        u = remaining @ v
        remaining = remaining - u.unsqueeze(1) * v.unsqueeze(0)
    while len(eigenvalues) < top_k:
        eigenvalues.append(0.0)
    return eigenvalues[:top_k]
