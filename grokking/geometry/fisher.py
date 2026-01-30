"""Empirical Fisher information matrix (FIM) and top eigenvalues via FVP."""

import torch


def _get_grad_vector(model: torch.nn.Module) -> torch.Tensor:
    """Flatten current param.grad into a single vector."""
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])


def _fvp_from_grads(
    grads: list[torch.Tensor],
    v: torch.Tensor,
) -> torch.Tensor:
    """Compute F @ v where F = (1/n) sum_i g_i g_i^T (empirical Fisher)."""
    if not grads:
        return torch.zeros_like(v)
    n = len(grads)
    out = sum((g.to(v.device).to(v.dtype) * (g.to(v.device).to(v.dtype) @ v)) for g in grads) / n
    return out


def empirical_fisher_top_eigenvalues(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    n_batches: int = 20,
    n_power_iter: int = 15,
    top_k: int = 5,
) -> list[float]:
    """
    Compute top-k eigenvalues of the empirical Fisher (batch-averaged grads).
    Uses power iteration; no full matrix stored.
    """
    model.eval()
    grads: list[torch.Tensor] = []
    n = 0
    for batch in dataloader:
        if n >= n_batches:
            break
        x, y = batch[0], batch[1]
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        loss, _ = model.get_loss(x, y)
        loss.backward()
        g = _get_grad_vector(model).detach()
        grads.append(g)
        n += 1

    if not grads:
        return [0.0] * top_k

    num_params = grads[0].numel()
    eigenvalues: list[float] = []

    # Power iteration for top eigenvalue; then deflate and repeat for next.
    remaining_grads = list(grads)
    for _ in range(top_k):
        if not remaining_grads:
            eigenvalues.append(0.0)
            continue
        v = torch.randn(num_params, device=device, dtype=grads[0].dtype)
        v = v / v.norm()
        for _ in range(n_power_iter):
            Fv = _fvp_from_grads(remaining_grads, v)
            v = Fv / (Fv.norm() + 1e-12)
        lam = (v * _fvp_from_grads(remaining_grads, v)).sum().item()
        eigenvalues.append(max(0.0, lam))
        # Deflate: remove component along v from each g so next eigenvalue is second largest.
        v_device = v.to(remaining_grads[0].device)
        remaining_grads = [g - (g.to(v_device.dtype) @ v_device) * v_device for g in remaining_grads]

    return eigenvalues


def empirical_fisher_top_eigenpairs(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    n_batches: int = 20,
    n_power_iter: int = 15,
    top_k: int = 5,
) -> tuple[list[float], list[torch.Tensor]]:
    """
    Compute top-k eigenpairs (eigenvalues, eigenvectors) of the empirical Fisher.
    Returns (eigenvalues, eigenvectors) where eigenvectors are tensors of shape (num_params,).
    Used for path length and finer geometry (no full d×d matrix).
    """
    model.eval()
    grads: list[torch.Tensor] = []
    n = 0
    for batch in dataloader:
        if n >= n_batches:
            break
        x, y = batch[0], batch[1]
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        loss, _ = model.get_loss(x, y)
        loss.backward()
        g = _get_grad_vector(model).detach()
        grads.append(g)
        n += 1

    if not grads:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        dtype = next((p.dtype for p in model.parameters() if p.requires_grad), torch.float32)
        return [0.0] * top_k, [torch.zeros(num_params, device=device, dtype=dtype) for _ in range(top_k)]

    num_params = grads[0].numel()
    eigenvalues: list[float] = []
    eigenvectors: list[torch.Tensor] = []
    remaining_grads = list(grads)
    for _ in range(top_k):
        if not remaining_grads:
            eigenvalues.append(0.0)
            eigenvectors.append(torch.zeros(num_params, device=device, dtype=grads[0].dtype))
            continue
        v = torch.randn(num_params, device=device, dtype=grads[0].dtype)
        v = v / (v.norm() + 1e-12)
        for _ in range(n_power_iter):
            Fv = _fvp_from_grads(remaining_grads, v)
            v = Fv / (Fv.norm() + 1e-12)
        lam = (v * _fvp_from_grads(remaining_grads, v)).sum().item()
        eigenvalues.append(max(0.0, lam))
        eigenvectors.append(v.detach().clone())
        v_device = v.to(remaining_grads[0].device)
        remaining_grads = [g - (g.to(v_device.dtype) @ v_device) * v_device for g in remaining_grads]

    return eigenvalues, eigenvectors
