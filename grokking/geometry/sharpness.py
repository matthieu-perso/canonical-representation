"""Sharpness via Hutchinson trace of the Hessian (HVP)."""

import torch


def _hvp(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    v_list: list[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    """Hessian-vector product: H @ v where H is d²loss/dparams². Returns list of Hv per param."""
    model.zero_grad()
    loss, _ = model.get_loss(x, y)
    loss.backward(create_graph=True)
    grad_params = [p.grad.clone() for p in model.parameters() if p.requires_grad]
    # d/dparams (grad . v) = H v
    dot = sum((gp.flatten() * vp.flatten()).sum() for gp, vp in zip(grad_params, v_list))
    model.zero_grad()
    dot.backward()
    return [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters() if p.requires_grad]


def hutchinson_hessian_trace(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    n_samples: int = 20,
) -> float:
    """
    Estimate trace(Hessian of loss) via Hutchinson: (1/k) sum_i v_i^T H v_i with Rademacher v_i.
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    traces: list[float] = []
    batch_iter = iter(dataloader)
    for i in range(n_samples):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(dataloader)
            batch = next(batch_iter)
        x, y = batch[0].to(device), batch[1].to(device)
        # Rademacher v matching param shapes
        v_list = [torch.randint_like(p, high=2, device=device, dtype=p.dtype).float() * 2 - 1 for p in params]
        hv_list = _hvp(model, x, y, v_list, device)
        # v^T H v = sum over params of (v * Hv).sum()
        v_h_v = sum((v.flatten() * hv.flatten()).sum().item() for v, hv in zip(v_list, hv_list))
        traces.append(v_h_v)
    return sum(traces) / len(traces) if traces else 0.0
