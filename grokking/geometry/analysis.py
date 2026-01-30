"""Single entry point: compute geometry metrics at a checkpoint."""

from __future__ import annotations

from grokking.geometry.curvature import curvature_proxy_from_fisher_eigenvalues
from grokking.geometry.fisher import empirical_fisher_top_eigenvalues
from grokking.geometry.representation import representation_spectrum
from grokking.geometry.sharpness import hutchinson_hessian_trace


def compute_geometry_at_checkpoint(
    model,
    train_dataloader,
    device,
    *,
    step: int,
    n_fisher_batches: int = 20,
    n_fisher_eigenvalues: int = 5,
    n_hutchinson: int = 20,
    n_representation_batches: int = 20,
    n_representation_eigenvalues: int = 10,
    metrics: list[str] | None = None,
) -> dict:
    """
    Compute geometry metrics for the given model and dataloader.
    Returns dict with step, fisher, curvature_proxy, sharpness, representation_spectrum, etc.
    """
    if metrics is None:
        metrics = ["fisher", "curvature", "sharpness", "representation"]
    out: dict = {"step": step}

    if "fisher" in metrics or "curvature" in metrics:
        fisher_eigs = empirical_fisher_top_eigenvalues(
            model=model,
            dataloader=train_dataloader,
            device=device,
            n_batches=n_fisher_batches,
            top_k=n_fisher_eigenvalues,
        )
        out["fisher_top_eigenvalues"] = fisher_eigs
        out["fisher_max_eig"] = max(fisher_eigs) if fisher_eigs else 0.0

    if "curvature" in metrics:
        eigs = out.get("fisher_top_eigenvalues", [])
        out.update(curvature_proxy_from_fisher_eigenvalues(eigs))

    if "sharpness" in metrics:
        out["sharpness_hessian_trace"] = hutchinson_hessian_trace(
            model=model,
            dataloader=train_dataloader,
            device=device,
            n_samples=n_hutchinson,
        )

    if "representation" in metrics:
        out["representation_spectrum"] = representation_spectrum(
            model=model,
            dataloader=train_dataloader,
            device=device,
            n_batches=n_representation_batches,
            top_k=n_representation_eigenvalues,
        )

    return out
