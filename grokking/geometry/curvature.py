"""Curvature proxies from Fisher eigenvalues."""

from __future__ import annotations


def curvature_proxy_from_fisher_eigenvalues(
    eigenvalues: list[float],
) -> dict[str, float]:
    """
    Simple curvature-related quantities from Fisher eigenvalues.
    - fisher_max_eig: largest eigenvalue (inverse of smallest length scale).
    - fisher_cond: condition number (max/min of top eigenvalues, or 1 if min is 0).
    """
    if not eigenvalues:
        return {"fisher_max_eig": 0.0, "fisher_cond": 1.0}
    max_eig = max(eigenvalues)
    min_eig = min(e for e in eigenvalues if e > 0) if any(e > 0 for e in eigenvalues) else max_eig or 1.0
    cond = max_eig / min_eig if min_eig > 0 else 1.0
    out = {
        "fisher_max_eig": max_eig,
        "fisher_cond": cond,
    }
    out["curvature_proxy"] = curvature_proxy_scalar(eigenvalues)
    return out


def curvature_proxy_scalar(eigenvalues: list[float]) -> float:
    """
    Scalar curvature proxy from Fisher eigenvalue spectrum.
    R_proxy = (λ_max - λ_min) / (λ_max + λ_min) in [0, 1]; high = more anisotropic/curved.
    Uses only the top-k eigenvalues we have; interpret as local anisotropy of the metric.
    """
    if not eigenvalues or len(eigenvalues) < 2:
        return 0.0
    pos = [e for e in eigenvalues if e > 0]
    if len(pos) < 2:
        return 0.0
    lam_max = max(pos)
    lam_min = min(pos)
    return (lam_max - lam_min) / (lam_max + lam_min + 1e-12)
