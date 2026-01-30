"""Geometry analysis for grokking: Fisher, curvature, sharpness, path, representation."""

from grokking.geometry.analysis import compute_geometry_at_checkpoint
from grokking.geometry.curvature import curvature_proxy_from_fisher_eigenvalues
from grokking.geometry.fisher import empirical_fisher_top_eigenvalues
from grokking.geometry.path import get_flat_params, segment_natural_length
from grokking.geometry.representation import representation_spectrum
from grokking.geometry.sharpness import hutchinson_hessian_trace

__all__ = [
    "compute_geometry_at_checkpoint",
    "curvature_proxy_from_fisher_eigenvalues",
    "empirical_fisher_top_eigenvalues",
    "get_flat_params",
    "segment_natural_length",
    "representation_spectrum",
    "hutchinson_hessian_trace",
]
