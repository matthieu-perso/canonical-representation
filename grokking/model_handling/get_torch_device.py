"""Get the preferred torch device."""

import logging

import torch

from grokking.typing.enums import PreferredTorchBackend, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_torch_device(
    preferred_torch_backend: PreferredTorchBackend,
    verbosity: int = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
    *,
    cuda_device_id: int | None = None,
) -> torch.device:
    """Get the preferred torch device.

    When cuda_device_id is set and CUDA is available, that GPU is used (for parallel
    multirun so each job gets a different GPU).
    """
    # Directly select 'cpu' if preferred,
    # since it is always available
    if preferred_torch_backend == PreferredTorchBackend.CPU:
        device = torch.device(device="cpu")
    # For 'cuda', check if it is the preference
    # and if it is available
    elif preferred_torch_backend == PreferredTorchBackend.CUDA and torch.cuda.is_available():
        if cuda_device_id is not None and 0 <= cuda_device_id < torch.cuda.device_count():
            device = torch.device(device=f"cuda:{cuda_device_id}")
        else:
            device = torch.device(device="cuda")
    # For 'mps', check if it is the preference
    # and if it is available
    elif (
        preferred_torch_backend == PreferredTorchBackend.MPS and torch.backends.mps.is_available()
    ) or torch.backends.mps.is_available():
        device = torch.device(device="mps")
    elif torch.cuda.is_available():
        if cuda_device_id is not None and 0 <= cuda_device_id < torch.cuda.device_count():
            device = torch.device(device=f"cuda:{cuda_device_id}")
        else:
            device = torch.device(device="cuda")
    else:
        device = torch.device(device="cpu")

    if verbosity >= 1:
        logger.info(
            msg=f"Selected {device = }",  # noqa: G004 - low overhead
        )

    return device
