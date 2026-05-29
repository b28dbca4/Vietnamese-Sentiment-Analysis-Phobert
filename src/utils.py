"""Cross-cutting utilities for reproducibility, timing, and formatting."""

from __future__ import annotations

import os
import random
import time
from types import TracebackType
from typing import Optional, Type

import numpy as np
import torch

from src import config


def set_seed(seed: int = config.SEED) -> None:
    """Ensure reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer random seed used by every supported library.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic kernels make comparison across experiments more reliable.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detect the best available compute device.

    Returns:
        A CUDA device when available, otherwise a CPU device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Using CUDA device: {gpu_name} ({total_memory_gb:.1f} GB)")
        return device

    device = torch.device("cpu")
    print("Using CPU device")
    return device


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count model parameters by trainable status.

    Args:
        model: PyTorch model to inspect.

    Returns:
        Dictionary with total, trainable, and non-trainable parameter counts.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "non_trainable": int(total - trainable),
    }


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Calculate a model's in-memory parameter footprint.

    Args:
        model: PyTorch model to inspect.

    Returns:
        Parameter size in megabytes.
    """
    param_size = sum(parameter.nelement() * parameter.element_size() for parameter in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    return float((param_size + buffer_size) / (1024**2))


def format_number(n: int | float) -> str:
    """Format large numbers with compact suffixes.

    Args:
        n: Numeric value to format.

    Returns:
        Human-readable string such as ``67.0M`` or ``1.2K``.
    """
    value = float(n)
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value)) if value.is_integer() else f"{value:.2f}"


class Timer:
    """Context manager for timing code blocks."""

    start: float
    elapsed: float

    def __enter__(self) -> "Timer":
        """Start the timer.

        Returns:
            The timer instance so callers can inspect ``elapsed`` afterward.
        """
        self.start = time.time()
        self.elapsed = 0.0
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Stop the timer when leaving the context."""
        self.elapsed = time.time() - self.start

    def __str__(self) -> str:
        """Format elapsed time as seconds, minutes, or hours."""
        seconds = int(round(self.elapsed))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {secs}s"
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"


def ensure_dirs() -> None:
    """Create all output directories required by the project."""
    directories = [
        config.MODELS_DIR,
        config.BEST_MODEL_DIR,
        config.BASELINE_DIR,
        config.LOG_DIR,
        config.RESULTS_DIR,
        config.FIGURES_DIR,
        config.METRICS_DIR,
    ]
    directories.extend(
        os.path.join(config.MODELS_DIR, model_name)
        for model_name in config.TRANSFORMER_MODELS.values()
    )

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

