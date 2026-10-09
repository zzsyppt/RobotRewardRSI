"""Isaac Lab 3 array/rotation representation at the Torch consumer boundary."""

from __future__ import annotations

from typing import Any


def as_torch(value: Any) -> Any:
    """Read a zero-copy ProxyArray view; existing Torch buffers stay unchanged."""
    return getattr(value, "torch", value)


def quaternion_wxyz(value: Any) -> Any:
    """Convert Lab 3's xyzw rotation to the project's frozen public wxyz ABI."""
    return as_torch(value)[..., [3, 0, 1, 2]]
