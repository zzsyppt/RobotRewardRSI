"""Independent true RL agent package.

The package root intentionally imports only local modules. Parent project code is
reference material and must not be a runtime dependency.
"""

from ._boundary import RUNTIME_BOUNDARY, RuntimeBoundary

__all__ = ["RUNTIME_BOUNDARY", "RuntimeBoundary"]
__version__ = "0.1.0"

