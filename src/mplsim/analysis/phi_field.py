"""
Derive gravitational potential ϕ from the update fraction f(x).

IMPORTANT: This is a DERIVED quantity for analysis/comparison only.
The engine computes f(x) directly from congestion — ϕ is not used in the engine.

Two modes:
- Linear: ϕ = α(f - 1), so ϕ ∈ [-α, 0] where f ∈ [0, 1]
- Log: ϕ = α·log(f), asymptotes to -∞ as f → 0 (horizon-like)

In GR terms: f ≈ √(1 + 2ϕ/c²) for weak fields, so ϕ ≈ (f² - 1)c²/2 ≈ (f-1)c²
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class PhiFromF:
    """
    Compute gravitational potential ϕ from update fraction f.

    This is purely for analysis — the engine never uses ϕ.
    """

    mode: Literal["linear", "log"] = "linear"
    alpha: float = 1.0  # Scaling factor

    def compute(self, f: np.ndarray) -> np.ndarray:
        """
        Compute ϕ field from f field.

        Args:
            f: Update fraction field, shape [ny, nx], values in [0, 1]

        Returns:
            ϕ field, shape [ny, nx], values ≤ 0 (deeper = more negative)
        """
        if self.mode == "linear":
            # ϕ = α(f - 1), so ϕ = 0 when f = 1, ϕ = -α when f = 0
            return self.alpha * (f - 1.0)

        elif self.mode == "log":
            # ϕ = α·log(f), asymptotes to -∞ as f → 0
            # Clip f to avoid log(0)
            f_safe = np.clip(f, 1e-10, 1.0)
            return self.alpha * np.log(f_safe)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def compute_gradient(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient of ϕ (which determines gravitational acceleration).

        Returns:
            (dϕ/dx, dϕ/dy) - gradient components
        """
        phi = self.compute(f)

        # Use central differences for gradient
        dphi_dy, dphi_dx = np.gradient(phi)

        return dphi_dx, dphi_dy

    def compute_acceleration(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gravitational acceleration a = -∇ϕ.

        Returns:
            (ax, ay) - acceleration components pointing toward mass
        """
        dphi_dx, dphi_dy = self.compute_gradient(f)
        return -dphi_dx, -dphi_dy


def compute_phi_linear(f: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Convenience function for linear ϕ computation."""
    return PhiFromF(mode="linear", alpha=alpha).compute(f)


def compute_phi_log(f: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Convenience function for log ϕ computation."""
    return PhiFromF(mode="log", alpha=alpha).compute(f)
