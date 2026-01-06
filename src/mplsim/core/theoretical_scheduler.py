"""
Theoretical Scheduler: Implements the paper's linear λ dynamics directly.

The paper's core equation (Eq. A11) is:
    λ(x) = γ·a(x) + α·⟨λ⟩_x

Where α is the coupling strength:
- α < 1: screened Poisson (short-range Yukawa)
- α = 1: pure Poisson (long-range Newtonian)

This scheduler directly computes the theoretical formula iteratively, useful for:
- Fast simulation when you don't need to prove emergence
- Comparing against BandwidthScheduler (which has true causal emergence)
- Testing theoretical predictions

Note: This scheduler computes the answer directly rather than letting it
emerge from microscopic dynamics. For true causal emergence, use BandwidthScheduler.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mplsim.core.lattice import Lattice
    from mplsim.core.source_map import SourceMap
    from mplsim.core.kernel import Kernel


@dataclass
class TheoreticalSchedulerConfig:
    """Configuration for the theoretical scheduler."""

    gamma: float = 0.5  # γ: local activity contribution to λ
    alpha: float = 0.9  # α (coupling): 1.0 = long-range Poisson, <1 = screened Yukawa
    lambda_decay: float = 0.01  # Natural decay to prevent unbounded growth
    ema_smoothing: float = 0.1  # EMA smoothing for λ updates
    link_capacity: float = 5.0  # Normalization for activity → λ


@dataclass
class TheoreticalScheduler:
    """
    Scheduler implementing the paper's linear λ dynamics directly.

    Directly computes λ(x) = γ·a(x) + α·⟨λ⟩_x at each step,
    then derives f from λ.

    This is a "theoretical" scheduler that computes the expected answer
    rather than letting it emerge from causal dynamics. Use BandwidthScheduler
    for true emergence.
    """

    lattice: "Lattice"
    source_map: "SourceMap"
    kernel: "Kernel"
    config: TheoreticalSchedulerConfig = field(default_factory=TheoreticalSchedulerConfig)

    # Simulation state
    current_tick: int = field(default=0, init=False)

    # The λ field is primary state; f is derived
    _lambda_field: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        """Initialize λ field."""
        ny, nx = self.lattice.shape
        self._lambda_field = np.zeros((ny, nx), dtype=np.float64)
        self.lattice.f.fill(1.0)

    def run(self, n_ticks: int) -> dict:
        """
        Run simulation for n ticks.

        Each tick applies the update:
            λ_new(x) = (1-decay) * [γ·a_norm(x) + α·⟨λ_old⟩_x]

        Args:
            n_ticks: Number of ticks to run

        Returns:
            Statistics dictionary
        """
        for _ in range(n_ticks):
            self._tick()
            self.current_tick += 1

        # Update f_smooth for particle gradient computation
        self.update_f_smooth()

        return {
            "n_ticks": n_ticks,
            "mean_lambda": float(self._lambda_field.mean()),
            "max_lambda": float(self._lambda_field.max()),
            "mean_f": float(self.lattice.f.mean()),
            "min_f": float(self.lattice.f.min()),
        }

    def _tick(self):
        """Execute one simulation tick."""
        gamma = self.config.gamma
        alpha = self.config.alpha
        decay = self.config.lambda_decay
        ema = self.config.ema_smoothing

        # Scale activity by link capacity (like bandwidth saturation)
        # Higher rates relative to capacity → more congestion → higher λ
        rates = self.source_map.rates
        capacity = self.config.link_capacity
        a_norm = rates / capacity

        # Compute neighbor average using boundary-aware shifting
        # For absorbing boundaries, missing neighbors contribute λ=0 (f=1 reference)
        avg_neighbor_lambda = (
            self.lattice.shift_field(self._lambda_field, 0, -1, fill_value=0.0) +  # N
            self.lattice.shift_field(self._lambda_field, 0, 1, fill_value=0.0) +   # S
            self.lattice.shift_field(self._lambda_field, 1, 0, fill_value=0.0) +   # E
            self.lattice.shift_field(self._lambda_field, -1, 0, fill_value=0.0)    # W
        ) / 4.0

        # Apply the paper's linear equation (Eq. A11):
        # λ = γ·a + α·⟨λ⟩
        # With decay to prevent unbounded growth:
        # λ_target = (1 - decay) * (γ·a + α·⟨λ⟩)
        lambda_target = (1.0 - decay) * (gamma * a_norm + alpha * avg_neighbor_lambda)

        # EMA smooth update
        self._lambda_field = (1 - ema) * self._lambda_field + ema * lambda_target

        # Clamp λ to [0, 1] since f = 1 - λ must be in [0, 1]
        self._lambda_field = np.clip(self._lambda_field, 0.0, 1.0)

        # Enforce Dirichlet BC: λ=0 at boundaries for absorbing BC
        if self.lattice.config.boundary == "absorbing":
            self._lambda_field[0, :] = 0.0   # Top edge
            self._lambda_field[-1, :] = 0.0  # Bottom edge
            self._lambda_field[:, 0] = 0.0   # Left edge
            self._lambda_field[:, -1] = 0.0  # Right edge

        # Derive f from λ using paper's definition (Eq. 6): f = 1 - λ
        self.lattice.f = 1.0 - self._lambda_field

    def get_lambda_field(self) -> np.ndarray:
        """Get current λ field."""
        return self._lambda_field.copy()

    @property
    def canonical_tick(self) -> int:
        """Current tick for compatibility."""
        return self.current_tick

    def update_f_smooth(self):
        """Update spatially smoothed f field."""
        from scipy.ndimage import gaussian_filter
        sigma = self.lattice.config.spatial_sigma
        if sigma > 0:
            self.lattice.f_smooth = gaussian_filter(
                self.lattice.f, sigma=sigma, mode=self.lattice.get_scipy_mode()
            )
        else:
            self.lattice.f_smooth = self.lattice.f.copy()
