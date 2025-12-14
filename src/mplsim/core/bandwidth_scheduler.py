"""
Bandwidth Scheduler: Emergent gravity from bandwidth limits.

Implements the paper's equation: λ(x) = γ·a(x) + β·⟨λ⟩_x

Each tick:
1. Generate message sizes ~ Poisson(activity × scale)
2. Local stall if message_size > capacity
3. Sync contribution = β × average neighbor λ
4. Update λ iteratively (relaxation toward steady state)

f(x) = 1 - λ(x) is the update fraction (gravity field).
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
class BandwidthSchedulerConfig:
    """Configuration for bandwidth scheduler."""
    link_capacity: float = 10.0      # Max message size per tick
    message_scale: float = 10.0      # message_size ~ Poisson(activity × scale)
    beta: float = 0.9                # Sync coupling: λ includes β·⟨λ⟩_neighbors
    stochastic_messages: bool = True # Poisson (True) or deterministic (False)


@dataclass
class BandwidthScheduler:
    """Scheduler implementing λ = local_stall + β·⟨λ⟩ iteratively."""

    lattice: "Lattice"
    source_map: "SourceMap"
    kernel: "Kernel"
    config: BandwidthSchedulerConfig = field(default_factory=BandwidthSchedulerConfig)

    current_tick: int = field(default=0, init=False)
    total_updates: int = field(default=0, init=False)
    _lambda_field: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        ny, nx = self.lattice.shape
        self._lambda_field = np.zeros((ny, nx), dtype=np.float64)
        self.lattice.f.fill(1.0)

    def run(self, n_ticks: int) -> dict:
        """Run simulation for n_ticks."""
        for _ in range(n_ticks):
            self._tick()
        self.update_f_smooth()

        return {
            "n_ticks": n_ticks,
            "current_tick": self.current_tick,
            "total_updates": self.total_updates,
            "mean_lambda": float(self._lambda_field.mean()),
            "max_lambda": float(self._lambda_field.max()),
            "mean_f": float(self.lattice.f.mean()),
            "min_f": float(self.lattice.f.min()),
        }

    def _tick(self):
        """One tick of the simulation.

        Update rule (Jacobi relaxation):
            λ_new(x,y) = local_stall(x,y) + β × [λ(x,y-1) + λ(x,y+1) + λ(x-1,y) + λ(x+1,y)] / 4

        The gradient emerges from finite propagation speed (1 cell/tick).
        """
        self.current_tick += 1
        cfg = self.config

        # Local stall: message_size > capacity
        if cfg.stochastic_messages:
            msg_sizes = np.random.poisson(self.source_map.rates * cfg.message_scale)
        else:
            msg_sizes = self.source_map.rates * cfg.message_scale
        local_stall = (msg_sizes > cfg.link_capacity).astype(np.float64)

        # Sync contribution: β × neighbor average
        neighbor_avg = self._neighbor_avg(self._lambda_field)

        # Update λ (relaxation toward steady state)
        self._lambda_field = np.clip(local_stall + cfg.beta * neighbor_avg, 0.0, 1.0)
        self.lattice.f = 1.0 - self._lambda_field

        self.total_updates += int(np.sum(1.0 - local_stall))

    def _neighbor_avg(self, field: np.ndarray) -> np.ndarray:
        """Average of Von Neumann neighbors (4-connected)."""
        return (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
        ) / 4.0

    def get_lambda_field(self) -> np.ndarray:
        return self._lambda_field.copy()

    @property
    def canonical_tick(self) -> int:
        return self.current_tick

    def update_f_smooth(self):
        """Apply spatial Gaussian smoothing to f field."""
        from scipy.ndimage import gaussian_filter
        sigma = self.lattice.config.spatial_sigma
        if sigma > 0:
            self.lattice.f_smooth = gaussian_filter(self.lattice.f, sigma=sigma, mode='wrap')
        else:
            self.lattice.f_smooth = self.lattice.f.copy()
