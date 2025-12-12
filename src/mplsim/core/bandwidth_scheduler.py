"""
Bandwidth Scheduler: True causal emergence from bandwidth limits.

This scheduler implements proper causal waiting:
1. Global clock ticks uniformly
2. Each node has data to push based on its activity rate
3. Bandwidth limits how fast data can be pushed
4. A node can only update when:
   - All its data has been pushed (bandwidth constraint)
   - All neighbors have sent their messages (sync constraint)

This creates REAL causal propagation without deadlock:
- Heavy nodes take longer to push → their messages arrive late
- Neighbors waiting for those messages get delayed
- The delay cascades outward, creating spatial λ structure

The key insight: we don't block, we compute "earliest possible update time".
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
    """Configuration for the bandwidth scheduler."""

    canonical_interval: int = 100  # Ticks between updates at f=1 (no load)
    bandwidth_per_tick: float = 1.0  # Data units that can be pushed per tick
    propagation_delay: int = 1  # Ticks for message to travel one link
    f_smoothing_alpha: float = 0.1  # EMA smoothing for f field

    # If True, require messages from all neighbors before updating
    sync_required: bool = True

    # Probabilistic message size: L ~ Poisson(activity * message_rate_scale)
    # Keep mean L << link_capacity for weak-congestion regime
    message_rate_scale: float = 10.0  # Scale factor for Poisson mean
    stochastic_messages: bool = True  # If False, use deterministic (mean-field)


@dataclass
class BandwidthScheduler:
    """
    Scheduler with true causal bandwidth and waiting dynamics.

    Each node tracks:
    - When it can next update (based on bandwidth + sync constraints)
    - When it last received a message from each neighbor
    """

    lattice: "Lattice"
    source_map: "SourceMap"
    kernel: "Kernel"
    config: BandwidthSchedulerConfig = field(default_factory=BandwidthSchedulerConfig)

    # Simulation state
    current_tick: int = field(default=0, init=False)
    total_updates: int = field(default=0, init=False)

    # Per-node timing state
    _next_update_tick: np.ndarray = field(default=None, init=False)
    _last_update_tick: np.ndarray = field(default=None, init=False)
    _last_received: np.ndarray = field(default=None, init=False)  # [ny, nx, n_dirs]

    # For λ tracking
    _lambda_field: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        """Initialize timing state."""
        ny, nx = self.lattice.shape
        n_dirs = len(self.lattice.directions)

        # All nodes start ready at tick 0
        self._next_update_tick = np.zeros((ny, nx), dtype=np.int64)
        self._last_update_tick = np.zeros((ny, nx), dtype=np.int64)

        # Bootstrap: all nodes have "received" from all neighbors at tick 0
        self._last_received = np.zeros((ny, nx, n_dirs), dtype=np.int64)

        # Initialize λ field
        self._lambda_field = np.zeros((ny, nx), dtype=np.float64)

        # Initialize f field
        self.lattice.f.fill(1.0)

        # Direction mapping
        self._dir_to_idx = {d: i for i, d in enumerate(self.lattice.directions)}
        self._directions = self.lattice._directions

    def run(self, n_ticks: int) -> dict:
        """
        Run simulation for n ticks.

        Args:
            n_ticks: Number of global ticks to run

        Returns:
            Statistics dictionary
        """
        for _ in range(n_ticks):
            self._tick()

        # Update f_smooth for particle gradient computation
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
        """Execute one global tick."""
        self.current_tick += 1
        ny, nx = self.lattice.shape

        # Find and update all nodes that are ready
        ready_mask = self._next_update_tick <= self.current_tick

        # Process ready nodes
        for y in range(ny):
            for x in range(nx):
                if ready_mask[y, x]:
                    self._update_node(y, x)

    def _update_node(self, y: int, x: int):
        """Update a single node."""
        ny, nx = self.lattice.shape
        config = self.config

        # === COMPUTE MESSAGE SIZE (bandwidth constraint) ===
        # Paper: "higher activity -> higher probability of large delta"
        # L ~ Poisson(activity * scale), then push_ticks = ceil(L / bandwidth)
        rate = self.source_map.rates[y, x]
        if config.stochastic_messages:
            # Probabilistic: draw message size from Poisson distribution
            poisson_mean = rate * config.message_rate_scale
            message_size = np.random.poisson(poisson_mean)
        else:
            # Deterministic mean-field approximation
            message_size = rate * config.message_rate_scale

        push_ticks = int(np.ceil(message_size / config.bandwidth_per_tick))
        push_ticks = max(push_ticks, 1)  # At least 1 tick

        # === SEND MESSAGES TO NEIGHBORS ===
        # Messages arrive after we finish pushing + propagation delay
        message_arrival_tick = self.current_tick + push_ticks + config.propagation_delay

        for direction in self.lattice.directions:
            dy, dx = self._directions[direction]
            neighbor_y = (y + dy) % ny
            neighbor_x = (x + dx) % nx

            # Mark that neighbor received from us (from opposite direction)
            opposite_dir = self._get_opposite_direction(direction)
            opp_idx = self._dir_to_idx[opposite_dir]
            self._last_received[neighbor_y, neighbor_x, opp_idx] = message_arrival_tick

        # === COMPUTE NEXT UPDATE TIME ===
        # 1. Bandwidth constraint: can't update until we've pushed our data
        bandwidth_ready = self.current_tick + push_ticks

        # 2. Sync constraint: can't update until we've received from all neighbors
        if config.sync_required:
            # Need to receive from all neighbors AFTER our current update
            # So we look at when the NEXT message from each neighbor will arrive
            sync_ready = self._last_received[y, x, :].max()
        else:
            sync_ready = 0

        self._next_update_tick[y, x] = max(bandwidth_ready, sync_ready)

        # === COMPUTE λ AND f ===
        last_update = self._last_update_tick[y, x]
        actual_interval = self.current_tick - last_update

        if actual_interval > 0:
            # λ = slowdown factor = (actual - canonical) / canonical
            # This is UNBOUNDED: if actual >> canonical, λ >> 1
            lambda_local = (actual_interval - config.canonical_interval) / config.canonical_interval
            lambda_local = max(0.0, lambda_local)  # Can't be negative

            # f = 1 / (1 + λ), maps λ ∈ [0, ∞) to f ∈ (0, 1]
            local_f = 1.0 / (1.0 + lambda_local)

            # Update with EMA smoothing
            alpha = config.f_smoothing_alpha
            self._lambda_field[y, x] = (1 - alpha) * self._lambda_field[y, x] + alpha * lambda_local
            self.lattice.f[y, x] = (1 - alpha) * self.lattice.f[y, x] + alpha * local_f

        # Update last update time
        self._last_update_tick[y, x] = self.current_tick
        self.total_updates += 1

    def _get_opposite_direction(self, direction: str) -> str:
        """Get opposite direction."""
        opposites = {
            "N": "S", "S": "N", "E": "W", "W": "E",
            "NE": "SW", "SW": "NE", "NW": "SE", "SE": "NW"
        }
        return opposites.get(direction, direction)

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
                self.lattice.f, sigma=sigma, mode='wrap'
            )
        else:
            self.lattice.f_smooth = self.lattice.f.copy()
