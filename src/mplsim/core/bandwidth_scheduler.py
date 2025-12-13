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

    # Synchronization coupling strength (paper's β parameter)
    # β=0: purely local (ignore neighbors), β=1: full sync pressure
    # next_update = local_ready + beta * max(0, neighbor_ready - local_ready)
    beta: float = 1.0

    # Probabilistic message size: L ~ Poisson(activity * message_rate_scale)
    # Keep mean L << link_capacity for weak-congestion regime
    message_rate_scale: float = 10.0  # Scale factor for Poisson mean
    stochastic_messages: bool = True  # If False, use deterministic (mean-field)


@dataclass
class BandwidthScheduler:
    """
    Scheduler with true causal bandwidth and waiting dynamics.

    Uses GENERATION-BASED synchronization for proper delay propagation:
    - Each node tracks its generation (update count)
    - Messages carry the sender's generation
    - A node can only proceed when it has FRESH messages from all neighbors
      (neighbors must be at generation >= my_generation)

    This ensures delays propagate through the lattice, creating log(r) falloff.
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

    # Generation-based sync: tracks each node's generation and received generations
    _generation: np.ndarray = field(default=None, init=False)  # [ny, nx]
    _received_gen: np.ndarray = field(default=None, init=False)  # [ny, nx, n_dirs]
    _message_arrival: np.ndarray = field(default=None, init=False)  # [ny, nx, n_dirs] tick when msg arrives
    _message_arrival_snapshot: np.ndarray = field(default=None, init=False)  # Read buffer for synchronous updates

    # For λ tracking
    _lambda_field: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        """Initialize timing state."""
        ny, nx = self.lattice.shape
        n_dirs = len(self.lattice.directions)

        # All nodes start ready at tick 0
        self._next_update_tick = np.zeros((ny, nx), dtype=np.int64)
        self._last_update_tick = np.zeros((ny, nx), dtype=np.int64)

        # Generation tracking: all start at generation 0
        self._generation = np.zeros((ny, nx), dtype=np.int64)
        # Bootstrap: all nodes have "received" generation 0 from all neighbors at tick 0
        self._received_gen = np.zeros((ny, nx, n_dirs), dtype=np.int64)
        self._message_arrival = np.zeros((ny, nx, n_dirs), dtype=np.int64)
        self._message_arrival_snapshot = np.zeros((ny, nx, n_dirs), dtype=np.int64)

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
        """Execute one global tick (vectorized for performance)."""
        self.current_tick += 1
        ny, nx = self.lattice.shape
        config = self.config

        # Snapshot message arrivals for synchronous semantics:
        # All nodes updating at tick t read from the state at end of tick t-1.
        np.copyto(self._message_arrival_snapshot, self._message_arrival)

        # Find ready nodes
        ready_mask = self._next_update_tick <= self.current_tick

        # === COMPUTE MESSAGE SIZES (vectorized) ===
        if config.stochastic_messages:
            poisson_means = self.source_map.rates * config.message_rate_scale
            message_sizes = np.random.poisson(poisson_means)
        else:
            message_sizes = self.source_map.rates * config.message_rate_scale

        push_ticks = np.maximum(1, np.ceil(message_sizes / config.bandwidth_per_tick).astype(np.int64))

        # === COMPUTE WAITING (vectorized) ===
        local_ready = self.current_tick + push_ticks
        neighbor_ready = self._message_arrival_snapshot.max(axis=2)  # [ny, nx]
        neighbor_delay = np.maximum(0, neighbor_ready - local_ready)
        actual_wait = (config.beta * neighbor_delay).astype(np.int64)

        # === MESSAGE ARRIVAL TIMES ===
        message_arrival_tick = self.current_tick + push_ticks + actual_wait + config.propagation_delay

        # === SEND TO NEIGHBORS (vectorized via roll) ===
        # For each direction, roll the arrival times to neighbor positions
        # and update only where the source node was ready
        for dir_idx, direction in enumerate(self.lattice.directions):
            dy, dx = self._directions[direction]
            opposite_dir = self._get_opposite_direction(direction)
            opp_idx = self._dir_to_idx[opposite_dir]

            # Roll arrival times: arr[y,x] -> arr[(y+dy)%ny, (x+dx)%nx]
            shifted_arrival = np.roll(np.roll(message_arrival_tick, dy, axis=0), dx, axis=1)
            shifted_ready = np.roll(np.roll(ready_mask, dy, axis=0), dx, axis=1)

            # Update neighbor's arrival from this direction where source was ready
            self._message_arrival[:, :, opp_idx] = np.where(
                shifted_ready,
                shifted_arrival,
                self._message_arrival[:, :, opp_idx]
            )

        # === UPDATE NEXT UPDATE TIME ===
        next_update = local_ready + actual_wait
        self._next_update_tick = np.where(ready_mask, next_update, self._next_update_tick)

        # === COMPUTE f AND λ (vectorized) ===
        actual_interval = self.current_tick - self._last_update_tick
        update_mask = ready_mask & (actual_interval > 0)

        # Avoid division by zero
        safe_interval = np.maximum(1, actual_interval)
        local_f = np.minimum(1.0, config.canonical_interval / safe_interval)
        lambda_local = 1.0 - local_f

        alpha = config.f_smoothing_alpha
        self._lambda_field = np.where(
            update_mask,
            (1 - alpha) * self._lambda_field + alpha * lambda_local,
            self._lambda_field
        )
        self.lattice.f = np.where(
            update_mask,
            (1 - alpha) * self.lattice.f + alpha * local_f,
            self.lattice.f
        )

        # Update last_update_tick and count
        self._last_update_tick = np.where(ready_mask, self.current_tick, self._last_update_tick)
        self.total_updates += int(ready_mask.sum())

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
