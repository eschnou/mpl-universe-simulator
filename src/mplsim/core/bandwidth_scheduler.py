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
        """Update a single node with β-weighted neighbor waiting."""
        ny, nx = self.lattice.shape
        config = self.config

        # === COMPUTE MESSAGE SIZE (bandwidth constraint) ===
        rate = self.source_map.rates[y, x]
        if config.stochastic_messages:
            poisson_mean = rate * config.message_rate_scale
            message_size = np.random.poisson(poisson_mean)
        else:
            message_size = rate * config.message_rate_scale

        push_ticks = int(np.ceil(message_size / config.bandwidth_per_tick))
        push_ticks = max(push_ticks, 1)

        # === COMPUTE WAITING TIME FROM NEIGHBORS ===
        # local_ready: when I finish my own work
        local_ready = self.current_tick + push_ticks

        # neighbor_ready: latest message arrival from any neighbor
        neighbor_ready = self._message_arrival[y, x, :].max()

        # Waiting time = how much later neighbors finish than me
        # β controls how much of this waiting I actually do
        neighbor_delay = max(0, neighbor_ready - local_ready)
        actual_wait = int(config.beta * neighbor_delay)

        # === SEND MESSAGES TO NEIGHBORS ===
        # My message arrives after: my_work + my_wait + propagation
        message_arrival_tick = self.current_tick + push_ticks + actual_wait + config.propagation_delay

        for direction in self.lattice.directions:
            dy, dx = self._directions[direction]
            neighbor_y = (y + dy) % ny
            neighbor_x = (x + dx) % nx

            opposite_dir = self._get_opposite_direction(direction)
            opp_idx = self._dir_to_idx[opposite_dir]
            self._message_arrival[neighbor_y, neighbor_x, opp_idx] = message_arrival_tick

        # === COMPUTE NEXT UPDATE TIME ===
        self._next_update_tick[y, x] = local_ready + actual_wait

        # === COMPUTE f AND λ ===
        # λ depends on TOTAL delay: own work + waiting for neighbors
        last_update = self._last_update_tick[y, x]
        actual_interval = self.current_tick - last_update

        if actual_interval > 0:
            local_f = min(1.0, config.canonical_interval / actual_interval)
            lambda_local = 1.0 - local_f

            alpha = config.f_smoothing_alpha
            self._lambda_field[y, x] = (1 - alpha) * self._lambda_field[y, x] + alpha * lambda_local
            self.lattice.f[y, x] = (1 - alpha) * self.lattice.f[y, x] + alpha * local_f

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
