"""
Bandwidth Scheduler: Emergent gravity from bandwidth limits.

CORE MECHANISM (purely local):
    send_interval = max(local_time, avg_neighbor_gap × damping)
    f = base_interval / send_interval

Where:
- local_time = base_interval × data_size / bandwidth
- avg_neighbor_gap = EMA of observed gaps between messages from neighbors
- damping < 1 prevents runaway synchronization

Each node observes when messages arrive from neighbors. Slow neighbors
have larger gaps between messages. This creates sync pressure without
reading any neighbor state directly.

DERIVED QUANTITIES (for analysis, not used in logic):
- λ(x) = 1 - f(x) is the "slowness" field
- The steady-state satisfies: λ ≈ γ·a + α·⟨λ⟩ (paper's Eq. A11)
- With α < 1: screened Poisson (short-range Yukawa)
- With α = 1: pure Poisson (long-range Newtonian)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mplsim.core.lattice import Lattice
    from mplsim.core.source_map import SourceMap
    from mplsim.core.kernel import Kernel

from mplsim.core.messages import DIRECTION_DELTAS, OPPOSITE_DIRECTION


@dataclass
class BandwidthSchedulerConfig:
    """Configuration for bandwidth scheduler."""
    bandwidth: float = 1.0           # Data units per base_interval
    damping: float = 0.9             # Coupling α: 1.0 = long-range, <1 = screened
    data_scale: float = 1.0          # data_size = activity × scale
    base_interval: float = 1.0       # Baseline ticks per update (f=1 baseline)
    gap_ema_alpha: float = 0.1       # EMA smoothing for gap observations


@dataclass
class GapTracker:
    """Tracks message arrival gaps from each direction at each node.

    For absorbing boundaries, edge nodes have missing neighbors. We handle this
    by using initial_gap (= base_interval) for those directions, which provides
    the "tension" that pins edge gaps toward f=1.
    """

    lattice: "Lattice"
    directions: list[str]
    ema_alpha: float = 0.1
    initial_gap: float = 1.0

    # EMA of observed gap per direction: [ny, nx]
    ema_gap: dict[str, np.ndarray] = field(default=None, init=False)
    # Last arrival tick per direction
    last_arrival: dict[str, np.ndarray] = field(default=None, init=False)
    # Mask of valid neighbors per direction (for absorbing boundaries)
    _valid_neighbor: dict[str, np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        ny, nx = self.lattice.shape
        self.ema_gap = {
            d: np.full((ny, nx), self.initial_gap, dtype=np.float64)
            for d in self.directions
        }
        self.last_arrival = {
            d: np.zeros((ny, nx), dtype=np.int64)
            for d in self.directions
        }
        # Pre-compute valid neighbor masks for absorbing boundaries
        self._valid_neighbor = self._compute_valid_neighbor_masks()

    def _compute_valid_neighbor_masks(self) -> dict[str, np.ndarray]:
        """Compute masks indicating which nodes have valid neighbors in each direction."""
        ny, nx = self.lattice.shape
        boundary = self.lattice.config.boundary

        if boundary == "periodic":
            # All neighbors valid for periodic
            return {d: np.ones((ny, nx), dtype=bool) for d in self.directions}

        # For absorbing/reflective, edges have missing neighbors
        masks = {}
        for d in self.directions:
            mask = np.ones((ny, nx), dtype=bool)
            if boundary == "absorbing":
                dx, dy = DIRECTION_DELTAS[d]
                # Direction points TO the neighbor, so neighbor is at (x+dx, y+dy)
                # Missing neighbor means (x+dx, y+dy) is out of bounds
                if dy == -1:  # North neighbor: top row has no neighbor
                    mask[0, :] = False
                elif dy == 1:  # South neighbor: bottom row has no neighbor
                    mask[ny - 1, :] = False
                if dx == -1:  # West neighbor: left column has no neighbor
                    mask[:, 0] = False
                elif dx == 1:  # East neighbor: right column has no neighbor
                    mask[:, nx - 1] = False
            masks[d] = mask
        return masks

    def record_arrivals(self, tick: int, sent_masks: dict[str, np.ndarray]):
        """Record message arrivals and update gap EMA."""
        for send_dir, sent_mask in sent_masks.items():
            recv_dir = OPPOSITE_DIRECTION[send_dir]
            dx, dy = DIRECTION_DELTAS[send_dir]

            # Shift mask to receiver positions using boundary-aware shift
            # For absorbing: messages from edges don't wrap to opposite side
            receiver_got_msg = self.lattice.shift_field(
                sent_mask.astype(np.float64), dx, dy, fill_value=0.0
            ).astype(bool)

            # For nodes that received: compute gap and update EMA
            has_prior = self.last_arrival[recv_dir] > 0
            valid = receiver_got_msg & has_prior

            if np.any(valid):
                gap = tick - self.last_arrival[recv_dir]
                self.ema_gap[recv_dir] = np.where(
                    valid,
                    self.ema_alpha * gap + (1 - self.ema_alpha) * self.ema_gap[recv_dir],
                    self.ema_gap[recv_dir]
                )

            # Update last arrival for all receivers
            self.last_arrival[recv_dir] = np.where(
                receiver_got_msg,
                tick,
                self.last_arrival[recv_dir]
            )

    def get_avg_neighbor_gap(self) -> np.ndarray:
        """Get average gap observed from all neighbors.

        For absorbing boundaries, edge nodes use initial_gap for missing neighbors.
        This provides the "tension" that pins boundaries toward f=1.
        """
        ny, nx = self.lattice.shape
        total = np.zeros((ny, nx), dtype=np.float64)

        for d in self.directions:
            # Use actual gap where neighbor exists, initial_gap where it doesn't
            valid = self._valid_neighbor[d]
            total += np.where(valid, self.ema_gap[d], self.initial_gap)

        return total / len(self.directions)


@dataclass
class BandwidthScheduler:
    """Scheduler using purely local bandwidth and sync rules.

    Core rule:
        send_interval = max(local_time, avg_neighbor_gap × damping)
        f = base_interval / send_interval
    """

    lattice: "Lattice"
    source_map: "SourceMap"
    kernel: "Kernel"
    config: BandwidthSchedulerConfig = field(default_factory=BandwidthSchedulerConfig)

    current_tick: int = field(default=0, init=False)
    total_updates: int = field(default=0, init=False)
    _send_interval: np.ndarray = field(default=None, init=False)
    _next_send_tick: np.ndarray = field(default=None, init=False)
    _tracker: GapTracker = field(default=None, init=False)

    def __post_init__(self):
        ny, nx = self.lattice.shape
        base = self.config.base_interval
        self._send_interval = np.full((ny, nx), base, dtype=np.float64)
        self._next_send_tick = np.full((ny, nx), base, dtype=np.float64)
        self.lattice.f.fill(1.0)
        self._tracker = GapTracker(
            lattice=self.lattice,
            directions=self.lattice.directions,
            ema_alpha=self.config.gap_ema_alpha,
            initial_gap=base
        )

    def run(self, n_ticks: int) -> dict:
        """Run simulation for n_ticks."""
        for _ in range(n_ticks):
            self._tick()
        self._update_f_smooth()

        return {
            "n_ticks": n_ticks,
            "current_tick": self.current_tick,
            "total_updates": self.total_updates,
            "mean_f": float(self.lattice.f.mean()),
            "min_f": float(self.lattice.f.min()),
        }

    def _tick(self):
        """One tick: compute send_interval, send if ready, update f."""
        self.current_tick += 1
        cfg = self.config
        tick = self.current_tick

        # Local time from data generation: data_size = activity × scale
        data_sizes = self.source_map.rates * cfg.data_scale
        local_time = np.maximum(cfg.base_interval, cfg.base_interval * data_sizes / cfg.bandwidth)

        # Sync time from observed neighbor gaps
        # Paper: send_interval = max(local_time, avg_neighbor_gap × damping)
        avg_gap = self._tracker.get_avg_neighbor_gap()
        sync_time = avg_gap * cfg.damping

        # Send interval: max of local and sync constraints
        self._send_interval = np.maximum(local_time, sync_time)

        # Enforce boundary condition on send_interval BEFORE using it
        # This ensures boundary nodes actually send at base rate
        if self.lattice.config.boundary == "absorbing":
            self._send_interval[0, :] = cfg.base_interval   # Top edge
            self._send_interval[-1, :] = cfg.base_interval  # Bottom edge
            self._send_interval[:, 0] = cfg.base_interval   # Left edge
            self._send_interval[:, -1] = cfg.base_interval  # Right edge

        # Send if ready
        sends = tick >= self._next_send_tick
        self._next_send_tick = np.where(sends, tick + self._send_interval, self._next_send_tick)

        # Record arrivals for gap tracking
        sent_masks = {d: sends.copy() for d in self.lattice.directions}
        self._tracker.record_arrivals(tick, sent_masks)

        # Update f field
        self.lattice.f = np.clip(cfg.base_interval / self._send_interval, 0.0, 1.0)

        # Enforce boundary condition: f=1 at boundaries for absorbing BC
        # (send_interval already enforced above before computing next_send_tick)
        if self.lattice.config.boundary == "absorbing":
            self.lattice.f[0, :] = 1.0   # Top edge
            self.lattice.f[-1, :] = 1.0  # Bottom edge
            self.lattice.f[:, 0] = 1.0   # Left edge
            self.lattice.f[:, -1] = 1.0  # Right edge

        self.total_updates += int(np.sum(sends))

    def _update_f_smooth(self):
        """Apply spatial Gaussian smoothing to f field."""
        from scipy.ndimage import gaussian_filter
        sigma = self.lattice.config.spatial_sigma
        if sigma > 0:
            self.lattice.f_smooth = gaussian_filter(
                self.lattice.f, sigma=sigma, mode=self.lattice.get_scipy_mode()
            )
        else:
            self.lattice.f_smooth = self.lattice.f.copy()

    @property
    def canonical_tick(self) -> int:
        return self.current_tick
