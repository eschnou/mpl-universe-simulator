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
- The steady-state satisfies: λ ≈ γ·a + β·⟨λ⟩ (paper's equation)
- This implies screened Poisson: (L + m²)λ ≈ κρ
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
    damping: float = 0.9             # Sync damping (< 1 to avoid runaway)
    data_scale: float = 1.0          # data_size ~ Poisson(activity × scale)
    base_interval: float = 1.0       # Baseline ticks per update (f=1 baseline)
    stochastic: bool = True          # Poisson (True) or deterministic (False)
    gap_ema_alpha: float = 0.1       # EMA smoothing for gap observations


@dataclass
class GapTracker:
    """Tracks message arrival gaps from each direction at each node."""

    ny: int
    nx: int
    directions: list[str]
    ema_alpha: float = 0.1
    initial_gap: float = 1.0

    # EMA of observed gap per direction: [ny, nx]
    ema_gap: dict[str, np.ndarray] = field(default=None, init=False)
    # Last arrival tick per direction
    last_arrival: dict[str, np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        self.ema_gap = {
            d: np.full((self.ny, self.nx), self.initial_gap, dtype=np.float64)
            for d in self.directions
        }
        self.last_arrival = {
            d: np.zeros((self.ny, self.nx), dtype=np.int64)
            for d in self.directions
        }

    def record_arrivals(self, tick: int, sent_masks: dict[str, np.ndarray]):
        """Record message arrivals and update gap EMA."""
        for send_dir, sent_mask in sent_masks.items():
            recv_dir = OPPOSITE_DIRECTION[send_dir]
            dx, dy = DIRECTION_DELTAS[send_dir]

            # Shift mask to receiver positions
            receiver_got_msg = np.roll(np.roll(sent_mask, dy, axis=0), dx, axis=1)

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
        """Get average gap observed from all neighbors."""
        total = np.zeros((self.ny, self.nx), dtype=np.float64)
        for d in self.directions:
            total += self.ema_gap[d]
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
            ny, nx,
            self.lattice.directions,
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

        # Local time from data generation
        if cfg.stochastic:
            data_sizes = np.random.poisson(self.source_map.rates * cfg.data_scale)
        else:
            data_sizes = self.source_map.rates * cfg.data_scale
        local_time = np.maximum(cfg.base_interval, cfg.base_interval * data_sizes / cfg.bandwidth)

        # Sync time from observed neighbor gaps
        avg_gap = self._tracker.get_avg_neighbor_gap()
        sync_time = avg_gap * cfg.damping

        # Send interval: max of local and sync constraints
        self._send_interval = np.maximum(local_time, sync_time)

        # Send if ready
        sends = tick >= self._next_send_tick
        self._next_send_tick = np.where(sends, tick + self._send_interval, self._next_send_tick)

        # Record arrivals for gap tracking
        sent_masks = {d: sends.copy() for d in self.lattice.directions}
        self._tracker.record_arrivals(tick, sent_masks)

        # Update f field
        self.lattice.f = np.clip(cfg.base_interval / self._send_interval, 0.0, 1.0)
        self.total_updates += int(np.sum(sends))

    def _update_f_smooth(self):
        """Apply spatial Gaussian smoothing to f field."""
        from scipy.ndimage import gaussian_filter
        sigma = self.lattice.config.spatial_sigma
        if sigma > 0:
            self.lattice.f_smooth = gaussian_filter(self.lattice.f, sigma=sigma, mode='wrap')
        else:
            self.lattice.f_smooth = self.lattice.f.copy()

    @property
    def canonical_tick(self) -> int:
        return self.current_tick
