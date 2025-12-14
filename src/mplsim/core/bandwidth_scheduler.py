"""
Bandwidth Scheduler: Emergent gravity from bandwidth limits.

Implements the paper's equation: λ(x) = γ·a(x) + β·⟨λ⟩_x

Each tick:
1. Generate message sizes ~ Poisson(activity × scale)
2. Local stall if message_size > capacity
3. Sync contribution = β × observed neighbor λ (from message gaps)
4. Update λ iteratively (relaxation toward steady state)

f(x) = 1 - λ(x) is the update fraction (gravity field).

LOCAL RULE: Instead of "stealing" neighbor λ values directly, we observe
the gap between messages from each neighbor. If a neighbor is slow (high λ),
their messages arrive less frequently → larger gaps → we infer their λ.
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
    link_capacity: float = 10.0      # Max message size per tick
    message_scale: float = 10.0      # message_size ~ Poisson(activity × scale)
    beta: float = 0.9                # Sync coupling: λ includes β·⟨λ⟩_neighbors
    stochastic_messages: bool = True # Poisson (True) or deterministic (False)
    gap_ema_alpha: float = 0.1       # EMA smoothing for gap observations


@dataclass
class MessageArrivalTracker:
    """Tracks when messages arrive from each direction at each node."""

    ny: int
    nx: int
    directions: list[str]
    ema_alpha: float = 0.1

    # EMA of observed gap per direction: [ny, nx]
    ema_gap: dict[str, np.ndarray] = field(default=None, init=False)
    # Last arrival tick per direction
    last_arrival: dict[str, np.ndarray] = field(default=None, init=False)

    def __post_init__(self):
        self.ema_gap = {d: np.ones((self.ny, self.nx), dtype=np.float64) for d in self.directions}
        self.last_arrival = {d: np.zeros((self.ny, self.nx), dtype=np.int64) for d in self.directions}

    def record_arrivals(self, tick: int, sent_masks: dict[str, np.ndarray]):
        """
        Record message arrivals and update gap EMA.

        sent_masks[direction] = bool array where True means sender at (y,x) sent in that direction.
        """
        for send_dir, sent_mask in sent_masks.items():
            recv_dir = OPPOSITE_DIRECTION[send_dir]
            dx, dy = DIRECTION_DELTAS[send_dir]

            # Shift mask to receiver positions
            receiver_got_msg = np.roll(np.roll(sent_mask, dy, axis=0), dx, axis=1)

            # For nodes that received: compute gap and update EMA
            # gap = tick - last_arrival (but only if last_arrival > 0)
            has_prior = self.last_arrival[recv_dir] > 0
            valid = receiver_got_msg & has_prior

            if np.any(valid):
                gap = tick - self.last_arrival[recv_dir]
                # Update EMA only where valid
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

    def get_observed_neighbor_lambda(self) -> np.ndarray:
        """
        Estimate neighbor lambda from observed gaps.

        λ_observed = (gap - 1) / gap

        If gap=1, λ=0 (neighbor sends every tick)
        If gap=2, λ=0.5 (neighbor sends every other tick)
        If gap→∞, λ→1 (neighbor stalled)
        """
        lambda_sum = np.zeros((self.ny, self.nx), dtype=np.float64)
        for d in self.directions:
            gap = np.maximum(self.ema_gap[d], 1.0)
            lambda_sum += (gap - 1.0) / gap

        return lambda_sum / len(self.directions)


@dataclass
class BandwidthScheduler:
    """Scheduler implementing λ = local_stall + β·⟨λ⟩_observed iteratively.

    Uses LOCAL RULE: neighbor λ is inferred from message arrival gaps,
    not read directly from neighbor state.
    """

    lattice: "Lattice"
    source_map: "SourceMap"
    kernel: "Kernel"
    config: BandwidthSchedulerConfig = field(default_factory=BandwidthSchedulerConfig)

    current_tick: int = field(default=0, init=False)
    total_updates: int = field(default=0, init=False)
    _lambda_field: np.ndarray = field(default=None, init=False)
    _tracker: MessageArrivalTracker = field(default=None, init=False)

    def __post_init__(self):
        ny, nx = self.lattice.shape
        self._lambda_field = np.zeros((ny, nx), dtype=np.float64)
        self.lattice.f.fill(1.0)
        self._tracker = MessageArrivalTracker(
            ny, nx,
            self.lattice.directions,
            ema_alpha=self.config.gap_ema_alpha
        )

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

        Update rule:
            1. Determine which nodes send (probability = f = 1 - λ)
            2. Record message arrivals → update gap observations
            3. Compute observed neighbor λ from gaps
            4. λ_new = local_stall + β × observed_neighbor_avg
        """
        self.current_tick += 1
        cfg = self.config

        # Local stall: message_size > capacity
        if cfg.stochastic_messages:
            msg_sizes = np.random.poisson(self.source_map.rates * cfg.message_scale)
        else:
            msg_sizes = self.source_map.rates * cfg.message_scale
        local_stall = (msg_sizes > cfg.link_capacity).astype(np.float64)

        # Determine which nodes send (probability = f = 1 - λ)
        f_field = 1.0 - self._lambda_field
        sends = np.random.random(f_field.shape) < f_field

        # Create sent_masks per direction
        sent_masks = {d: sends.copy() for d in self.lattice.directions}

        # Record arrivals → update gap observations
        self._tracker.record_arrivals(self.current_tick, sent_masks)

        # Get observed neighbor λ from gaps (LOCAL RULE)
        neighbor_avg = self._tracker.get_observed_neighbor_lambda()

        # Update λ (relaxation toward steady state)
        self._lambda_field = np.clip(local_stall + cfg.beta * neighbor_avg, 0.0, 1.0)
        self.lattice.f = 1.0 - self._lambda_field

        self.total_updates += int(np.sum(1.0 - local_stall))

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
