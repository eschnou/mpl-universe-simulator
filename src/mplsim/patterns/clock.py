"""
ClockPattern: measures time dilation by tracking proper time.

A clock is placed at a specific lattice location. It "ticks" whenever
its host node realizes an update. By comparing clock ticks to canonical
time, we measure gravitational time dilation.

Key observation:
- Clock near source: fewer ticks (low f, slow proper time)
- Clock far from source: more ticks (high f, fast proper time)
- Ratio of ticks ≈ ratio of f values ≈ time dilation factor
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mplsim.core.lattice import Lattice

from mplsim.patterns.base import Pattern, PatternConfig


@dataclass
class ClockConfig(PatternConfig):
    """Configuration for a clock pattern."""

    pass  # Inherits pattern_id, world_tag, center_x, center_y


class ClockPattern(Pattern):
    """
    A clock that ticks with local proper time.

    The clock:
    1. Lives at a specific lattice position (x, y)
    2. "Ticks" whenever its host node realizes an update
    3. Records (canonical_tick, clock_ticks) readings over time
    4. Allows comparison of tick rates at different positions

    This demonstrates gravitational time dilation:
    - Clocks near sources tick slower (low f region)
    - Clocks far from sources tick faster (high f region)
    """

    def __init__(self, config: ClockConfig, lattice: "Lattice"):
        super().__init__(config, lattice)

        # Track clock ticks (accumulated proper time at this location)
        self.clock_ticks: int = 0

        # Track previous proper time to detect increments
        self._prev_proper_time: int = int(lattice.proper_time[self.y, self.x])

        # Record readings: list of (canonical_tick, clock_ticks)
        self.readings: list[tuple[int, int]] = []

    def update(self, canonical_tick: int) -> None:
        """
        Update clock state for the current tick.

        Check if the host node's proper time increased since last tick.
        If so, increment our clock_ticks counter.
        """
        self._canonical_tick = canonical_tick

        # Check current proper time at our location
        current_proper_time = int(self.lattice.proper_time[self.y, self.x])

        # Did the host node realize an update?
        delta = current_proper_time - self._prev_proper_time
        if delta > 0:
            self.clock_ticks += delta

        self._prev_proper_time = current_proper_time

    def record_reading(self, canonical_tick: int) -> None:
        """Record current (canonical_tick, clock_ticks) pair."""
        self.readings.append((canonical_tick, self.clock_ticks))

    def get_tick_rate(self) -> float:
        """
        Calculate average tick rate (clock_ticks / canonical_ticks).

        Returns:
            Rate in range [0, 1] where 1 = no dilation, <1 = slowed
        """
        if self._canonical_tick == 0:
            return 1.0
        return self.clock_ticks / self._canonical_tick

    def get_measurements(self) -> dict:
        """Return clock measurements."""
        return {
            "pattern_id": self.config.pattern_id,
            "position": (self.x, self.y),
            "clock_ticks": self.clock_ticks,
            "canonical_ticks": self._canonical_tick,
            "tick_rate": self.get_tick_rate(),
            "readings": self.readings.copy(),
        }


def create_clock(
    pattern_id: str,
    x: int,
    y: int,
    lattice: "Lattice",
    world_tag: int = 0,
) -> ClockPattern:
    """
    Convenience factory for creating a clock.

    Args:
        pattern_id: Unique identifier for this clock
        x, y: Position on the lattice
        lattice: The lattice the clock observes
        world_tag: Which world this clock belongs to

    Returns:
        Configured ClockPattern
    """
    config = ClockConfig(
        pattern_id=pattern_id,
        world_tag=world_tag,
        center_x=x,
        center_y=y,
    )
    return ClockPattern(config, lattice)
