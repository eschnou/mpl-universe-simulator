"""
Base classes for patterns.

Patterns are persistent structures placed on the lattice that can:
- Track their own proper time (clocks)
- Move according to dynamics (particles)
- Propagate across the grid (light rays)

IMPORTANT: Patterns observe the engine state but don't modify the core physics.
They're diagnostic tools for measuring time dilation, free fall, lensing, etc.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mplsim.core.lattice import Lattice


@dataclass
class PatternConfig:
    """Base configuration for patterns."""

    pattern_id: str  # Unique identifier
    world_tag: int = 0  # Which world this pattern belongs to
    center_x: int = 0  # Initial x position
    center_y: int = 0  # Initial y position


class Pattern(ABC):
    """
    Base class for patterns that live on the lattice.

    Patterns observe the lattice state and accumulate measurements.
    They experience f(x) directly - they don't know about ϕ or Poisson.
    """

    def __init__(self, config: PatternConfig, lattice: "Lattice"):
        self.config = config
        self.lattice = lattice
        self.x = config.center_x
        self.y = config.center_y
        self._canonical_tick = 0

    @property
    def center(self) -> tuple[int, int]:
        """Current center position (x, y)."""
        return self.x, self.y

    def get_center(self) -> tuple[int, int]:
        """Get current center position."""
        return self.center

    @abstractmethod
    def update(self, canonical_tick: int) -> None:
        """
        Update the pattern state for the current tick.

        Called once per canonical tick. The pattern should:
        - Check what happened to its host node(s)
        - Update its internal state accordingly
        - Record any measurements

        Args:
            canonical_tick: The current canonical tick number
        """
        ...

    @abstractmethod
    def get_measurements(self) -> dict:
        """
        Return recorded measurements from this pattern.

        Returns:
            Dict with pattern-specific measurements
        """
        ...
