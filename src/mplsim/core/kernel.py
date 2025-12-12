"""
Kernels define local update logic for each node.

The key insight: for gravity experiments, the kernel's job is simply to
generate configurable message traffic. Congestion from that traffic is
what creates f(x) < 1, which IS gravity.

PRIMITIVES vs DERIVED:
- Kernel outputs: state updates + outgoing messages (primitives)
- These drive congestion → f(x) → time dilation
- ϕ(x) is derived later in the analysis layer
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mplsim.core.source_map import SourceMap


class Kernel(Protocol):
    """Protocol for local update kernels."""

    def compute_output_bits(
        self,
        y: int,
        x: int,
        source_map: "SourceMap",
    ) -> float:
        """
        Compute total output bits this node wants to send.

        Args:
            y, x: Node position
            source_map: Defines output rates per node

        Returns:
            Total bits this node wants to send to neighbors
        """
        ...

    @property
    def required_inputs(self) -> set[str]:
        """
        Which neighbor directions must have delivered messages before update.

        For LoadGeneratorKernel, this is empty: nodes don't wait for inputs.
        Congestion comes purely from output queue saturation.
        """
        ...


@dataclass
class LoadGeneratorKernel:
    """
    Minimal kernel that generates configurable message traffic.

    NOT physically realistic — just creates the load patterns needed to
    demonstrate bandwidth-limited gravity.

    Usage:
    - SourceMap marks some nodes as "sources" with high output_rate
    - Background nodes use low output_rate
    - Observe f(x) drop near sources due to link saturation

    The kernel doesn't really "compute" anything interesting.
    It just determines how many bits each node wants to send per tick.

    Synchronization:
    - When sync_required=True, nodes must wait for inputs from neighbors
    - This creates the β·⟨λ⟩_x synchronization pressure from the paper
    - Slow neighbors → we wait → our f drops → gravity emerges from waiting
    """

    message_size: float = 1.0  # Bits per unit of rate
    sync_required: bool = True  # Wait for neighbor inputs before updating
    _directions: set[str] | None = None  # Will be set by scheduler

    def compute_output_bits(
        self,
        y: int,
        x: int,
        source_map: "SourceMap",
    ) -> float:
        """
        Compute total output bits this node wants to send.

        Each node sends to all 4 neighbors (von Neumann) or 8 (Moore).
        The bits per direction = rate * message_size / n_directions.
        For simplicity, we return total bits and let the scheduler split.
        """
        rate = source_map.get_rate(x, y)
        return rate * self.message_size

    @property
    def required_inputs(self) -> set[str]:
        """
        Return directions from which inputs are required before updating.

        When sync_required=True, returns all neighbor directions.
        This creates synchronization pressure: slow neighbors slow us down.
        """
        if self.sync_required and self._directions is not None:
            return self._directions
        return set()

    def set_directions(self, directions: list[str]):
        """Set the neighbor directions for this kernel (called by scheduler)."""
        self._directions = set(directions)


def create_default_kernel(sync_required: bool = True) -> LoadGeneratorKernel:
    """
    Factory for the default kernel.

    Args:
        sync_required: If True, nodes wait for neighbor inputs (creates β coupling).
                       Set to False to disable synchronization pressure.
    """
    return LoadGeneratorKernel(message_size=1.0, sync_required=sync_required)
