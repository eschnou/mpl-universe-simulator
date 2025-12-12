"""
Lattice: the 2D grid of nodes that forms the universe engine.

The lattice stores ONLY engine primitives:
- Node state (small vector per node)
- Proper time τ(x) (accumulated realized updates)
- Update statistics (realized/possible counts, f(x))

It does NOT store ϕ, ρ_act, or any derived quantities.
"""

from dataclasses import dataclass, field
from typing import Literal, Iterator
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class LatticeConfig:
    """Configuration for a 2D lattice."""

    nx: int  # Grid width
    ny: int  # Grid height
    neighborhood: Literal["von_neumann", "moore"] = "von_neumann"  # 4 or 8 neighbors
    boundary: Literal["periodic", "reflective", "absorbing"] = "periodic"
    link_capacity: float = 10.0  # Default bits per link per tick
    n_channels: int = 4  # State channels per node

    # Spatial smoothing for f_smooth (coarse-graining scale)
    # This represents the paper's "long-time, spatially-averaged f(x)" for computing
    # geodesics in the continuum limit. Small sigma (1-2) for minimal coarse-graining.
    # The synchronization pressure creates the gross spatial pattern; smoothing just
    # stabilizes gradient computation for particle dynamics.
    spatial_sigma: float = 1.0  # Minimal coarse-graining for stable gradients


# Direction vectors for neighbor lookup
DIRECTIONS_VON_NEUMANN = {
    "N": (0, -1),   # North: y decreases
    "S": (0, 1),    # South: y increases
    "E": (1, 0),    # East: x increases
    "W": (-1, 0),   # West: x decreases
}

DIRECTIONS_MOORE = {
    **DIRECTIONS_VON_NEUMANN,
    "NE": (1, -1),
    "NW": (-1, -1),
    "SE": (1, 1),
    "SW": (-1, 1),
}

OPPOSITE_DIRECTION = {
    "N": "S", "S": "N", "E": "W", "W": "E",
    "NE": "SW", "NW": "SE", "SE": "NW", "SW": "NE",
}


class Lattice:
    """
    The universe engine's state.

    IMPORTANT: This class contains ONLY engine primitives.
    ϕ(x), ρ_act(x), and other derived quantities live in the analysis layer.
    """

    def __init__(self, config: LatticeConfig):
        self.config = config
        ny, nx = config.ny, config.nx
        nc = config.n_channels

        # ═══════════════════════════════════════════════════════════════
        # PRIMITIVE STATE (what the engine actually tracks)
        # ═══════════════════════════════════════════════════════════════

        # Node state: [ny, nx, n_channels]
        self.state = np.zeros((ny, nx, nc), dtype=np.float64)

        # Proper time: THE fundamental clock — increments on realized updates
        self.proper_time = np.zeros((ny, nx), dtype=np.int64)

        # Update statistics (primitives from which f(x) is computed)
        self.realized_updates = np.zeros((ny, nx), dtype=np.int64)
        self.possible_updates = np.zeros((ny, nx), dtype=np.int64)

        # Update fraction f(x) = realized / possible
        # This IS the "gravity" — not derived from anything else
        self.f = np.ones((ny, nx), dtype=np.float64)  # Start at 1 (no slowdown)

        # Spatially smoothed f for physics (trajectories, gradients, ϕ)
        # This is the coarse-grained field that appears in the continuum limit
        self.f_smooth = np.ones((ny, nx), dtype=np.float64)

        # Direction info
        if config.neighborhood == "von_neumann":
            self._directions = DIRECTIONS_VON_NEUMANN
        else:
            self._directions = DIRECTIONS_MOORE

    @property
    def directions(self) -> list[str]:
        """List of direction names for this lattice's neighborhood type."""
        return list(self._directions.keys())

    @property
    def shape(self) -> tuple[int, int]:
        """Return (ny, nx) grid dimensions."""
        return self.config.ny, self.config.nx

    def get_neighbor_coords(
        self, x: int, y: int, direction: str
    ) -> tuple[int, int] | None:
        """
        Get coordinates of neighbor in given direction.

        Returns None if neighbor is out of bounds (for non-periodic boundaries).
        """
        dx, dy = self._directions[direction]
        nx, ny = self.config.nx, self.config.ny

        new_x = x + dx
        new_y = y + dy

        if self.config.boundary == "periodic":
            return new_x % nx, new_y % ny

        elif self.config.boundary == "reflective":
            # Clamp to boundaries
            return max(0, min(nx - 1, new_x)), max(0, min(ny - 1, new_y))

        else:  # absorbing
            if 0 <= new_x < nx and 0 <= new_y < ny:
                return new_x, new_y
            return None

    def get_all_neighbors(self, x: int, y: int) -> dict[str, tuple[int, int] | None]:
        """Get all neighbor coordinates for a node."""
        return {
            direction: self.get_neighbor_coords(x, y, direction)
            for direction in self._directions
        }

    def iter_nodes(self) -> Iterator[tuple[int, int]]:
        """Iterate over all (x, y) node coordinates."""
        for y in range(self.config.ny):
            for x in range(self.config.nx):
                yield x, y

    def update_f(self, smoothing_window: int = 100, spatial_sigma: float | None = None):
        """
        Update f fields from realized/possible counts.

        Two fields are maintained:
        - f: temporal EMA only (raw physics, engine diagnostics)
        - f_smooth: temporal + spatial smoothing (for trajectories, gradients, ϕ)

        The spatial smoothing represents coarse-graining: in the paper, we use
        the long-time, spatially-averaged f(x) for computing geodesics and ϕ.

        Args:
            smoothing_window: EMA window for temporal smoothing
            spatial_sigma: Gaussian sigma for spatial smoothing (uses config default if None)
        """
        if spatial_sigma is None:
            spatial_sigma = self.config.spatial_sigma

        # Step 1: Compute raw f from discrete counts
        with np.errstate(divide='ignore', invalid='ignore'):
            f_raw = np.where(
                self.possible_updates > 0,
                self.realized_updates / self.possible_updates,
                1.0
            )

        # Step 2: Temporal smoothing (EMA) → f
        alpha = 2.0 / (smoothing_window + 1)
        self.f = alpha * f_raw + (1 - alpha) * self.f

        # Step 3: Spatial smoothing → f_smooth
        # This creates continuous gradients for physics
        if spatial_sigma > 0:
            self.f_smooth = gaussian_filter(
                self.f,
                sigma=spatial_sigma,
                mode='wrap'  # Periodic boundary
            )
        else:
            np.copyto(self.f_smooth, self.f)

    def reset_statistics(self):
        """Reset update counters (for starting a new measurement window)."""
        self.realized_updates.fill(0)
        self.possible_updates.fill(0)
