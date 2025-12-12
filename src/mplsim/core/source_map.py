"""
SourceMap: defines which nodes are "sources" (high message output).

This is how we create "mass" in the engine:
- Place a Gaussian blob of high output_rate somewhere
- That region generates lots of messages
- Links saturate → queues build up → neighbors wait → f(x) drops
- Time dilation emerges around the "mass"

SourceMap is also the single source of truth for ρ in analysis:
- ρ_engine = sum of all world SourceMaps
- ρ_visible = SourceMap of visible world only
"""

from __future__ import annotations
import numpy as np


class SourceMap:
    """
    Defines output rates (message traffic) per node.

    High rate = "massive" region = will create congestion = will slow f(x).
    """

    def __init__(
        self,
        ny: int,
        nx: int,
        background_rate: float = 0.1,
    ):
        """
        Create a source map.

        Args:
            ny, nx: Grid dimensions
            background_rate: Default output rate for all nodes
        """
        self.ny = ny
        self.nx = nx
        self.rates = np.full((ny, nx), background_rate, dtype=np.float64)

    def add_point_source(self, x: int, y: int, rate: float):
        """Add a point source at (x, y) with given output rate."""
        if 0 <= x < self.nx and 0 <= y < self.ny:
            self.rates[y, x] = rate

    def add_gaussian_source(
        self,
        cx: int,
        cy: int,
        peak_rate: float,
        sigma: float,
    ):
        """
        Add a Gaussian blob of activity centered at (cx, cy).

        Args:
            cx, cy: Center coordinates
            peak_rate: Output rate at the peak
            sigma: Standard deviation (spread)
        """
        yy, xx = np.ogrid[:self.ny, :self.nx]
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        gaussian = peak_rate * np.exp(-dist_sq / (2 * sigma ** 2))
        self.rates += gaussian

    def add_ring_source(
        self,
        cx: int,
        cy: int,
        radius: float,
        width: float,
        rate: float,
    ):
        """
        Add a ring of activity centered at (cx, cy).

        Useful for galaxy rotation curve experiments.

        Args:
            cx, cy: Center coordinates
            radius: Ring radius
            width: Ring thickness
            rate: Output rate in the ring
        """
        yy, xx = np.ogrid[:self.ny, :self.nx]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        ring_mask = np.abs(dist - radius) < width / 2
        self.rates[ring_mask] = rate

    def add_uniform_disk(
        self,
        cx: int,
        cy: int,
        radius: float,
        rate: float,
    ):
        """Add a uniform disk of activity."""
        yy, xx = np.ogrid[:self.ny, :self.nx]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        disk_mask = dist <= radius
        self.rates[disk_mask] = rate

    def __add__(self, other: SourceMap) -> SourceMap:
        """Combine two source maps by summing rates."""
        if self.ny != other.ny or self.nx != other.nx:
            raise ValueError("SourceMaps must have same dimensions to add")

        result = SourceMap(self.ny, self.nx, background_rate=0.0)
        result.rates = self.rates + other.rates
        return result

    def __iadd__(self, other: SourceMap) -> SourceMap:
        """In-place addition of source maps."""
        if self.ny != other.ny or self.nx != other.nx:
            raise ValueError("SourceMaps must have same dimensions to add")
        self.rates += other.rates
        return self

    def copy(self) -> SourceMap:
        """Create a copy of this source map."""
        result = SourceMap(self.ny, self.nx, background_rate=0.0)
        result.rates = self.rates.copy()
        return result

    def total_activity(self) -> float:
        """Total activity (sum of all rates)."""
        return float(self.rates.sum())

    def max_rate(self) -> float:
        """Maximum rate in the map."""
        return float(self.rates.max())

    def get_rate(self, x: int, y: int) -> float:
        """Get output rate at a specific node."""
        return self.rates[y, x]
