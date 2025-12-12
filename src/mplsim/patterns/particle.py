"""
GeodesicParticle: test particle that follows geodesics in the f(x) field.

The particle experiences "gravity" through the gradient of f:
- Acceleration a ∝ ∇f (points toward lower f = higher mass)
- Local time runs slower where f is low (proper time dilation)
- Trajectory curves toward mass concentrations

This is geodesic motion in the emergent geometry, demonstrating that
bandwidth-limited message passing creates gravitational free fall.

NOTE: This is an APPROXIMATE geodesic integrator using the f field.
A fully native implementation would use wave packets, but this suffices
to demonstrate the gravitational effect.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mplsim.core.lattice import Lattice

from mplsim.patterns.base import Pattern, PatternConfig


@dataclass
class ParticleConfig(PatternConfig):
    """Configuration for a test particle."""

    # Initial velocity
    vx: float = 0.0
    vy: float = 0.0

    # Physics parameters
    mass: float = 1.0  # Particle mass (affects inertia)
    acceleration_scale: float = 1.0  # Scale factor for ∇f acceleration


class GeodesicParticle(Pattern):
    """
    A test particle that follows geodesics in the f(x) field.

    The particle:
    1. Experiences acceleration a = scale * ∇f (toward low f regions)
    2. Has its time step modulated by local f(x)
    3. Records its trajectory for analysis

    This demonstrates gravitational free fall emerging from bandwidth limits.
    """

    def __init__(self, config: ParticleConfig, lattice: "Lattice"):
        super().__init__(config, lattice)

        # Position (floating point for smooth motion)
        self.px: float = float(config.center_x)
        self.py: float = float(config.center_y)

        # Velocity
        self.vx: float = config.vx
        self.vy: float = config.vy

        # Physics parameters
        self.mass = config.mass
        self.acceleration_scale = config.acceleration_scale

        # Trajectory recording
        self.trajectory: list[tuple[float, float, float, float]] = []
        # Each entry: (px, py, vx, vy)

        # Proper time accumulator for this particle
        self.proper_time: float = 0.0

        # Record initial position
        self._record_state()

    @property
    def center(self) -> tuple[int, int]:
        """Current center position (rounded to grid)."""
        return int(round(self.px)), int(round(self.py))

    @property
    def position(self) -> tuple[float, float]:
        """Current position (floating point)."""
        return self.px, self.py

    @property
    def velocity(self) -> tuple[float, float]:
        """Current velocity."""
        return self.vx, self.vy

    @property
    def speed(self) -> float:
        """Current speed."""
        return np.sqrt(self.vx**2 + self.vy**2)

    def update(self, canonical_tick: int) -> None:
        """
        Update particle position using geodesic motion.

        Integration scheme:
        1. Compute local f at current position
        2. Compute acceleration from ∇f
        3. Update velocity (with time dilation factor)
        4. Update position
        5. Record trajectory
        """
        self._canonical_tick = canonical_tick

        # Get local f value (raw, not smoothed) for time dilation
        f_local = self._get_f_at_position(self.px, self.py, use_smoothed=False)

        # Time step is modulated by local f
        # In low-f regions, time runs slower → smaller effective dt
        dt = f_local  # dt ∈ [0, 1]

        # Compute acceleration from f gradient
        ax, ay = self._compute_acceleration(self.px, self.py)

        # Update velocity (Euler integration with time dilation)
        self.vx += ax * dt
        self.vy += ay * dt

        # Update position
        self.px += self.vx * dt
        self.py += self.vy * dt

        # Keep particle in bounds (periodic wrapping)
        ny, nx = self.lattice.shape
        self.px = self.px % nx
        self.py = self.py % ny

        # Accumulate proper time
        self.proper_time += dt

        # Record state
        self._record_state()

    def _get_f_at_position(self, px: float, py: float, use_smoothed: bool = True) -> float:
        """
        Get f value at position using bilinear interpolation.

        Args:
            px, py: Position (will be wrapped to grid)
            use_smoothed: If True, use f_smooth (for gradients); if False, use raw f (for time dilation)
        """
        ny, nx = self.lattice.shape

        # Wrap coordinates
        px = px % nx
        py = py % ny

        # Get integer coordinates
        x0 = int(px)
        y0 = int(py)
        x1 = (x0 + 1) % nx
        y1 = (y0 + 1) % ny

        # Fractional parts
        fx = px - x0
        fy = py - y0

        # Use f_smooth for gradient computation (continuous), raw f for time dilation (discrete)
        f_field = self.lattice.f_smooth if use_smoothed else self.lattice.f

        # Bilinear interpolation
        f00 = f_field[y0, x0]
        f10 = f_field[y0, x1]
        f01 = f_field[y1, x0]
        f11 = f_field[y1, x1]

        f = (
            f00 * (1 - fx) * (1 - fy)
            + f10 * fx * (1 - fy)
            + f01 * (1 - fx) * fy
            + f11 * fx * fy
        )

        return float(f)

    def _compute_acceleration(self, px: float, py: float) -> tuple[float, float]:
        """
        Compute acceleration from gradient of f.

        Acceleration points toward lower f (higher mass concentration).
        a = scale * ∇f
        """
        ny, nx = self.lattice.shape

        # Use central differences for gradient
        # With smoothed f field, we can use small epsilon for accurate gradients
        eps = 1.0

        f_xp = self._get_f_at_position(px + eps, py)
        f_xm = self._get_f_at_position(px - eps, py)
        f_yp = self._get_f_at_position(px, py + eps)
        f_ym = self._get_f_at_position(px, py - eps)

        df_dx = (f_xp - f_xm) / (2 * eps)
        df_dy = (f_yp - f_ym) / (2 * eps)

        # Acceleration = scale * ∇f
        # This points toward higher f, but we want to fall toward lower f (mass)
        # Since f is lower near mass, ∇f points away from mass
        # So we use a = -scale * ∇f? No wait...
        #
        # Actually: ∇f points from low to high f (away from mass)
        # We want acceleration toward mass (toward low f)
        # So a = -scale * ∇f? But that would push away from mass!
        #
        # The confusion: in standard gravity, a = -∇ϕ where ϕ is negative near mass
        # Here, f is LOW near mass (like 1 + ϕ in weak field)
        # So ∇f points away from mass, and we want a = +∇f to fall toward mass? No...
        #
        # Let me think again: we want particles to accelerate toward LOW f regions.
        # If f is low at x=0 and high at x=10, df/dx > 0.
        # We want ax < 0 to accelerate toward x=0.
        # So ax = -scale * df/dx. Same for y.

        ax = -self.acceleration_scale * df_dx
        ay = -self.acceleration_scale * df_dy

        return ax, ay

    def _record_state(self):
        """Record current state in trajectory."""
        self.trajectory.append((self.px, self.py, self.vx, self.vy))

    def get_measurements(self) -> dict:
        """Return particle measurements."""
        return {
            "pattern_id": self.config.pattern_id,
            "initial_position": (self.config.center_x, self.config.center_y),
            "final_position": (self.px, self.py),
            "final_velocity": (self.vx, self.vy),
            "proper_time": self.proper_time,
            "canonical_time": self._canonical_tick,
            "trajectory_length": len(self.trajectory),
        }

    def get_trajectory_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return trajectory as (x_array, y_array) for plotting."""
        if not self.trajectory:
            return np.array([]), np.array([])

        traj = np.array(self.trajectory)
        return traj[:, 0], traj[:, 1]

    def distance_from_start(self) -> float:
        """Compute distance from starting position."""
        dx = self.px - self.config.center_x
        dy = self.py - self.config.center_y
        return np.sqrt(dx**2 + dy**2)

    def distance_to_point(self, x: float, y: float) -> float:
        """Compute distance to a point."""
        dx = self.px - x
        dy = self.py - y
        return np.sqrt(dx**2 + dy**2)


def create_particle(
    pattern_id: str,
    x: float,
    y: float,
    lattice: "Lattice",
    vx: float = 0.0,
    vy: float = 0.0,
    acceleration_scale: float = 1.0,
) -> GeodesicParticle:
    """
    Convenience factory for creating a test particle.

    Args:
        pattern_id: Unique identifier
        x, y: Initial position
        lattice: The lattice to move on
        vx, vy: Initial velocity
        acceleration_scale: Scale factor for gravitational acceleration

    Returns:
        Configured GeodesicParticle

    Note:
        The particle uses lattice.f_smooth for gradient computation (continuous
        gradients from coarse-grained field) and lattice.f for time dilation
        (discrete counting semantics). Smoothing is configured at the Lattice
        level via LatticeConfig.spatial_sigma.
    """
    config = ParticleConfig(
        pattern_id=pattern_id,
        center_x=int(x),
        center_y=int(y),
        vx=vx,
        vy=vy,
        acceleration_scale=acceleration_scale,
    )
    particle = GeodesicParticle(config, lattice)
    # Override with exact floating point position
    particle.px = float(x)
    particle.py = float(y)
    return particle
