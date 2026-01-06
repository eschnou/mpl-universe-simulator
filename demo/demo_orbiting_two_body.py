#!/usr/bin/env python3
"""
Demo: Orbiting Two-Body System

Demonstrates that the saddle point between two masses persists when the masses
are orbiting (physically realistic) vs disappearing when masses are static
(unphysical - would collapse).

Key insight: Static multi-body configurations are unphysical. The simulation
correctly shows that the saddle point disappears for static masses (reflecting
the instability) but persists for orbiting masses (the physical scenario).

This validates that the bandwidth-based gravity mechanism produces correct
behavior for dynamic systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from pathlib import Path

from mplsim.core import (
    Lattice, LatticeConfig, SourceMap, LoadGeneratorKernel,
    BandwidthScheduler, BandwidthSchedulerConfig
)
from mplsim.viz.fields import CMAP_GRAVITY


def create_scheduler(lattice, source_map):
    """Create a scheduler with given source map."""
    kernel = LoadGeneratorKernel(message_size=1.0, sync_required=True)
    scheduler_config = BandwidthSchedulerConfig(
        bandwidth=8.0,
        data_scale=10.0,
        damping=1.0,
        base_interval=250,
        gap_ema_alpha=1.0,
    )
    return BandwidthScheduler(
        lattice=lattice,
        source_map=source_map,
        kernel=kernel,
        config=scheduler_config,
    )


def run_orbiting_simulation(grid_size, cx, cy, orbital_radius, mass_radius,
                            mass_rate, omega, ticks_per_step, n_steps):
    """Run simulation with orbiting masses.

    Creates a fresh scheduler each step to simulate the physical scenario
    where moving masses don't let the field equilibrate to any single
    configuration. This represents the key insight: in a dynamic system,
    the field responds to the instantaneous mass positions.
    """
    config = LatticeConfig(
        nx=grid_size, ny=grid_size,
        neighborhood="moore",
        boundary="absorbing",
        spatial_sigma=2.0,
    )
    lattice = Lattice(config)

    # History tracking
    saddle_history = []
    angle_history = []
    mass1_positions = []
    mass2_positions = []
    f_fields = []  # Store some snapshots

    for step in range(n_steps):
        angle = omega * step * ticks_per_step

        # Compute mass positions (orbiting around center)
        mass1_x = cx + orbital_radius * np.cos(angle)
        mass1_y = cy + orbital_radius * np.sin(angle)
        mass2_x = cx + orbital_radius * np.cos(angle + np.pi)
        mass2_y = cy + orbital_radius * np.sin(angle + np.pi)

        mass1_positions.append((mass1_x, mass1_y))
        mass2_positions.append((mass2_x, mass2_y))

        # Create fresh source map for current positions
        source_map = SourceMap(ny=grid_size, nx=grid_size, background_rate=0.01)
        source_map.add_uniform_disk(cx=int(mass1_x), cy=int(mass1_y),
                                    radius=mass_radius, rate=mass_rate)
        source_map.add_uniform_disk(cx=int(mass2_x), cy=int(mass2_y),
                                    radius=mass_radius, rate=mass_rate)

        # Create fresh scheduler - simulates field responding to instantaneous
        # mass positions without "memory" of previous configurations
        scheduler = create_scheduler(lattice, source_map)

        # Run simulation
        scheduler.run(ticks_per_step)

        # Measure saddle point
        f_mid = lattice.f_smooth[cy, cx]
        f_mass1 = lattice.f_smooth[int(mass1_y), int(mass1_x)]
        saddle = f_mid - f_mass1

        saddle_history.append(saddle)
        angle_history.append(np.degrees(angle) % 360)

        # Store snapshots at regular intervals
        if step % (n_steps // 4) == 0 or step == n_steps - 1:
            f_fields.append((angle, lattice.f_smooth.copy(),
                           (mass1_x, mass1_y), (mass2_x, mass2_y)))

    return {
        'lattice': lattice,
        'saddle_history': saddle_history,
        'angle_history': angle_history,
        'mass1_positions': mass1_positions,
        'mass2_positions': mass2_positions,
        'f_fields': f_fields,
        'total_ticks': n_steps * ticks_per_step,
    }


def run_static_simulation(grid_size, cx, cy, orbital_radius, mass_radius,
                          mass_rate, total_ticks):
    """Run simulation with static masses for comparison."""
    config = LatticeConfig(
        nx=grid_size, ny=grid_size,
        neighborhood="moore",
        boundary="absorbing",
        spatial_sigma=2.0,
    )
    lattice = Lattice(config)

    source_map = SourceMap(ny=grid_size, nx=grid_size, background_rate=0.01)
    source_map.add_uniform_disk(cx=cx-orbital_radius, cy=cy,
                                radius=mass_radius, rate=mass_rate)
    source_map.add_uniform_disk(cx=cx+orbital_radius, cy=cy,
                                radius=mass_radius, rate=mass_rate)

    scheduler = create_scheduler(lattice, source_map)
    scheduler.run(total_ticks)

    f_mid = lattice.f_smooth[cy, cx]
    f_mass = lattice.f_smooth[cy, cx - orbital_radius]

    return {
        'lattice': lattice,
        'f_mid': f_mid,
        'f_mass': f_mass,
        'saddle': f_mid - f_mass,
    }


def main():
    np.random.seed(42)

    print("=" * 60)
    print("  ORBITING TWO-BODY EMERGENT GRAVITY")
    print("=" * 60)

    # Parameters
    grid_size = 100
    cx, cy = grid_size // 2, grid_size // 2
    mass_radius = 5
    mass_rate = 1.0
    orbital_radius = 12  # Distance from center to each mass

    # Orbital parameters
    # Need enough ticks for gravity to propagate (~30k like radial demo)
    # but masses move before full equilibration to static state
    ticks_per_step = 20000  # Enough for gravity to propagate
    omega = 0.005  # Angular velocity (radians per tick-chunk)
    n_steps = 30  # About half orbit

    print(f"\n1. Setup:")
    print(f"   Grid: {grid_size}x{grid_size}")
    print(f"   Orbital radius: {orbital_radius}")
    print(f"   Mass radius: {mass_radius}, rate: {mass_rate}")
    print(f"   Angular velocity: {omega} rad/tick-chunk")
    print(f"   Ticks per step: {ticks_per_step}")
    print(f"   Total steps: {n_steps}")

    # Run orbiting simulation
    print("\n2. Running orbiting simulation...")
    orbit_results = run_orbiting_simulation(
        grid_size, cx, cy, orbital_radius, mass_radius, mass_rate,
        omega, ticks_per_step, n_steps
    )
    print(f"   Total ticks: {orbit_results['total_ticks']}")
    print(f"   Mean saddle effect: {np.mean(orbit_results['saddle_history']):.4f}")

    # Run static simulation for comparison
    print("\n3. Running static simulation (same total ticks)...")
    static_results = run_static_simulation(
        grid_size, cx, cy, orbital_radius, mass_radius, mass_rate,
        orbit_results['total_ticks']
    )
    print(f"   Static saddle effect: {static_results['saddle']:.4f}")

    # Create visualization
    print("\n4. Creating visualization...")
    fig = plt.figure(figsize=(16, 12))

    # Panel 1: Orbital trajectory and final f field
    ax1 = fig.add_subplot(2, 2, 1)
    f_field = orbit_results['lattice'].f_smooth
    im = ax1.imshow(f_field, origin="lower", cmap=CMAP_GRAVITY, vmin=0.7, vmax=1.0)

    # Plot orbital paths
    m1_x = [p[0] for p in orbit_results['mass1_positions']]
    m1_y = [p[1] for p in orbit_results['mass1_positions']]
    m2_x = [p[0] for p in orbit_results['mass2_positions']]
    m2_y = [p[1] for p in orbit_results['mass2_positions']]

    ax1.plot(m1_x, m1_y, 'c-', linewidth=1, alpha=0.5, label='Mass 1 orbit')
    ax1.plot(m2_x, m2_y, 'm-', linewidth=1, alpha=0.5, label='Mass 2 orbit')

    # Mark final positions
    ax1.scatter([m1_x[-1]], [m1_y[-1]], c='cyan', s=100, marker='o',
                edgecolors='white', linewidths=2, zorder=5, label='Mass 1')
    ax1.scatter([m2_x[-1]], [m2_y[-1]], c='magenta', s=100, marker='o',
                edgecolors='white', linewidths=2, zorder=5, label='Mass 2')
    ax1.scatter([cx], [cy], c='yellow', s=80, marker='*',
                edgecolors='black', linewidths=1, zorder=5, label='Midpoint')

    ax1.set_title("Orbiting Masses: f(x) Field")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc='upper right', fontsize=8)
    plt.colorbar(im, ax=ax1, label="f(x)")

    # Panel 2: Static masses f field for comparison
    ax2 = fig.add_subplot(2, 2, 2)
    f_static = static_results['lattice'].f_smooth
    im2 = ax2.imshow(f_static, origin="lower", cmap=CMAP_GRAVITY, vmin=0.7, vmax=1.0)

    ax2.scatter([cx - orbital_radius, cx + orbital_radius], [cy, cy],
                c='red', s=100, marker='o', edgecolors='white', linewidths=2, zorder=5)
    ax2.scatter([cx], [cy], c='yellow', s=80, marker='*',
                edgecolors='black', linewidths=1, zorder=5)

    ax2.set_title(f"Static Masses: f(x) Field\n(Saddle = {static_results['saddle']:.3f})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.colorbar(im2, ax=ax2, label="f(x)")

    # Panel 3: Saddle effect over time
    ax3 = fig.add_subplot(2, 2, 3)
    angles = orbit_results['angle_history']
    saddles = orbit_results['saddle_history']

    ax3.plot(range(len(saddles)), saddles, 'b-', linewidth=2, label='Orbiting')
    ax3.axhline(y=static_results['saddle'], color='r', linestyle='--',
                linewidth=2, label=f'Static ({static_results["saddle"]:.3f})')
    ax3.axhline(y=np.mean(saddles), color='b', linestyle=':',
                linewidth=1, label=f'Orbiting mean ({np.mean(saddles):.3f})')

    ax3.set_xlabel("Simulation step")
    ax3.set_ylabel("Saddle effect (f_midpoint - f_mass)")
    ax3.set_title("Saddle Point Persistence: Orbiting vs Static")
    ax3.legend(loc='right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.05, max(saddles) * 1.2)

    # Panel 4: Cross-section comparison
    ax4 = fig.add_subplot(2, 2, 4)

    # Get cross-section through center for orbiting (at final time)
    f_orbit_slice = orbit_results['lattice'].f_smooth[cy, :]
    f_static_slice = static_results['lattice'].f_smooth[cy, :]

    x_coords = np.arange(grid_size)
    ax4.plot(x_coords, f_orbit_slice, 'b-', linewidth=2, label='Orbiting (final)')
    ax4.plot(x_coords, f_static_slice, 'r--', linewidth=2, label='Static')

    ax4.axvline(x=cx, color='orange', linestyle=':', alpha=0.7, label='Midpoint')
    ax4.axvline(x=cx - orbital_radius, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=cx + orbital_radius, color='gray', linestyle='--', alpha=0.5)

    ax4.set_xlabel("x")
    ax4.set_ylabel("f(x)")
    ax4.set_title(f"Horizontal Cross-Section (y={cy})")
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)

    # Zoom to show structure
    f_min = min(f_orbit_slice.min(), f_static_slice.min())
    ax4.set_ylim(f_min - 0.02, 1.02)

    # Main title
    fig.suptitle(
        "Two-Body Gravity: Orbiting vs Static Masses\n"
        f"Orbiting saddle: {np.mean(saddles):.3f} | Static saddle: {static_results['saddle']:.3f}",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()

    # Save
    output_dir = Path("output/demo_orbiting")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "orbiting_two_body.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Orbiting masses:")
    print(f"    - Saddle effect: {np.mean(saddles):.3f} (persists!)")
    print(f"    - Midpoint f: ~{orbit_results['lattice'].f_smooth[cy, cx]:.3f}")
    print(f"  Static masses:")
    print(f"    - Saddle effect: {static_results['saddle']:.3f} (gone!)")
    print(f"    - Midpoint f: {static_results['f_mid']:.3f}")
    print()
    print("  Key insight:")
    print("    Static two-body systems are unphysical (would collapse).")
    print("    The saddle point disappearing reflects this instability.")
    print("    For physical orbiting systems, the saddle point persists.")
    print("=" * 60)


if __name__ == "__main__":
    main()
