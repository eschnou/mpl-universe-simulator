#!/usr/bin/env python3
"""
Demo: Gravitational Flyby (Light Bending Analog)

A particle passes near a mass concentration and gets deflected,
demonstrating gravitational lensing from bandwidth limits.

The mechanism:
1. High activity at source → message congestion → low f(x)
2. Particle experiences acceleration toward low f (∇f)
3. Trajectory bends toward the mass
4. This is gravitational deflection!

Output: output/flyby/flyby.png
"""

import numpy as np
import matplotlib.pyplot as plt

from mplsim.core import Lattice, LatticeConfig, SourceMap, LoadGeneratorKernel, TheoreticalScheduler, TheoreticalSchedulerConfig
from mplsim.patterns import create_particle
from mplsim.viz.fields import CMAP_GRAVITY


def main():
    print("=" * 50)
    print("  GRAVITATIONAL FLYBY DEMONSTRATION")
    print("=" * 50)

    # Setup grid and source
    grid_size = 140
    cx, cy = 70, 70

    print("\nSetting up simulation...")
    source_map = SourceMap(ny=grid_size, nx=grid_size, background_rate=0.05)
    source_map.add_gaussian_source(cx=cx, cy=cy, peak_rate=30.0, sigma=12)

    config = LatticeConfig(
        nx=grid_size,
        ny=grid_size,
        neighborhood="von_neumann",
        boundary="periodic",
        link_capacity=15.0,  # Higher capacity = weaker gravity
        spatial_sigma=1.0,  # Enable spatial smoothing for gradients
    )
    lattice = Lattice(config)
    kernel = LoadGeneratorKernel(message_size=1.0, sync_required=True)

    # Use TheoreticalScheduler for proper f field (implements λ = γa + β⟨λ⟩)
    scheduler_config = TheoreticalSchedulerConfig(
        gamma=0.5,
        beta=0.9,
        lambda_decay=0.0,
        lambda_smoothing=0.3,
    )
    scheduler = TheoreticalScheduler(
        lattice=lattice,
        source_map=source_map,
        kernel=kernel,
        config=scheduler_config,
    )

    # Establish f field
    print("Establishing f field (500 ticks)...", end=" ", flush=True)
    scheduler.run(500)
    print("done")

    print(f"  f at center: {lattice.f[cy, cx]:.3f}")
    print(f"  f at edge: {lattice.f[cy, 130]:.3f}")

    # Create flyby particle
    # Starts left side, above center, moving right
    # Impact parameter ~30 units (distance above source center)
    particle = create_particle(
        "flyby",
        x=5, y=100,             # 30 units above center
        lattice=lattice,
        vx=1.5, vy=0.0,         # Moving right, fast enough to escape
        acceleration_scale=1.0,
    )

    print(f"\nParticle initial state:")
    print(f"  Position: ({particle.px:.1f}, {particle.py:.1f})")
    print(f"  Velocity: ({particle.vx:.3f}, {particle.vy:.3f})")

    # Run simulation (particles move through established f field)
    # Long enough to see the full deflection as particle exits gravity well
    sim_ticks = 150
    print(f"\nRunning simulation ({sim_ticks} ticks)...", end=" ", flush=True)
    for t in range(sim_ticks):
        particle.update(t)
    print("done")

    # Analyze trajectory
    x_traj, y_traj = particle.get_trajectory_arrays()
    distances = np.sqrt((x_traj - cx)**2 + (y_traj - cy)**2)
    min_dist = distances.min()
    min_idx = distances.argmin()

    initial_angle = 0  # Started moving purely in +x direction
    final_angle = np.degrees(np.arctan2(particle.vy, particle.vx))
    deflection = abs(final_angle)

    print(f"\nParticle final state:")
    print(f"  Position: ({particle.px:.1f}, {particle.py:.1f})")
    print(f"  Velocity: ({particle.vx:.3f}, {particle.vy:.3f})")
    print(f"\nTrajectory analysis:")
    print(f"  Closest approach: {min_dist:.1f} units (at t={min_idx})")
    print(f"  Deflection angle: {deflection:.1f}°")

    # Create visualization
    print("\nGenerating visualization...")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Background: f_smooth field (coarse-grained, continuous gradient)
    im = ax.imshow(lattice.f_smooth, origin="lower", cmap=CMAP_GRAVITY, aspect="equal")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("f(x) smoothed (coarse-grained)", fontsize=11)

    # Trajectory
    ax.plot(x_traj, y_traj, color="white", linewidth=3, label="Particle trajectory", zorder=3)

    # Markers
    ax.scatter([x_traj[0]], [y_traj[0]], color="lime", s=200, marker="o",
               edgecolors="white", linewidths=2, label="Start", zorder=5)
    ax.scatter([x_traj[-1]], [y_traj[-1]], color="red", s=200, marker="s",
               edgecolors="white", linewidths=2, label="End", zorder=5)
    ax.scatter([cx], [cy], color="yellow", s=400, marker="*",
               edgecolors="black", linewidths=2, label="Mass (source)", zorder=5)
    ax.scatter([x_traj[min_idx]], [y_traj[min_idx]], color="orange", s=120,
               marker="x", linewidths=3, label=f"Closest: {min_dist:.0f} units", zorder=5)

    # Undeflected reference path
    ax.plot([5, 135], [100, 100], 'w--', linewidth=1.5, alpha=0.5, label="Undeflected path")

    ax.set_title(f"Gravitational Flyby: Deflection ≈ {deflection:.0f}°",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    # Add physics annotation
    ax.text(0.02, 0.02,
            "Particle deflected by gradient in f(x)\n"
            "f is low near mass → acceleration toward mass",
            transform=ax.transAxes, fontsize=9, style='italic',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()

    # Save
    import os
    os.makedirs("output/flyby", exist_ok=True)
    fig.savefig("output/flyby/flyby.png", dpi=150, bbox_inches="tight")
    print("\nSaved: output/flyby/flyby.png")
    plt.close()

    print("\n" + "=" * 50)
    print("  Flyby demonstration complete!")
    print("=" * 50)
    print("\nPhysical interpretation:")
    print("  • Particle approached mass with horizontal velocity")
    print(f"  • Deflected {deflection:.0f}° toward the mass")
    print("  • This is gravitational lensing from bandwidth limits!")


if __name__ == "__main__":
    main()
