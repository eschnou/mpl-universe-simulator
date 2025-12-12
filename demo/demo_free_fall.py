#!/usr/bin/env python3
"""
Demo: Gravitational Free Fall from Bandwidth Limits

This demonstration shows how particles "fall" toward mass concentrations
purely through the bandwidth-limiting mechanism:

1. High activity at the source creates message congestion
2. Congestion causes f(x) < 1 near the source
3. Particles accelerate along ∇f (toward low f = toward mass)
4. This IS gravitational free fall!

Output: PNG visualizations in output/free_fall/ of particle trajectories over f(x) field.
"""

import numpy as np
import matplotlib.pyplot as plt

from mplsim.core import Lattice, LatticeConfig, SourceMap, LoadGeneratorKernel, TheoreticalScheduler, TheoreticalSchedulerConfig
from mplsim.patterns import create_particle
from mplsim.viz import plot_trajectories, plot_free_fall_analysis, save_figure
from mplsim.viz.fields import CMAP_GRAVITY


def setup_experiment(
    grid_size: int = 80,
    source_rate: float = 40.0,
    sigma: float = 10.0,
) -> tuple[Lattice, TheoreticalScheduler, tuple[int, int]]:
    """Set up a standard free fall experiment using TheoreticalScheduler.

    The TheoreticalScheduler implements the paper's λ dynamics directly:
    λ(x) = γ·a(x) + β·⟨λ⟩_x

    This produces the proper gravitational well (low f at source).
    """
    center = grid_size // 2
    source_position = (center, center)

    source_map = SourceMap(ny=grid_size, nx=grid_size, background_rate=0.1)
    source_map.add_gaussian_source(cx=center, cy=center, peak_rate=source_rate, sigma=sigma)

    config = LatticeConfig(
        nx=grid_size,
        ny=grid_size,
        neighborhood="von_neumann",
        boundary="periodic",
        link_capacity=15.0,  # Higher capacity = weaker gravity for nicer demos
        spatial_sigma=1.0,  # Enable spatial smoothing for gradients
    )
    lattice = Lattice(config)
    kernel = LoadGeneratorKernel(message_size=1.0, sync_required=True)

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

    return lattice, scheduler, source_position


def establish_f_field(scheduler: TheoreticalScheduler, n_ticks: int = 400):
    """Run simulation to establish stable f(x) field."""
    print(f"  Establishing f field ({n_ticks} ticks)...", end=" ", flush=True)
    scheduler.run(n_ticks)
    print("done")


def demo_radial_infall():
    """Demo 1: Particles falling radially inward from all directions."""
    print("\n=== Demo 1: Radial Infall ===")

    lattice, scheduler, (cx, cy) = setup_experiment()
    establish_f_field(scheduler)

    # Create particles at 8 compass points, 25 units from center
    radius = 25
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    particles = []

    for i, angle in enumerate(angles):
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        name = f"p{i}"
        particle = create_particle(name, x=x, y=y, lattice=lattice, acceleration_scale=1.5)
        particles.append(particle)

    # Run simulation
    print("  Running particles...")
    for t in range(300):
        for p in particles:
            p.update(t)

    # Create visualization (use f_smooth for continuous gradient display)
    fig = plot_trajectories(
        particles,
        background=lattice.f_smooth,
        title="Gravitational Free Fall: Particles Fall Toward Mass",
        cmap=CMAP_GRAVITY,
        source_position=(cx, cy),
    )

    # Add annotation
    fig.text(0.02, 0.02,
             "Particles released at rest accelerate toward low-f region (high mass)",
             fontsize=9, style='italic', transform=fig.transFigure)

    save_figure(fig, "output/free_fall/radial.png")
    print("  Saved: output/free_fall/radial.png")

    return particles, (cx, cy)


def demo_orbital_deflection():
    """Demo 2: Particles with tangential velocity get deflected."""
    print("\n=== Demo 2: Orbital Deflection ===")

    lattice, scheduler, (cx, cy) = setup_experiment()
    establish_f_field(scheduler)

    # Create particles with tangential velocities at different distances
    particles = []

    # Particle moving "up" at different distances with higher velocities
    for dist, speed in [(20, 0.4), (25, 0.5), (30, 0.6)]:
        p = create_particle(
            f"d={dist}",
            x=cx + dist, y=cy,
            lattice=lattice,
            vx=0, vy=speed,  # Moving up (tangential)
            acceleration_scale=1.5,
        )
        particles.append(p)

    # Run simulation longer to see orbital deflection
    print("  Running particles...")
    for t in range(400):
        for p in particles:
            p.update(t)

    # Create visualization (use f_smooth for continuous gradient display)
    fig = plot_trajectories(
        particles,
        background=lattice.f_smooth,
        title="Orbital Deflection: Tangential Velocity Curves Toward Mass",
        cmap=CMAP_GRAVITY,
        source_position=(cx, cy),
    )

    save_figure(fig, "output/free_fall/orbital.png")
    print("  Saved: output/free_fall/orbital.png")


def demo_time_dilation_effect():
    """Demo 3: Show time dilation affecting particle motion."""
    print("\n=== Demo 3: Time Dilation Effect ===")

    lattice, scheduler, (cx, cy) = setup_experiment()
    establish_f_field(scheduler)

    # Two particles at different distances from source
    p_near = create_particle("near", x=cx+15, y=cy, lattice=lattice, acceleration_scale=1.5)
    p_far = create_particle("far", x=cx+30, y=cy, lattice=lattice, acceleration_scale=1.5)

    particles = [p_near, p_far]

    # Run simulation
    print("  Running particles...")
    for t in range(300):
        for p in particles:
            p.update(t)

    # Analyze results
    fig = plot_free_fall_analysis(
        particles,
        source_position=(cx, cy),
        title="Free Fall Analysis: Distance and Speed vs Time",
    )

    save_figure(fig, "output/free_fall/analysis.png")
    print("  Saved: output/free_fall/analysis.png")

    # Print time dilation results
    print("\n  Time dilation results:")
    print(f"    Particle 'near' (started 10 units from source):")
    print(f"      Proper time: {p_near.proper_time:.1f} (canonical: 200)")
    print(f"    Particle 'far' (started 25 units from source):")
    print(f"      Proper time: {p_far.proper_time:.1f} (canonical: 200)")
    print(f"    Ratio: {p_near.proper_time / p_far.proper_time:.3f}")


def demo_comparison_plot():
    """Demo 4: Combined trajectories and f-field visualization."""
    print("\n=== Demo 4: Combined Visualization ===")

    lattice, scheduler, (cx, cy) = setup_experiment(grid_size=100, source_rate=30.0, sigma=10)
    establish_f_field(scheduler, n_ticks=500)

    # Create diverse particle set
    particles = []

    # Radial infall from 4 cardinal directions
    for i, angle in enumerate([0, np.pi/2, np.pi, 3*np.pi/2]):
        x = cx + 30 * np.cos(angle)
        y = cy + 30 * np.sin(angle)
        p = create_particle(f"radial_{i}", x=x, y=y, lattice=lattice, acceleration_scale=1.5)
        particles.append(p)

    # Orbital - particle with tangential velocity
    p_orbit = create_particle("orbit", x=cx+35, y=cy, lattice=lattice,
                              vx=0, vy=0.4, acceleration_scale=1.5)
    particles.append(p_orbit)

    # Run
    print("  Running particles...")
    for t in range(400):
        for p in particles:
            p.update(t)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: f_smooth field with trajectories (continuous gradient)
    ax1 = axes[0]
    im = ax1.imshow(lattice.f_smooth, origin="lower", cmap=CMAP_GRAVITY, aspect="equal")
    plt.colorbar(im, ax=ax1, label="f(x) smoothed (coarse-grained)")

    colors = plt.cm.tab10(np.linspace(0, 1, len(particles)))
    for p, c in zip(particles, colors):
        x_traj, y_traj = p.get_trajectory_arrays()
        ax1.plot(x_traj, y_traj, color=c, linewidth=2, label=p.config.pattern_id)
        ax1.scatter([x_traj[0]], [y_traj[0]], color=c, s=80, marker="o", edgecolors="white")
        ax1.scatter([x_traj[-1]], [y_traj[-1]], color=c, s=80, marker="s", edgecolors="white")

    ax1.scatter([cx], [cy], color="yellow", s=200, marker="*",
                edgecolors="black", linewidths=1.5, label="Source", zorder=5)
    ax1.set_title("Particle Trajectories over f(x) Field")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="upper right", fontsize=7)

    # Right: Radial profile of f
    ax2 = axes[1]
    max_r = 45
    r_vals = np.arange(0, max_r)
    f_profile = []

    for r in r_vals:
        # Sample f_smooth along radius
        f_samples = []
        for angle in np.linspace(0, 2*np.pi, 20):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < lattice.shape[1] and 0 <= y < lattice.shape[0]:
                f_samples.append(lattice.f_smooth[y, x])
        f_profile.append(np.mean(f_samples) if f_samples else 1.0)

    ax2.plot(r_vals, f_profile, 'b-', linewidth=2, label="f(r)")
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label="f=1 (no congestion)")
    ax2.set_xlabel("Distance from source (r)")
    ax2.set_ylabel("f(r)")
    ax2.set_title("Radial Profile of f (Gravitational Potential Analog)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_r)
    ax2.set_ylim(0, 1.1)

    # Add annotation
    ax2.annotate("Low f = Slow time\n= High 'gravity'",
                 xy=(5, 0.4), fontsize=9, style='italic',
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.suptitle("Gravity from Bandwidth Limits: Free Fall Demonstration", fontsize=14, fontweight='bold')
    fig.tight_layout()

    save_figure(fig, "output/free_fall/combined.png")
    print("  Saved: output/free_fall/combined.png")


def main():
    """Run all free fall demonstrations."""
    print("=" * 60)
    print("  GRAVITATIONAL FREE FALL FROM BANDWIDTH LIMITS")
    print("  Demonstrating emergent gravity in the simulator")
    print("=" * 60)

    # Ensure output directory exists
    import os
    os.makedirs("output/free_fall", exist_ok=True)

    # Run demos
    demo_radial_infall()
    demo_orbital_deflection()
    demo_time_dilation_effect()
    demo_comparison_plot()

    print("\n" + "=" * 60)
    print("  All demonstrations complete!")
    print("  Output files in: ./output/free_fall/")
    print("=" * 60)

    print("\nPhysical interpretation:")
    print("  - High activity at source → message congestion")
    print("  - Congestion → nodes can't update → f(x) < 1")
    print("  - Particles accelerate toward low f (∝ ∇f)")
    print("  - This IS gravitational attraction!")
    print("  - Time runs slower where f is low (time dilation)")


if __name__ == "__main__":
    main()
