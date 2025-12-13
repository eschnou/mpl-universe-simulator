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

Output: output/demo_flyby/flyby.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from mplsim.core import Lattice, LatticeConfig, SourceMap, LoadGeneratorKernel, BandwidthScheduler, BandwidthSchedulerConfig
from mplsim.patterns import create_particle
from mplsim.viz.fields import CMAP_GRAVITY


def main():
    print("=" * 60)
    print("  GRAVITATIONAL FLYBY DEMONSTRATION")
    print("=" * 60)

    # Setup grid and point source (scaled to 250x250)
    grid_size = 250
    cx, cy = 125, 125

    print("\n1. Setting up mass (small disk)...")
    mass_radius = 9
    source_map = SourceMap(ny=grid_size, nx=grid_size, background_rate=0.01)
    source_map.add_uniform_disk(cx=cx, cy=cy, radius=mass_radius, rate=50.0)
    print(f"   Disk source at ({cx}, {cy}), radius={mass_radius}, rate=50.0")

    config = LatticeConfig(
        nx=grid_size,
        ny=grid_size,
        neighborhood="von_neumann",
        boundary="absorbing",
        link_capacity=15.0,
        spatial_sigma=3.5,  # Scaled for 250x250 grid
    )
    lattice = Lattice(config)
    kernel = LoadGeneratorKernel(message_size=1.0, sync_required=True)

    # BandwidthScheduler with β<1 for screened Poisson regime
    scheduler_config = BandwidthSchedulerConfig(
        canonical_interval=10,
        bandwidth_per_tick=1.0,
        propagation_delay=1,
        f_smoothing_alpha=0.03,
        beta=0.9,  # β<1 for proper gradient decay
        message_rate_scale=12.0,  # Higher activity for stronger effect
        stochastic_messages=True,
    )
    scheduler = BandwidthScheduler(
        lattice=lattice,
        source_map=source_map,
        kernel=kernel,
        config=scheduler_config,
    )

    # Establish f field
    print("\n2. Establishing f field (5000 ticks)...")
    scheduler.run(5000)
    print(f"   f at center: {lattice.f[cy, cx]:.3f}")
    print(f"   f at r=35: {lattice.f[cy, cx+35]:.3f}")
    print(f"   f at r=90: {lattice.f[cy, cx+90]:.3f}")

    # Compute radial profile for analysis (start outside the mass)
    print("\n3. Computing radial profile...")
    r_vals = np.arange(mass_radius + 1, 100, 1)  # Start outside mass
    lambda_profile = []
    for r in r_vals:
        samples = []
        for angle in np.linspace(0, 2*np.pi, 36, endpoint=False):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < grid_size and 0 <= y < grid_size:
                samples.append(1.0 - lattice.f_smooth[y, x])
        lambda_profile.append(np.mean(samples))

    r_vals = np.array(r_vals)
    lambda_profile = np.array(lambda_profile)

    # Fit to screened Poisson (exponential decay)
    def exp_decay(r, a, xi, b):
        return a * np.exp(-r / xi) + b

    mask = lambda_profile > 0.001
    try:
        popt, _ = curve_fit(exp_decay, r_vals[mask], lambda_profile[mask],
                           p0=[0.3, 10, 0], bounds=([0, 1, -0.1], [10, 100, 0.1]), maxfev=5000)
        xi_fit = popt[1]
        fit_exp = exp_decay(r_vals, *popt)
        r2 = 1 - np.sum((lambda_profile - fit_exp)**2) / np.sum((lambda_profile - lambda_profile.mean())**2)
        print(f"   Fitted: λ(r) = {popt[0]:.3f} × exp(-r/{xi_fit:.1f}) + {popt[2]:.4f}")
        print(f"   R² = {r2:.4f}")
    except Exception as e:
        xi_fit = 10
        fit_exp = None
        r2 = 0
        print(f"   Fit failed: {e}")

    # Create flyby particle
    print("\n4. Creating flyby particle...")
    impact_param = 18  # Distance above center (inside gravity well)
    particle = create_particle(
        "flyby",
        x=18, y=cy + impact_param,
        lattice=lattice,
        vx=1.0, vy=0.0,  # Moving right
        acceleration_scale=2.0,
    )
    print(f"   Start: ({particle.px:.0f}, {particle.py:.0f})")
    print(f"   Velocity: ({particle.vx:.2f}, {particle.vy:.2f})")
    print(f"   Impact parameter: {impact_param}")

    # Run simulation
    print("\n5. Running flyby simulation...")
    sim_ticks = 180  # Scaled for larger grid
    for t in range(sim_ticks):
        particle.update(t)

    # Analyze trajectory
    x_traj, y_traj = particle.get_trajectory_arrays()
    distances = np.sqrt((x_traj - cx)**2 + (y_traj - cy)**2)
    min_dist = distances.min()
    min_idx = distances.argmin()

    final_angle = np.degrees(np.arctan2(particle.vy, particle.vx))
    deflection = abs(final_angle)

    print(f"   End: ({particle.px:.0f}, {particle.py:.0f})")
    print(f"   Closest approach: {min_dist:.1f} units")
    print(f"   Deflection angle: {deflection:.1f}°")

    # Create 4-panel visualization
    print("\n6. Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Activity map
    ax = axes[0, 0]
    im = ax.imshow(source_map.rates, origin="lower", cmap="hot", aspect="equal")
    circle = plt.Circle((cx, cy), mass_radius, fill=False, color='cyan', linewidth=2)
    ax.add_patch(circle)
    ax.set_title(f"Activity Map (Disk r={mass_radius})", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="Activity rate")

    # Panel 2: f(x) field
    ax = axes[0, 1]
    im = ax.imshow(lattice.f_smooth, origin="lower", cmap=CMAP_GRAVITY, aspect="equal")
    ax.scatter([cx], [cy], color='red', s=100, marker='*', edgecolors='white', zorder=5)
    ax.set_title("Emergent f(x) Field", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="f(x)")

    # Panel 3: Radial profile with fit
    ax = axes[1, 0]
    ax.plot(r_vals, lambda_profile, 'ko-', markersize=3, label='Simulated λ(r)')
    if fit_exp is not None:
        ax.plot(r_vals, fit_exp, 'r-', linewidth=2,
                label=f'exp(-r/{xi_fit:.0f}), R²={r2:.3f}')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=mass_radius, color='orange', linestyle=':', alpha=0.7, label=f'Mass surface (r={mass_radius})')
    ax.set_xlabel("Distance r from center")
    ax.set_ylabel("λ(r) = 1 - f(r)")
    ax.set_title("Radial Profile: Screened Poisson λ(r) ∝ exp(-r/ξ)", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    # Add annotation about screening
    ax.annotate(f"Screening length ξ ≈ {xi_fit:.0f}\n(β = {scheduler_config.beta})",
                xy=(60, max(0.02, lambda_profile[5] if len(lambda_profile) > 5 else 0.05)), fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel 4: Flyby trajectory
    ax = axes[1, 1]
    im = ax.imshow(lattice.f_smooth, origin="lower", cmap=CMAP_GRAVITY, aspect="equal", alpha=0.8)

    # Trajectory
    ax.plot(x_traj, y_traj, color="white", linewidth=3, label="Particle trajectory", zorder=3)

    # Markers
    ax.scatter([x_traj[0]], [y_traj[0]], color="lime", s=200, marker="o",
               edgecolors="white", linewidths=2, label="Start", zorder=5)
    ax.scatter([x_traj[-1]], [y_traj[-1]], color="red", s=200, marker="s",
               edgecolors="white", linewidths=2, label="End", zorder=5)
    ax.scatter([cx], [cy], color="yellow", s=400, marker="*",
               edgecolors="black", linewidths=2, label="Mass", zorder=5)
    ax.scatter([x_traj[min_idx]], [y_traj[min_idx]], color="orange", s=120,
               marker="x", linewidths=3, label=f"Closest: {min_dist:.0f}", zorder=5)

    # Undeflected reference path
    ax.plot([18, 232], [cy + impact_param, cy + impact_param], 'w--',
            linewidth=1.5, alpha=0.5, label="Undeflected")

    ax.set_title(f"Gravitational Flyby: Deflection = {deflection:.0f}°", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(55, 195)

    fig.suptitle("Gravitational Lensing from Bandwidth Limits", fontsize=14, fontweight='bold')
    fig.tight_layout()

    # Save
    import os
    os.makedirs("output/demo_flyby", exist_ok=True)
    fig.savefig("output/demo_flyby/flyby.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n   Saved: output/demo_flyby/flyby.png")

    print("\n" + "=" * 60)
    print("  Flyby demonstration complete!")
    print("=" * 60)
    print("\nPhysical interpretation:")
    print(f"  • Disk source (r={mass_radius}) creates congestion → low f near mass")
    print(f"  • λ(r) follows screened Poisson: exp(-r/{xi_fit:.0f})")
    print(f"  • Particle deflected {deflection:.0f}° by gradient ∇f")
    print("  • This IS gravitational lensing from bandwidth limits!")


if __name__ == "__main__":
    main()
