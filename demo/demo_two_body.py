#!/usr/bin/env python3
"""
Demo: Two-Body Gravity Field

Shows how two masses create overlapping gravity wells:
1. Set up two uniform disk masses
2. Run the engine to steady state
3. Visualize the combined f field
4. Show cross-sections through both masses

This demonstrates superposition of emergent gravity from bandwidth limits.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mplsim.core import (
    Lattice, LatticeConfig, SourceMap, LoadGeneratorKernel,
    BandwidthScheduler, BandwidthSchedulerConfig
)
from mplsim.viz.fields import CMAP_GRAVITY


def main():
    np.random.seed(42)

    print("=" * 60)
    print("  TWO-BODY EMERGENT GRAVITY")
    print("=" * 60)

    # Grid setup
    grid_size = 200
    cx, cy = grid_size // 2, grid_size // 2

    # Mass parameters - two masses separated horizontally
    mass_radius = 10
    mass_rate = 1.0
    separation = 50  # Distance between mass centers

    # Mass positions
    mass1_x, mass1_y = cx - separation // 2, cy
    mass2_x, mass2_y = cx + separation // 2, cy

    print(f"\n1. Setup:")
    print(f"   Grid: {grid_size}x{grid_size}")
    print(f"   Mass 1: disk at ({mass1_x}, {mass1_y}), radius={mass_radius}, rate={mass_rate}")
    print(f"   Mass 2: disk at ({mass2_x}, {mass2_y}), radius={mass_radius}, rate={mass_rate}")
    print(f"   Separation: {separation} lattice units")

    # Create source map with two masses
    source_map = SourceMap(ny=grid_size, nx=grid_size, background_rate=0.01)
    source_map.add_uniform_disk(cx=mass1_x, cy=mass1_y, radius=mass_radius, rate=mass_rate)
    source_map.add_uniform_disk(cx=mass2_x, cy=mass2_y, radius=mass_radius, rate=mass_rate)

    # Create lattice
    config = LatticeConfig(
        nx=grid_size,
        ny=grid_size,
        neighborhood="von_neumann",
        boundary="periodic",
        link_capacity=10.0,
        spatial_sigma=2.0,
    )
    lattice = Lattice(config)
    kernel = LoadGeneratorKernel(message_size=1.0, sync_required=True)

    # Scheduler with same settings as radial profile demo
    damping = 0.9
    scheduler_config = BandwidthSchedulerConfig(
        bandwidth=8.0,
        data_scale=8.0,
        damping=damping,
        base_interval=50.0,
        stochastic=True,
    )
    scheduler = BandwidthScheduler(
        lattice=lattice,
        source_map=source_map,
        kernel=kernel,
        config=scheduler_config,
    )

    print(f"   Scheduler: damping={damping}, bandwidth={scheduler_config.bandwidth}")
    print(f"   Message size: Poisson(activity * {scheduler_config.data_scale})")

    # Run to steady state
    print("\n2. Running to steady state...")
    n_ticks = 10000
    stats = scheduler.run(n_ticks)
    print(f"   {n_ticks} ticks completed")
    print(f"   Mean f: {stats['mean_f']:.4f}")
    print(f"   Min f:  {stats['min_f']:.4f}")

    # Compute f values at key locations
    print("\n3. Field values at key locations:")
    print(f"   f at mass 1 center: {lattice.f_smooth[mass1_y, mass1_x]:.3f}")
    print(f"   f at mass 2 center: {lattice.f_smooth[mass2_y, mass2_x]:.3f}")
    print(f"   f at midpoint:      {lattice.f_smooth[cy, cx]:.3f}")
    print(f"   f at edge (r=80):   {lattice.f_smooth[cy, cx + 80]:.3f}")

    # Create visualization
    print("\n4. Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Activity map
    ax = axes[0, 0]
    im = ax.imshow(source_map.rates, origin="lower", cmap="hot")
    circle1 = plt.Circle((mass1_x, mass1_y), mass_radius, fill=False, color='cyan', linewidth=2)
    circle2 = plt.Circle((mass2_x, mass2_y), mass_radius, fill=False, color='cyan', linewidth=2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.set_title(f"Activity Map (Two Masses, r={mass_radius})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="Activity rate")

    # Panel 2: f field
    ax = axes[0, 1]
    im = ax.imshow(lattice.f_smooth, origin="lower", cmap=CMAP_GRAVITY, vmin=0, vmax=1)
    ax.scatter([mass1_x, mass2_x], [mass1_y, mass2_y], color='red', s=100, marker='*', edgecolors='white', zorder=5)
    ax.scatter([cx], [cy], color='orange', s=80, marker='o', edgecolors='white', zorder=5, label='Midpoint')
    circle1 = plt.Circle((mass1_x, mass1_y), mass_radius, fill=False, color='white', linewidth=1, linestyle='--')
    circle2 = plt.Circle((mass2_x, mass2_y), mass_radius, fill=False, color='white', linewidth=1, linestyle='--')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.set_title("Emergent f(x) Field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="f(x)")

    # Panel 3: Horizontal cross-section through both masses
    ax = axes[1, 0]
    x_coords = np.arange(grid_size)
    f_slice = lattice.f_smooth[cy, :]
    lambda_slice = 1.0 - f_slice

    ax.plot(x_coords, f_slice, 'b-', linewidth=2, label='f(x)')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=mass1_x, color='red', linestyle='--', alpha=0.5, label='Mass centers')
    ax.axvline(x=mass2_x, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=cx, color='orange', linestyle=':', alpha=0.7, label='Midpoint')

    # Mark mass regions
    ax.axvspan(mass1_x - mass_radius, mass1_x + mass_radius, alpha=0.2, color='red')
    ax.axvspan(mass2_x - mass_radius, mass2_x + mass_radius, alpha=0.2, color='red')

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Horizontal Cross-Section (y={cy})")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, 1.05)

    # Panel 4: λ field (slowness) with potential well visualization
    ax = axes[1, 1]
    lambda_field = 1.0 - lattice.f_smooth
    im = ax.imshow(lambda_field, origin="lower", cmap="magma", vmin=0)
    ax.scatter([mass1_x, mass2_x], [mass1_y, mass2_y], color='cyan', s=100, marker='*', edgecolors='white', zorder=5)

    # Add contour lines
    contour_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    contours = ax.contour(lambda_field, levels=contour_levels, colors='white', alpha=0.5, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

    ax.set_title("λ(x) = 1 - f(x) (Gravity Potential Analog)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="λ(x)")

    fig.suptitle(f"Two-Body Emergent Gravity\nSeparation={separation}, damping={damping}, rate={mass_rate}",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    # Save
    output_dir = Path("output/demo_two_body")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "two_body.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  • Two masses create overlapping gravity wells")
    print(f"  • f is lowest at mass centers ({lattice.f_smooth[mass1_y, mass1_x]:.2f})")
    print(f"  • f at midpoint: {lattice.f_smooth[cy, cx]:.2f} (between the wells)")
    print(f"  • Gravity wells superpose: combined λ field shows interaction")
    print(f"  • This is emergent two-body gravity from bandwidth limits!")
    print("=" * 60)


if __name__ == "__main__":
    main()
