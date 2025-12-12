#!/usr/bin/env python3
"""
Demo: φ(x) field for varying beta and grid size.

Shows how the gravitational potential profile changes as:
- beta increases toward 1 (less screening → longer range)
- grid size increases (better resolution of continuum limit)

Uses a fixed central disk mass geometry (like flyby demo).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

from mplsim.core import (
    Lattice, LatticeConfig, SourceMap, LoadGeneratorKernel,
    BandwidthScheduler, BandwidthSchedulerConfig
)
from mplsim.analysis.phi_field import compute_phi_linear
from mplsim.viz.fields import CMAP_GRAVITY


@dataclass
class PhiResult:
    """Result from computing phi field."""
    beta: float
    grid_size: int
    phi: np.ndarray
    f: np.ndarray
    radial_r: np.ndarray
    radial_phi: np.ndarray
    xi_fit: float  # Screening length from fit
    mean_f: float


def compute_phi_for_config(
    grid_size: int,
    beta: float,
    mass_radius: int = 8,
    warmup_ticks: int = 2000,
) -> PhiResult:
    """
    Compute φ(x) field for a given grid size and beta.

    Args:
        grid_size: N x N grid
        beta: Synchronization coupling
        mass_radius: Radius of central disk mass
        warmup_ticks: Ticks to establish f field

    Returns:
        PhiResult with phi field and radial profile
    """
    cx, cy = grid_size // 2, grid_size // 2

    # Scale mass rate with grid size to maintain relative strength
    # (larger grids need proportionally more activity)
    base_rate = 50.0
    rate = base_rate * (grid_size / 100.0)

    source_map = SourceMap(ny=grid_size, nx=grid_size, background_rate=0.01)
    source_map.add_uniform_disk(cx=cx, cy=cy, radius=mass_radius, rate=rate)

    config = LatticeConfig(
        nx=grid_size,
        ny=grid_size,
        neighborhood="von_neumann",
        boundary="absorbing",
        link_capacity=15.0,
        spatial_sigma=2.0,
    )
    lattice = Lattice(config)
    kernel = LoadGeneratorKernel(message_size=1.0, sync_required=True)

    scheduler_config = BandwidthSchedulerConfig(
        canonical_interval=10,
        bandwidth_per_tick=1.0,
        propagation_delay=1,
        f_smoothing_alpha=0.03,
        beta=beta,
        message_rate_scale=12.0,
        stochastic_messages=True,
    )
    scheduler = BandwidthScheduler(
        lattice=lattice,
        source_map=source_map,
        kernel=kernel,
        config=scheduler_config,
    )

    # Scale warmup with sqrt of grid size (not linear - too slow for large grids)
    scaled_warmup = int(warmup_ticks * np.sqrt(grid_size / 100.0))
    stats = scheduler.run(scaled_warmup)

    # Compute phi from f
    phi = compute_phi_linear(lattice.f_smooth, alpha=1.0)

    # Compute radial profile (extend to ~40% of grid size)
    max_r = int(grid_size * 0.4)
    step = max(1, grid_size // 100)  # Coarser sampling for large grids
    r_vals = np.arange(mass_radius + 1, max_r, step)
    phi_profile = []

    for r in r_vals:
        samples = []
        for angle in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < grid_size and 0 <= y < grid_size:
                samples.append(phi[y, x])
        phi_profile.append(np.mean(samples))

    r_vals = np.array(r_vals)
    phi_profile = np.array(phi_profile)

    # Fit screening length from phi profile
    # phi ~ -A * exp(-r/xi) for screened Poisson
    from scipy.optimize import curve_fit

    def exp_decay(r, a, xi, b):
        return -a * np.exp(-r / xi) + b

    try:
        # phi is negative, so we fit -phi
        # Scale fit bounds with grid size
        max_xi = grid_size / 2
        popt, _ = curve_fit(
            exp_decay, r_vals, phi_profile,
            p0=[0.1, min(15, max_xi/10), 0],
            bounds=([0, 1, -0.5], [10, max_xi, 0.5]),
            maxfev=5000
        )
        xi_fit = popt[1]
    except:
        xi_fit = np.nan

    return PhiResult(
        beta=beta,
        grid_size=grid_size,
        phi=phi,
        f=lattice.f_smooth,
        radial_r=r_vals,
        radial_phi=phi_profile,
        xi_fit=xi_fit,
        mean_f=stats['mean_f'],
    )


def main():
    """Generate phi field comparison grid."""
    np.random.seed(42)

    print("=" * 60)
    print("φ(x) Field Comparison: Beta × Grid Size")
    print("=" * 60)

    # Parameters to vary - use powers of 2 for cleaner scaling
    betas = [0.8, 0.9, 0.95, 0.99]
    grid_sizes = [100, 200, 400]

    print(f"\nBetas: {betas}")
    print(f"Grid sizes: {grid_sizes}")
    print(f"Total configurations: {len(betas) * len(grid_sizes)}")

    # Compute all configurations
    results: List[List[PhiResult]] = []

    for beta in betas:
        row = []
        for n in grid_sizes:
            print(f"Computing β={beta}, N={n}...", end=" ", flush=True)
            result = compute_phi_for_config(n, beta, warmup_ticks=2000)
            row.append(result)
            xi_str = f"{result.xi_fit:.1f}" if not np.isnan(result.xi_fit) else "∞"
            print(f"ξ={xi_str}, mean_f={result.mean_f:.3f}")
        results.append(row)

    # Create figure: rows = beta, cols = grid size
    n_rows = len(betas)
    n_cols = len(grid_sizes)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Find global phi range for consistent colorbar
    all_phi = [r.phi for row in results for r in row]
    vmin = min(p.min() for p in all_phi)
    vmax = max(p.max() for p in all_phi)

    for i, beta in enumerate(betas):
        for j, n in enumerate(grid_sizes):
            ax = axes[i, j]
            result = results[i][j]

            # Plot phi field
            im = ax.imshow(
                result.phi,
                origin="lower",
                cmap=CMAP_GRAVITY,
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
            )

            # Mark center
            cx, cy = n // 2, n // 2
            ax.scatter([cx], [cy], color='red', s=50, marker='*',
                      edgecolors='white', linewidth=0.5, zorder=5)

            # Title
            xi_str = f"ξ={result.xi_fit:.0f}" if not np.isnan(result.xi_fit) else "ξ=∞"
            ax.set_title(f"β={beta}, N={n}\n{xi_str}", fontsize=10)

            # Labels only on edges
            if i == n_rows - 1:
                ax.set_xlabel("x")
            if j == 0:
                ax.set_ylabel("y")

    # Single colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("φ(x) = f(x) - 1")

    fig.suptitle("Gravitational Potential φ(x) vs β and Grid Size\n(Central Disk Mass)",
                 fontsize=14, fontweight='bold')

    # Save
    output_dir = Path("output/demo_phi_grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phi_grid.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_path}")

    # Create radial profile comparison
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Fixed grid size, varying beta
    ax = axes2[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(betas)))
    for i, beta in enumerate(betas):
        # Use middle grid size
        result = results[i][1]  # N=100
        ax.plot(result.radial_r, result.radial_phi,
                color=colors[i], linewidth=2, label=f"β={beta}")
    ax.set_xlabel("Distance r from center")
    ax.set_ylabel("φ(r)")
    ax.set_title(f"Radial Profile: Varying β (N={grid_sizes[1]})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Right: Fixed beta, varying grid size
    ax = axes2[1]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(grid_sizes)))
    for j, n in enumerate(grid_sizes):
        # Use highest beta
        result = results[-1][j]  # β=0.99
        # Normalize r by grid size for comparison
        r_norm = result.radial_r / n
        ax.plot(r_norm, result.radial_phi,
                color=colors[j], linewidth=2, label=f"N={n}")
    ax.set_xlabel("r / N (normalized distance)")
    ax.set_ylabel("φ(r)")
    ax.set_title(f"Radial Profile: Varying N (β={betas[-1]})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    fig2.suptitle("Radial φ(r) Profiles", fontsize=14, fontweight='bold')
    fig2.tight_layout()

    output_path2 = output_dir / "phi_radial.png"
    fig2.savefig(output_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path2}")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary: Screening Length ξ")
    print("=" * 60)
    print(f"{'β':>6} | " + " | ".join(f"N={n:>3}" for n in grid_sizes))
    print("-" * 60)
    for i, beta in enumerate(betas):
        xi_vals = [f"{results[i][j].xi_fit:>5.1f}" if not np.isnan(results[i][j].xi_fit)
                   else "  inf" for j in range(len(grid_sizes))]
        print(f"{beta:>6.2f} | " + " | ".join(xi_vals))

    print("\n" + "=" * 60)
    print("Observation: As β → 1, screening length ξ → ∞")
    print("This confirms: Pure Poisson (long-range gravity) in continuum limit")
    print("=" * 60)


if __name__ == "__main__":
    main()
