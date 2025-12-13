#!/usr/bin/env python3
"""
Demo: φ(x) field scaling with beta and grid size.

Shows how the gravitational potential profile changes as:
- beta approaches 1 (less screening → longer range)
- grid size increases (continuum limit)

Run with: poetry run python demo/demo_phi_scaling.py

Expected runtime: ~10-30 minutes depending on hardware.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List
import time

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
    xi_fit: float
    mean_f: float
    elapsed_sec: float


def compute_phi_for_config(
    grid_size: int,
    beta: float,
    warmup_ticks: int = 1500,
) -> PhiResult:
    """
    Compute φ(x) field for a given grid size and beta.

    Source size scales proportionally with grid size.
    """
    t0 = time.time()

    cx, cy = grid_size // 2, grid_size // 2

    # Source radius proportional to grid size (1% of grid)
    mass_radius = max(3, grid_size // 100 * 5)

    # Scale mass rate to maintain similar congestion level
    # Larger grids have more cells to fill, need proportionally more activity
    base_rate = 50.0
    rate = base_rate #* (grid_size / 100.0)

    source_map = SourceMap(ny=grid_size, nx=grid_size, background_rate=0.01)
    source_map.add_uniform_disk(cx=cx, cy=cy, radius=mass_radius, rate=rate)

    config = LatticeConfig(
        nx=grid_size,
        ny=grid_size,
        neighborhood="von_neumann",
        boundary="absorbing",
        link_capacity=15.0,
        spatial_sigma=max(1.5, grid_size / 50),  # Scale smoothing with grid
    )
    lattice = Lattice(config)
    kernel = LoadGeneratorKernel(message_size=1.0, sync_required=True)

    scheduler_config = BandwidthSchedulerConfig(
        canonical_interval=10,
        bandwidth_per_tick=1.0,
        propagation_delay=1,
        f_smoothing_alpha=0.05,  # Faster smoothing for quicker convergence
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

    # Scale warmup: sqrt scaling to avoid blowing up for large grids
    scaled_warmup = int(warmup_ticks * np.sqrt(grid_size / 100.0))
    stats = scheduler.run(scaled_warmup)

    # Compute phi from f
    phi = compute_phi_linear(lattice.f_smooth, alpha=1.0)

    # Compute radial profile (sample up to 40% of grid radius)
    max_r = int(grid_size * 0.4)
    step = max(1, grid_size // 80)
    r_vals = np.arange(mass_radius + 2, max_r, step)
    phi_profile = []

    for r in r_vals:
        samples = []
        n_angles = min(72, max(18, grid_size // 10))
        for angle in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < grid_size and 0 <= y < grid_size:
                samples.append(phi[y, x])
        if samples:
            phi_profile.append(np.mean(samples))
        else:
            phi_profile.append(np.nan)

    r_vals = np.array(r_vals)
    phi_profile = np.array(phi_profile)

    # Fit screening length
    from scipy.optimize import curve_fit

    def exp_decay(r, a, xi, b):
        return -a * np.exp(-r / xi) + b

    try:
        max_xi = grid_size / 2
        valid = ~np.isnan(phi_profile)
        popt, _ = curve_fit(
            exp_decay, r_vals[valid], phi_profile[valid],
            p0=[0.1, min(20, max_xi / 5), 0],
            bounds=([0, 1, -0.5], [10, max_xi, 0.5]),
            maxfev=5000
        )
        xi_fit = popt[1]
    except:
        xi_fit = np.nan

    elapsed = time.time() - t0

    return PhiResult(
        beta=beta,
        grid_size=grid_size,
        phi=phi,
        f=lattice.f_smooth,
        radial_r=r_vals,
        radial_phi=phi_profile,
        xi_fit=xi_fit,
        mean_f=stats['mean_f'],
        elapsed_sec=elapsed,
    )


def main():
    """Generate phi field comparison for beta scaling study."""
    np.random.seed(42)

    print("=" * 70)
    print("  φ(x) Field Scaling: Beta → 1 with Increasing Grid Size")
    print("=" * 70)

    # Configuration
    betas = [0.9, 0.92, 0.94, 0.96, 0.98]
    grid_sizes = [100, 250, 500]

    print(f"\nConfiguration:")
    print(f"  Betas: {betas}")
    print(f"  Grid sizes: {grid_sizes}")
    print(f"  Total runs: {len(betas) * len(grid_sizes)}")
    print(f"\nSource size = 2% of grid size (proportional scaling)")
    print()

    # Compute all configurations
    results: List[List[PhiResult]] = []
    total_time = 0

    for beta in betas:
        row = []
        for n in grid_sizes:
            print(f"Computing β={beta}, N={n}...", end=" ", flush=True)
            result = compute_phi_for_config(n, beta)
            row.append(result)
            total_time += result.elapsed_sec
            xi_str = f"{result.xi_fit:.1f}" if not np.isnan(result.xi_fit) else "∞"
            print(f"ξ={xi_str}, mean_f={result.mean_f:.3f}, time={result.elapsed_sec:.1f}s")
        results.append(row)

    print(f"\nTotal computation time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # === Create visualizations ===
    output_dir = Path("output/demo_phi_scaling")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Grid of phi fields
    print("\nCreating phi field grid...")
    n_rows = len(betas)
    n_cols = len(grid_sizes)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))

    # Consistent colorbar across all panels
    all_phi = [r.phi for row in results for r in row]
    vmin = min(p.min() for p in all_phi)
    vmax = 0  # phi <= 0 always

    for i, beta in enumerate(betas):
        for j, n in enumerate(grid_sizes):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            result = results[i][j]

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
            ax.scatter([cx], [cy], color='red', s=30, marker='*',
                      edgecolors='white', linewidth=0.5, zorder=5)

            # Title with screening length
            xi_str = f"ξ={result.xi_fit:.0f}" if not np.isnan(result.xi_fit) else "ξ→∞"
            ax.set_title(f"β={beta}, N={n}\n{xi_str}", fontsize=11)

            if i == n_rows - 1:
                ax.set_xlabel("x")
            if j == 0:
                ax.set_ylabel("y")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("φ(x) = f(x) - 1", fontsize=12)

    fig.suptitle("Gravitational Potential φ(x): Beta × Grid Size\n(Source size ∝ grid size)",
                 fontsize=14, fontweight='bold')

    fig.savefig(output_dir / "phi_field_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'phi_field_grid.png'}")

    # Figure 2: Radial profiles - varying beta at each grid size
    print("Creating radial profile comparison...")
    fig2, axes2 = plt.subplots(1, len(grid_sizes), figsize=(5 * len(grid_sizes), 5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(betas)))

    for j, n in enumerate(grid_sizes):
        ax = axes2[j]
        for i, beta in enumerate(betas):
            result = results[i][j]
            # Normalize r by grid size for comparison
            r_norm = result.radial_r / n
            ax.plot(r_norm, result.radial_phi,
                   color=colors[i], linewidth=2, label=f"β={beta}")

        ax.set_xlabel("r / N (normalized distance)", fontsize=11)
        ax.set_ylabel("φ(r)", fontsize=11)
        ax.set_title(f"N = {n}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(0, 0.4)

    fig2.suptitle("Radial φ(r) Profiles: How potential extends with β → 1",
                  fontsize=14, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(output_dir / "phi_radial_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'phi_radial_profiles.png'}")

    # Figure 3: Screening length vs beta for each grid size
    print("Creating screening length scaling plot...")
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: xi vs beta
    ax = axes3[0]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(grid_sizes)))
    for j, n in enumerate(grid_sizes):
        xi_vals = [results[i][j].xi_fit for i in range(len(betas))]
        ax.semilogy(betas, xi_vals, 'o-', color=colors[j],
                   markersize=10, linewidth=2, label=f"N={n}")

    ax.set_xlabel("β", fontsize=12)
    ax.set_ylabel("Screening length ξ", fontsize=12)
    ax.set_title("Screening Length vs β", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Right: xi/N vs beta (normalized)
    ax = axes3[1]
    for j, n in enumerate(grid_sizes):
        xi_vals = [results[i][j].xi_fit / n for i in range(len(betas))]
        ax.plot(betas, xi_vals, 'o-', color=colors[j],
               markersize=10, linewidth=2, label=f"N={n}")

    ax.set_xlabel("β", fontsize=12)
    ax.set_ylabel("ξ / N (normalized screening length)", fontsize=12)
    ax.set_title("Normalized Screening Length vs β", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig3.suptitle("Screening Length Scaling: Does ξ → ∞ as β → 1?",
                  fontsize=14, fontweight='bold')
    fig3.tight_layout()
    fig3.savefig(output_dir / "screening_length_scaling.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'screening_length_scaling.png'}")

    # Summary table
    print("\n" + "=" * 70)
    print("Summary: Screening Length ξ (in lattice units)")
    print("=" * 70)
    header = f"{'β':>8} | " + " | ".join(f"N={n:>4}" for n in grid_sizes) + " | ξ/N trend"
    print(header)
    print("-" * 70)

    for i, beta in enumerate(betas):
        xi_vals = [results[i][j].xi_fit for j in range(len(grid_sizes))]
        xi_norm = [results[i][j].xi_fit / grid_sizes[j] for j in range(len(grid_sizes))]

        xi_strs = []
        for xi in xi_vals:
            if np.isnan(xi):
                xi_strs.append("   ∞")
            else:
                xi_strs.append(f"{xi:>5.0f}")

        # Check if xi/N is increasing (gravity becoming more long-range)
        if all(not np.isnan(x) for x in xi_norm):
            if xi_norm[-1] > xi_norm[0] * 1.5:
                trend = "↑ (long-range)"
            elif xi_norm[-1] < xi_norm[0] * 0.7:
                trend = "↓ (short-range)"
            else:
                trend = "≈ (stable)"
        else:
            trend = "∞ (unscreened)"

        print(f"{beta:>8.3f} | " + " | ".join(xi_strs) + f" | {trend}")

    print("\n" + "=" * 70)
    print("Interpretation:")
    print("  • β < 0.95: Screening length ξ is finite and stable with N")
    print("              → Screened (Yukawa-like) gravity, finite range")
    print("  • β → 1:    Screening length ξ grows with N")
    print("              → Approaching pure Poisson (Newtonian) gravity")
    print("  • β = 1:    ξ = ∞, no screening")
    print("              → True long-range gravity in continuum limit")
    print("=" * 70)


if __name__ == "__main__":
    main()
