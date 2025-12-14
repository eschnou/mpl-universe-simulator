#!/usr/bin/env python3
"""
Demo: Radial Profile of Emergent Gravity Field

Simple demonstration showing how a central mass creates a gravity well:
1. Set up a central mass (uniform disk)
2. Run the engine to steady state
3. Plot radial profiles of f(r) and λ(r)
4. Fit the effective β from the steady-state equation

This is the simplest test of emergent gravity from bandwidth limits.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mplsim.core import (
    Lattice, LatticeConfig, SourceMap, LoadGeneratorKernel,
    BandwidthScheduler, BandwidthSchedulerConfig
)
from mplsim.viz.fields import CMAP_GRAVITY


def compute_radial_profile(field: np.ndarray, center: tuple[int, int], max_r: int = None):
    """Compute azimuthally-averaged radial profile."""
    cy, cx = center
    ny, nx = field.shape
    if max_r is None:
        max_r = min(cx, cy, nx - cx, ny - cy) - 1

    r_vals = np.arange(1, max_r)
    profile = []

    for r in r_vals:
        samples = []
        n_angles = max(36, int(2 * np.pi * r))  # More samples at larger r
        for angle in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < nx and 0 <= y < ny:
                samples.append(field[y, x])
        profile.append(np.mean(samples) if samples else np.nan)

    return r_vals, np.array(profile)


def fit_effective_beta(lambda_field: np.ndarray, local_stall_field: np.ndarray) -> float:
    """
    Fit the effective β from the steady-state equation:
        λ(x) = local_stall(x) + β × ⟨λ⟩_neighbors(x)

    Rearranged: β = (λ - local_stall) / ⟨λ⟩_neighbors

    We fit this over all nodes where local_stall ≈ 0 (outside mass).
    """
    # Compute neighbor average of lambda
    neighbor_avg = (
        np.roll(lambda_field, 1, axis=0) + np.roll(lambda_field, -1, axis=0) +
        np.roll(lambda_field, 1, axis=1) + np.roll(lambda_field, -1, axis=1)
    ) / 4.0

    # Only fit where local_stall is negligible and neighbor_avg > 0
    mask = (local_stall_field < 0.01) & (neighbor_avg > 0.001) & (lambda_field > 0.001)

    if mask.sum() < 10:
        return np.nan

    # β = (λ - local_stall) / ⟨λ⟩_neighbors
    # Since local_stall ≈ 0 in masked region: β ≈ λ / ⟨λ⟩_neighbors
    beta_estimates = lambda_field[mask] / neighbor_avg[mask]

    # Return median (robust to outliers)
    return float(np.median(beta_estimates))


def main():
    np.random.seed(42)

    print("=" * 60)
    print("  RADIAL PROFILE OF EMERGENT GRAVITY")
    print("=" * 60)

    # Grid setup
    grid_size = 150
    cx, cy = grid_size // 2, grid_size // 2

    # Mass parameters
    mass_radius = 10
    mass_rate = 1.0

    print(f"\n1. Setup:")
    print(f"   Grid: {grid_size}x{grid_size}")
    print(f"   Mass: disk at center, radius={mass_radius}, rate={mass_rate}")

    # Create source map
    source_map = SourceMap(ny=grid_size, nx=grid_size, background_rate=0.01)
    source_map.add_uniform_disk(cx=cx, cy=cy, radius=mass_radius, rate=mass_rate)

    # Create lattice
    config = LatticeConfig(
        nx=grid_size,
        ny=grid_size,
        neighborhood="von_neumann",
        boundary="absorbing",
        link_capacity=10.0,
        spatial_sigma=2.0,
    )
    lattice = Lattice(config)
    kernel = LoadGeneratorKernel(message_size=1.0, sync_required=True)

    # Scheduler with damping for sync coupling
    # send_interval = max(local_time, avg_neighbor_gap × damping)
    # base_interval=10 gives better temporal resolution for gap measurement
    damping = 1.0
    scheduler_config = BandwidthSchedulerConfig(
        bandwidth=8.0,       # Data units per base_interval
        data_scale=50.0,      # Mass rate=1.0 → mean data=8 → local_time=10×8/8=10
        damping=damping,
        base_interval=50.0,  # f=1 means 1 update per 10 ticks
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

    # Compute theoretical stalling probabilities
    print("\n2. Bandwidth stalling analysis (theoretical)...")
    from scipy.stats import poisson

    capacity = scheduler_config.bandwidth
    scale = scheduler_config.data_scale

    # For each activity level, compute P(message_size > capacity)
    activity_levels = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, mass_rate]
    activity_levels = sorted(set(activity_levels))

    print(f"   Link capacity: {capacity}")
    print(f"   Message size ~ Poisson(activity × {scale})")
    print()
    print(f"   {'Activity':>10} | {'Mean msg':>10} | {'P(msg > cap)':>12}")
    print(f"   {'-'*10}-+-{'-'*10}-+-{'-'*12}")

    for act in activity_levels:
        mean_msg = act * scale
        p_exceed = 1.0 - poisson.cdf(capacity, mean_msg)
        label = f"{act:.2f}"
        if act == mass_rate:
            label += " (mass)"
        print(f"   {label:>10} | {mean_msg:>10.1f} | {p_exceed*100:>11.2f}%")

    # Run to steady state
    print("\n3. Running to steady state...")
    n_ticks = 10000
    stats = scheduler.run(n_ticks)
    print(f"   {n_ticks} ticks completed")
    print(f"   Mean f: {stats['mean_f']:.4f}")
    print(f"   Min f:  {stats['min_f']:.4f}")

    # Compute radial profiles
    print("\n4. Computing radial profiles...")
    r_vals, f_profile = compute_radial_profile(lattice.f_smooth, (cy, cx))
    lambda_profile = 1.0 - f_profile

    # Fit exponential decay (screened Poisson) - only outside mass
    from scipy.optimize import curve_fit

    def exp_decay_from_surface(r, lambda_surface, xi):
        return lambda_surface * np.exp(-(r - mass_radius) / xi)

    mask = (r_vals > mass_radius) & (lambda_profile > 0.001)
    try:
        lambda_at_surface = lambda_profile[r_vals >= mass_radius][0] if any(r_vals >= mass_radius) else 0.1
        popt, _ = curve_fit(
            exp_decay_from_surface, r_vals[mask], lambda_profile[mask],
            p0=[lambda_at_surface, 10],
            bounds=([0, 1], [2, 100]),
            maxfev=5000
        )
        lambda_surface_fit, xi_fit = popt
        fit_curve = exp_decay_from_surface(r_vals, *popt)
        fit_curve[r_vals < mass_radius] = np.nan

        r2 = 1 - np.sum((lambda_profile[mask] - exp_decay_from_surface(r_vals[mask], *popt))**2) / \
                 np.sum((lambda_profile[mask] - lambda_profile[mask].mean())**2)
        print(f"   Fit: λ(r) = {lambda_surface_fit:.3f} × exp(-(r-{mass_radius})/{xi_fit:.1f})")
        print(f"   R² = {r2:.4f}")
        print(f"   Screening length ξ = {xi_fit:.1f} lattice units")
        print(f"   λ at surface = {lambda_surface_fit:.3f}")
    except Exception as e:
        xi_fit = None
        fit_curve = None
        lambda_surface_fit = None
        print(f"   Fit failed: {e}")

    # Fit effective β from the paper's equation
    print("\n5. Fitting effective β from steady-state equation...")
    print("   λ(x) = local_stall(x) + β × ⟨λ⟩_neighbors(x)")

    # Create local stall field (where activity causes stalls)
    # local_stall > 0 inside mass, ≈ 0 outside
    local_stall_field = np.zeros_like(lattice.f)
    yy, xx = np.ogrid[:grid_size, :grid_size]
    dist_from_center = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    inside_mass = dist_from_center <= mass_radius
    # Estimate local stall from activity
    from scipy.stats import poisson as poisson_dist
    local_stall_field[inside_mass] = 1.0 - poisson_dist.cdf(capacity, mass_rate * scale)

    lambda_field = 1.0 - lattice.f_smooth
    effective_beta = fit_effective_beta(lambda_field, local_stall_field)

    print(f"   Input damping: {damping}")
    print(f"   Fitted effective β:   {effective_beta:.3f}")
    if not np.isnan(effective_beta):
        print(f"   Ratio (fitted/input): {effective_beta / damping:.2f}")

    # Create visualization
    print("\n6. Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Activity map
    ax = axes[0, 0]
    im = ax.imshow(source_map.rates, origin="lower", cmap="hot")
    circle = plt.Circle((cx, cy), mass_radius, fill=False, color='cyan', linewidth=2)
    ax.add_patch(circle)
    ax.set_title(f"Activity Map (Mass r={mass_radius})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="Activity rate")

    # Panel 2: f field
    ax = axes[0, 1]
    im = ax.imshow(lattice.f_smooth, origin="lower", cmap=CMAP_GRAVITY)
    ax.scatter([cx], [cy], color='red', s=100, marker='*', edgecolors='white', zorder=5)
    ax.set_title("Emergent f(x) Field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="f(x)")

    # Panel 3: f(r) radial profile (only outside mass)
    ax = axes[1, 0]
    outside_mask = r_vals >= mass_radius
    ax.plot(r_vals[outside_mask], f_profile[outside_mask], 'b-', linewidth=2, label='f(r)')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Distance r from mass surface")
    ax.set_ylabel("f(r)")
    ax.set_title("Radial Profile: f(r) outside mass")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(mass_radius, max(r_vals))
    ax.set_ylim(0, 1.05)

    # Panel 4: λ(r) radial profile with fit and theoretical 2D gravity
    ax = axes[1, 1]
    outside_mask = r_vals >= mass_radius
    r_outside = r_vals[outside_mask]
    lambda_outside = lambda_profile[outside_mask]

    ax.plot(r_outside, lambda_outside, 'ko-', markersize=3, label='λ(r) simulated')

    # Theoretical 2D unscreened gravity: λ ∝ log(r_ref/r)
    # Normalized to match simulation at mass surface
    if len(lambda_outside) > 0 and lambda_outside[0] > 0:
        lambda_surface = lambda_outside[0]
        r_ref = max(r_vals) * 2  # Reference radius (arbitrary, affects normalization)
        lambda_2d_theory = lambda_surface * np.log(r_ref / r_outside) / np.log(r_ref / mass_radius)
        lambda_2d_theory = np.maximum(lambda_2d_theory, 0)  # Clip negative values
        ax.plot(r_outside, lambda_2d_theory, 'b--', linewidth=2,
                label='2D Newtonian: λ ∝ log(1/r)')

    # Exponential fit (screened)
    if fit_curve is not None:
        fit_outside = fit_curve[outside_mask]
        ax.plot(r_outside, fit_outside, 'r-', linewidth=2,
                label=f'Screened fit: ξ={xi_fit:.0f}')

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Distance r from center")
    ax.set_ylabel("λ(r) = 1 - f(r)")
    beta_str = f"β_eff={effective_beta:.2f}" if not np.isnan(effective_beta) else "β_eff=?"
    ax.set_title(f"Radial Profile: λ(r) outside mass ({beta_str})")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(mass_radius, max(r_vals))

    fig.suptitle(f"Emergent Gravity from Bandwidth Limits\ndamping={damping}, β_eff={effective_beta:.2f}, Mass rate={mass_rate}",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    # Save
    output_dir = Path("output/demo_radial")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "radial_profile.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  • Central mass creates congestion → low f near center")
    print(f"  • λ(r) = 1 - f(r) is the 'slowness' field (gravity analog)")
    print(f"  • Screened Poisson: λ(r) ∝ exp(-r/ξ) with ξ ≈ {xi_fit:.0f}" if xi_fit else "  • Fit failed")
    print(f"  • f recovers to ~1 at r ≈ {mass_radius + 5*xi_fit:.0f} (surface + 5ξ)" if xi_fit else "")
    print()
    print(f"  • Input damping: {damping}")
    print(f"  • Fitted effective β:   {effective_beta:.3f}" if not np.isnan(effective_beta) else "  • β fit failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
