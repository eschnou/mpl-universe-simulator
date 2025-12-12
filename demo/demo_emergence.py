"""
Demo: Verify Poisson Equation Emergence from Bandwidth + Sync Dynamics.

This script demonstrates the key claim of the paper:
The Poisson equation (Lλ)(x) ≈ κ·ρ_act(x) EMERGES from bandwidth-limited
message passing with synchronization pressure.

The demo:
1. Sets up an engine with a Gaussian mass source
2. Runs to steady state
3. Computes the graph Laplacian of λ = 1 - f
4. Shows correlation between (Lλ) and ρ_act
5. Fits the effective coupling constant κ
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mplsim.core import Lattice, LatticeConfig, SourceMap, LoadGeneratorKernel, TheoreticalScheduler, TheoreticalSchedulerConfig
from mplsim.analysis import verify_poisson_emergence, verify_screened_poisson_emergence, compute_radial_profile


def main():
    """Run the Poisson emergence verification demo."""
    np.random.seed(42)  # For reproducibility

    print("=" * 60)
    print("Poisson Equation Emergence Demo")
    print("Verifying: (Lλ) + m²λ ≈ κ · ρ_act(x)  (Screened Poisson)")
    print("=" * 60)

    # Setup engine with TheoreticalScheduler (implements paper's linear λ dynamics directly)
    print("\n1. Setting up engine with TheoreticalScheduler...")
    nx, ny = 80, 80
    beta, gamma = 0.8, 0.1

    config = LatticeConfig(
        nx=nx, ny=ny,
        neighborhood="von_neumann",
        boundary="periodic",
        link_capacity=10.0,
        spatial_sigma=1.0,
    )
    lattice = Lattice(config)

    # Create source map with Gaussian mass
    source_map = SourceMap(ny, nx, background_rate=0.01)
    source_map.add_gaussian_source(cx=nx//2, cy=ny//2, peak_rate=0.5, sigma=8.0)

    # Use TheoreticalScheduler for direct computation of λ = γa + β⟨λ⟩
    kernel = LoadGeneratorKernel(message_size=1.0, sync_required=True)
    scheduler_config = TheoreticalSchedulerConfig(
        gamma=gamma,
        beta=beta,
        lambda_decay=0.0,
        lambda_smoothing=0.3,
    )
    scheduler = TheoreticalScheduler(
        lattice=lattice,
        source_map=source_map,
        kernel=kernel,
        config=scheduler_config,
    )

    print(f"   Grid size: {nx}x{ny}")
    print(f"   Source: Gaussian at center, peak_rate=0.5, sigma=8")
    print(f"   TheoreticalScheduler: β={beta}, γ={gamma}")

    # Run to steady state
    print("\n2. Running engine to steady state...")
    n_warmup = 500
    stats = scheduler.run(n_warmup)
    print(f"   Ran {n_warmup} ticks")
    print(f"   Mean f: {stats['mean_f']:.4f}")
    print(f"   Min f:  {stats['min_f']:.4f}")
    print(f"   Max λ:  {stats['max_lambda']:.4f}")

    # Verify Poisson emergence (both pure and screened)
    print("\n3. Verifying Poisson emergence...")
    result_pure = verify_poisson_emergence(lattice, source_map)
    result_screened = verify_screened_poisson_emergence(lattice, source_map, beta=beta)

    print(f"\n   RESULTS:")
    print(f"   ---------")
    print(f"   Pure Poisson (Lλ = κρ):")
    print(f"     Correlation:         {result_pure.correlation:.4f}")
    print(f"     Fitted κ:            {result_pure.fitted_kappa:.6f}")
    print(f"")
    print(f"   Screened Poisson (Lλ + m²λ = κρ):")
    print(f"     Theoretical m²:      {result_screened.m_squared_theoretical:.4f}")
    print(f"     Best-fit m²:         {result_screened.m_squared_best_fit:.4f}")
    print(f"     Correlation:         {result_screened.correlation_screened:.4f}")
    print(f"     Best-fit corr:       {result_screened.correlation_best:.4f}")

    # Success criterion
    success = result_screened.correlation_best > 0.8
    print(f"\n   Emergence {'CONFIRMED' if success else 'NOT CONFIRMED'}")
    print(f"   (Criterion: best-fit correlation > 0.8)")

    # Use screened result for visualization
    result = result_pure  # Keep for backward compatibility with plots

    # Create visualization
    print("\n4. Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: f field
    ax = axes[0, 0]
    im = ax.imshow(lattice.f, cmap='viridis', origin='lower')
    ax.set_title('f(x) - Update Fraction')
    plt.colorbar(im, ax=ax, label='f')

    # Plot 2: λ field (slowness)
    ax = axes[0, 1]
    im = ax.imshow(result.lambda_field, cmap='magma', origin='lower')
    ax.set_title('λ(x) = 1 - f(x) (Slowness)')
    plt.colorbar(im, ax=ax, label='λ')

    # Plot 3: (Lλ) - Laplacian of slowness
    ax = axes[0, 2]
    vmax = np.percentile(np.abs(result.laplacian_lambda), 95)
    im = ax.imshow(result.laplacian_lambda, cmap='RdBu_r', origin='lower',
                   vmin=-vmax, vmax=vmax)
    ax.set_title('(Lλ)(x) - Graph Laplacian of λ')
    plt.colorbar(im, ax=ax, label='(Lλ)')

    # Plot 4: ρ_act - Activity density
    ax = axes[1, 0]
    im = ax.imshow(result.rho_activity, cmap='hot', origin='lower')
    ax.set_title('ρ_act(x) - Activity Density')
    plt.colorbar(im, ax=ax, label='ρ')

    # Plot 5: Scatter plot - Screened Poisson (Lλ + m²λ) vs ρ
    ax = axes[1, 1]

    # Use screened Poisson: (Lλ) + m²λ vs ρ
    m_sq = result_screened.m_squared_best_fit
    margin = 5
    inner_slice = (slice(margin, -margin), slice(margin, -margin))
    lambda_inner = result_screened.lambda_field[inner_slice].flatten()
    laplacian_inner = result_screened.laplacian_lambda[inner_slice].flatten()
    rho_inner = result_screened.rho_activity[inner_slice].flatten()

    LHS = laplacian_inner + m_sq * lambda_inner  # Screened Poisson LHS

    ax.scatter(rho_inner, LHS, alpha=0.3, s=1)

    # Plot fit line
    slope, intercept = np.polyfit(rho_inner, LHS, 1)
    rho_range = np.array([rho_inner.min(), rho_inner.max()])
    fit_line = slope * rho_range + intercept
    ax.plot(rho_range, fit_line, 'r-', linewidth=2,
            label=f'Fit: κ={slope:.4f}')

    ax.set_xlabel('ρ_act (Activity Density)')
    ax.set_ylabel(f'(Lλ) + {m_sq:.1f}·λ')
    ax.set_title(f'Screened Poisson: r={result_screened.correlation_best:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Radial profiles
    ax = axes[1, 2]
    center = (nx // 2, ny // 2)

    radii_f, values_f = compute_radial_profile(lattice.f, center)
    radii_L, values_L = compute_radial_profile(result.laplacian_lambda, center)
    radii_rho, values_rho = compute_radial_profile(result.rho_activity, center)

    ax.plot(radii_f, values_f, 'b-', label='f(r)', linewidth=2)
    ax.set_xlabel('Radius from source')
    ax.set_ylabel('f(r)', color='b')
    ax.tick_params(axis='y', labelcolor='b')

    ax2 = ax.twinx()
    ax2.plot(radii_rho, values_rho / values_rho.max(), 'r--', label='ρ(r)/max', linewidth=2)
    ax2.set_ylabel('Normalized ρ(r)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax.set_title('Radial Profiles')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.suptitle(f'Screened Poisson Emergence: (Lλ) + m²λ ≈ κρ  (m²={m_sq:.1f}, r={result_screened.correlation_best:.3f})', fontsize=14)
    plt.tight_layout()

    # Save figure
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "poisson_emergence.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_path}")

    plt.show()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
