"""
Test: Does fitted beta tend to 1 with increasing grid size?

This script tests whether β_fit → 1 as N → ∞, which would indicate
the system approaches pure Poisson (no screening) in the continuum limit.

Key insight: Screening length ξ ~ sqrt(β/(1-β)), so β→1 means ξ→∞.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List

from mplsim.core import (
    Lattice, LatticeConfig, SourceMap, LoadGeneratorKernel,
    BandwidthScheduler, BandwidthSchedulerConfig
)


@dataclass
class ScalingResult:
    """Result from a single grid size test."""
    grid_size: int
    gamma_fit: float
    beta_fit: float
    correlation: float
    xi_fit: float  # Screening length from fitted beta
    mean_f: float
    ticks_run: int


def run_single_size(n: int, beta_config: float = 0.9, warmup_scale: float = 1.0) -> ScalingResult:
    """
    Run emergence test for a single grid size.

    Args:
        n: Grid size (n x n)
        beta_config: Beta in scheduler config
        warmup_scale: Scale factor for warmup ticks (reduce for speed)

    Returns:
        ScalingResult with fitted parameters
    """
    config = LatticeConfig(
        nx=n, ny=n,
        neighborhood="von_neumann",
        boundary="periodic",
        link_capacity=10.0,
        spatial_sigma=1.0,
    )
    lattice = Lattice(config)

    # Source sigma scales with grid size to maintain relative profile
    sigma = n / 10.0  # ~8 for n=80, proportional scaling

    source_map = SourceMap(n, n, background_rate=0.1)
    source_map.add_gaussian_source(cx=n//2, cy=n//2, peak_rate=2.0, sigma=sigma)

    kernel = LoadGeneratorKernel(message_size=1.0, sync_required=True)
    scheduler_config = BandwidthSchedulerConfig(
        canonical_interval=10,
        bandwidth_per_tick=1.0,
        propagation_delay=1,
        f_smoothing_alpha=0.05,
        beta=beta_config,
        message_rate_scale=8.0,
        stochastic_messages=True,
    )
    scheduler = BandwidthScheduler(
        lattice=lattice,
        source_map=source_map,
        kernel=kernel,
        config=scheduler_config,
    )

    # Scale warmup with grid size (larger grids need more ticks to equilibrate)
    # But reduce overall to save CPU
    base_warmup = 3000
    n_warmup = int(base_warmup * warmup_scale * (n / 80.0))

    stats = scheduler.run(n_warmup)

    # Fit β from: λ = γa + β⟨λ⟩
    lambda_sim = 1.0 - lattice.f
    lambda_neighbor_avg = (
        np.roll(lambda_sim, 1, axis=0) +
        np.roll(lambda_sim, -1, axis=0) +
        np.roll(lambda_sim, 1, axis=1) +
        np.roll(lambda_sim, -1, axis=1)
    ) / 4.0

    a_norm = source_map.rates / lattice.config.link_capacity

    margin = max(5, n // 16)  # Scale margin with grid size
    inner = (slice(margin, -margin), slice(margin, -margin))

    lambda_flat = lambda_sim[inner].flatten()
    a_flat = a_norm[inner].flatten()
    neighbor_flat = lambda_neighbor_avg[inner].flatten()

    X = np.column_stack([a_flat, neighbor_flat])
    coeffs, _, _, _ = np.linalg.lstsq(X, lambda_flat, rcond=None)
    gamma_fit, beta_fit = coeffs

    lambda_predicted = gamma_fit * a_flat + beta_fit * neighbor_flat
    corr = np.corrcoef(lambda_flat, lambda_predicted)[0, 1]

    # Compute screening length from fitted beta
    if beta_fit >= 1.0:
        xi_fit = float('inf')
    else:
        xi_fit = np.sqrt(max(0, beta_fit) / (4 * max(0.001, 1 - beta_fit)))

    return ScalingResult(
        grid_size=n,
        gamma_fit=gamma_fit,
        beta_fit=beta_fit,
        correlation=corr,
        xi_fit=xi_fit,
        mean_f=stats['mean_f'],
        ticks_run=n_warmup,
    )


def main():
    """Run beta scaling test across multiple grid sizes."""
    np.random.seed(42)

    print("=" * 60)
    print("Beta Scaling Test: Does β_fit → 1 as grid size → ∞?")
    print("=" * 60)

    # Grid sizes to test (be conservative to avoid blowing up CPU)
    # Start small, increase gradually
    grid_sizes = [40, 60, 80, 100, 120, 140]

    # Scheduler beta (fixed)
    beta_config = 0.9

    # Reduce warmup for speed (0.5 = half the standard warmup)
    warmup_scale = 0.5

    print(f"\nConfig: scheduler β={beta_config}, warmup_scale={warmup_scale}")
    print(f"Grid sizes: {grid_sizes}")
    print()

    results: List[ScalingResult] = []

    for n in grid_sizes:
        print(f"Running N={n}x{n}...", end=" ", flush=True)
        result = run_single_size(n, beta_config=beta_config, warmup_scale=warmup_scale)
        results.append(result)
        print(f"β_fit={result.beta_fit:.4f}, ξ={result.xi_fit:.1f}, "
              f"corr={result.correlation:.4f}, mean_f={result.mean_f:.3f}")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'N':>6} {'β_fit':>8} {'ξ_fit':>8} {'γ_fit':>8} {'corr':>8} {'1-β':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r.grid_size:>6} {r.beta_fit:>8.4f} {r.xi_fit:>8.1f} "
              f"{r.gamma_fit:>8.4f} {r.correlation:>8.4f} {1-r.beta_fit:>10.6f}")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    Ns = [r.grid_size for r in results]
    betas = [r.beta_fit for r in results]
    xis = [r.xi_fit for r in results]
    one_minus_beta = [1 - r.beta_fit for r in results]

    # Plot 1: β_fit vs N
    ax = axes[0]
    ax.plot(Ns, betas, 'bo-', markersize=8, linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', label='β=1 (pure Poisson)')
    ax.axhline(y=beta_config, color='g', linestyle=':', label=f'β_config={beta_config}')
    ax.set_xlabel('Grid Size N')
    ax.set_ylabel('Fitted β')
    ax.set_title('Does β_fit → 1 as N → ∞?')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Screening length vs N
    ax = axes[1]
    ax.plot(Ns, xis, 'go-', markersize=8, linewidth=2)
    ax.plot(Ns, [n/2 for n in Ns], 'r--', label='N/2 (half grid)')
    ax.set_xlabel('Grid Size N')
    ax.set_ylabel('Screening Length ξ')
    ax.set_title('Screening Length vs Grid Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: |1-β| vs 1/N (log-log to check power law)
    ax = axes[2]
    inv_N = [1/n for n in Ns]
    abs_one_minus_beta = [abs(1 - r.beta_fit) for r in results]
    ax.loglog(inv_N, abs_one_minus_beta, 'ro-', markersize=8, linewidth=2)
    ax.set_xlabel('1/N')
    ax.set_ylabel('|1 - β_fit|')
    ax.set_title('Log-log: Does |1-β| ∝ 1/N^α?')
    ax.grid(True, alpha=0.3, which='both')

    # Fit power law: |1-β| ∝ N^(-α) => log|1-β| = -α*log(N) + c
    log_N = np.log(Ns)
    log_abs_one_minus_beta = np.log(abs_one_minus_beta)
    coeffs = np.polyfit(log_N, log_abs_one_minus_beta, 1)
    ax.annotate(f'Slope ≈ {coeffs[0]:.2f}\n|1-β| ∝ N^{coeffs[0]:.2f}',
                xy=(0.05, 0.15), xycoords='axes fraction')

    plt.tight_layout()

    output_dir = Path("output/demo_beta_scaling")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "beta_scaling.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to: {output_path}")

    # Analysis
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    # Check if beta approaches 1 from above or below
    above_one = all(r.beta_fit > 1 for r in results)
    below_one = all(r.beta_fit < 1 for r in results)

    if above_one:
        print(f"β_fit > 1 for all grid sizes (approaches 1 from ABOVE)")
        print(f"  This means neighbor coupling is STRONGER than λ = γa + β⟨λ⟩ predicts")
    elif below_one:
        print(f"β_fit < 1 for all grid sizes")
    else:
        print(f"β_fit crosses 1 at some grid size")

    if coeffs[0] < -0.5:
        print(f"\n✓ |1-β| decreases with N: power law exponent = {coeffs[0]:.2f}")
        print(f"  |1-β| ∝ N^{coeffs[0]:.2f}")
        print(f"  β → 1 as N → ∞ (pure Poisson in continuum limit)")
    else:
        print(f"\n? Weak scaling: power law exponent = {coeffs[0]:.2f}")
        print(f"  May need larger N to see clear scaling")

    # Extrapolate: at what N would |1-β| < 0.001?
    # log|1-β| = slope*log(N) + intercept
    # target: |1-β| = 0.001 => log(0.001) = slope*log(N) + intercept
    intercept = coeffs[1]
    target_N = np.exp((np.log(0.001) - intercept) / coeffs[0])
    print(f"\n  Extrapolation: |1-β| < 0.001 at N ≈ {target_N:.0f}")

    print()


if __name__ == "__main__":
    main()
