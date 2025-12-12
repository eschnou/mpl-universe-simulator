# 2D Message-Passing Gravity Engine — Development Tasks

## Overview

This document defines the iterative development phases for the simulator.
Each phase produces a **working increment** that can be tested independently.

**Guiding principles:**
- Get something running early, then refine
- Test the core mechanism (congestion → f(x) → time dilation) before adding complexity
- Each phase should be completable in a focused work session
- Tests validate the physics, not just the code

---

## Phase 0: Project Scaffolding

### Goal
Set up the project structure so we can start writing code immediately.

### Tasks
- [ ] Create `pyproject.toml` with dependencies (numpy, scipy, matplotlib, pytest)
- [ ] Create directory structure under `src/manyworld/`:
  - [ ] `core/__init__.py`
  - [ ] `patterns/__init__.py`
  - [ ] `analysis/__init__.py`
  - [ ] `experiments/__init__.py`
  - [ ] `viz/__init__.py`
- [ ] Create `tests/` directory with `conftest.py`
- [ ] Verify `pip install -e .` works
- [ ] Verify `pytest` runs (even with no tests yet)

### Validation
```bash
cd simulator
pip install -e .
python -c "import manyworld; print('OK')"
pytest  # Should report "no tests" but not crash
```

---

## Phase 1: Lattice and Message Infrastructure

### Goal
A 2D lattice where nodes can send messages to neighbors, with capacity-limited links.
No congestion effects yet — just the plumbing.

### Tasks
- [ ] `core/lattice.py`:
  - [ ] `LatticeConfig` dataclass (nx, ny, neighborhood, boundary, link_capacity)
  - [ ] `Lattice` class with state array `[ny, nx, n_channels]`
  - [ ] Neighbor lookup methods (get_neighbors, get_neighbor_directions)
  - [ ] Boundary condition handling (periodic, reflective, absorbing)

- [ ] `core/messages.py`:
  - [ ] `Message` dataclass (source, size_bits, tick_created)
  - [ ] `LinkQueues` class managing outbound queues per direction
  - [ ] Methods: `enqueue(direction, message)`, `get_pending(direction)`, `clear_sent()`
  - [ ] Capacity tracking: `bytes_sent_this_tick`, `can_send(size)`

- [ ] `core/source_map.py`:
  - [ ] `SourceMap` class with `rates` array `[ny, nx]`
  - [ ] Methods: `add_point_source()`, `add_gaussian_source()`, `add_ring_source()`
  - [ ] `__add__` for combining source maps

### Validation
```python
# test_lattice.py
def test_lattice_creation():
    cfg = LatticeConfig(nx=50, ny=50, neighborhood="von_neumann",
                        boundary="periodic", link_capacity=10.0)
    lat = Lattice(cfg)
    assert lat.state.shape == (50, 50, 4)  # default channels

def test_neighbor_lookup_periodic():
    # Node at (0,0) should have neighbor at (49, 0) in periodic boundary
    ...

def test_message_queue_capacity():
    q = LinkQueues(ny=10, nx=10, capacity=5.0)
    q.enqueue("N", 0, 0, Message(size_bits=3.0, ...))
    q.enqueue("N", 0, 0, Message(size_bits=3.0, ...))  # Total 6 > 5
    assert q.get_overflow("N", 0, 0) == 1.0  # 1 bit over capacity

def test_source_map_gaussian():
    sm = SourceMap(100, 100)
    sm.add_gaussian_source(50, 50, peak_rate=10.0, sigma=5)
    assert sm.rates[50, 50] > sm.rates[0, 0]  # Peak at center
```

---

## Phase 2: Kernel and Congestion Dynamics

### Goal
Run the engine with `LoadGeneratorKernel`. Nodes generate messages based on `SourceMap`.
Links saturate. Measure `f(x)` — the fraction of realized updates.

**This is the core mechanism. If this works, "gravity" works.**

### Tasks
- [ ] `core/kernel.py`:
  - [ ] `Kernel` protocol with `compute_update()` method
  - [ ] `LoadGeneratorKernel` implementation:
    - Takes `source_map` to determine output rate per node
    - Generates messages with `size_bits = rate * message_size`
    - Returns trivial state update (just increment counter)

- [ ] `core/scheduler.py`:
  - [ ] `Scheduler` class with `tick()` method
  - [ ] Track `possible_updates[ny, nx]` (incremented every tick)
  - [ ] Track `realized_updates[ny, nx]` (incremented when node actually updates)
  - [ ] Compute `f = realized / possible` (with smoothing window)
  - [ ] Track `proper_time[ny, nx]` (sum of realized updates)
  - [ ] Update eligibility check:
    - For v1: node can update if outbound queue from previous tick has cleared
    - (Input-waiting deferred to later phase)

- [ ] `core/proper_time.py`:
  - [ ] Simple module to manage τ(x) accumulation
  - [ ] `increment_proper_time(mask)` — add 1 where mask is True

### Validation
```python
# test_congestion.py
def test_uniform_low_load_f_near_one():
    """With low uniform load, f(x) should be ~1 everywhere."""
    source_map = SourceMap(50, 50, background_rate=0.1)  # Low load
    lat, sched = setup_engine(source_map, link_capacity=10.0)

    for _ in range(100):
        sched.tick()

    assert np.mean(lat.f) > 0.95  # Almost all updates succeed

def test_point_source_creates_f_dip():
    """High-activity source should create local f(x) < 1."""
    source_map = SourceMap(50, 50, background_rate=0.1)
    source_map.add_point_source(25, 25, rate=50.0)  # Hot spot
    lat, sched = setup_engine(source_map, link_capacity=10.0)

    for _ in range(200):
        sched.tick()

    f_at_source = lat.f[25, 25]
    f_far_away = lat.f[0, 0]
    assert f_at_source < f_far_away  # Source region is slower
    assert f_at_source < 0.8  # Noticeably slowed

def test_f_gradient_around_source():
    """f(x) should increase with distance from source."""
    source_map = SourceMap(100, 100, background_rate=0.1)
    source_map.add_gaussian_source(50, 50, peak_rate=100.0, sigma=3)
    lat, sched = setup_engine(source_map, link_capacity=10.0)

    for _ in range(300):
        sched.tick()

    # Sample f at increasing distances from center
    f_r5 = lat.f[50, 55]
    f_r15 = lat.f[50, 65]
    f_r30 = lat.f[50, 80]

    assert f_r5 < f_r15 < f_r30  # f increases with distance
```

**Milestone:** At this point, we have demonstrated that bandwidth limits create a position-dependent slowdown f(x). This IS the gravitational mechanism.

---

## Phase 3: Clock Patterns and Time Dilation

### Goal
Place "clock" patterns at different locations. Show that clocks near sources tick slower than clocks far away.

**This is the first observable gravitational effect.**

### Tasks
- [ ] `patterns/base.py`:
  - [ ] `PatternConfig` dataclass (pattern_id, world_tag, center, extent)
  - [ ] `Pattern` base class with `initialize()`, `update(tick)`, `get_center()`

- [ ] `patterns/clock.py`:
  - [ ] `ClockPattern` class:
    - Tracks `clock_ticks` (how many times host node realized an update)
    - Tracks `readings: list[(canonical_tick, clock_tick)]`
    - `update()` checks if `lattice.proper_time` at host increased

- [ ] `experiments/time_dilation.py`:
  - [ ] Helper to set up time dilation experiment
  - [ ] Place source + multiple clocks at different distances
  - [ ] Run engine, collect clock readings

### Validation
```python
# test_time_dilation.py
def test_clock_near_source_ticks_slower():
    """Clock near source should accumulate fewer ticks than distant clock."""
    source_map = SourceMap(100, 100, background_rate=0.1)
    source_map.add_gaussian_source(50, 50, peak_rate=50.0, sigma=5)

    lat, sched = setup_engine(source_map, link_capacity=10.0)

    clock_near = ClockPattern(center=(55, 50), ...)  # 5 units from source
    clock_far = ClockPattern(center=(90, 50), ...)   # 40 units from source

    for _ in range(500):
        sched.tick()
        clock_near.update(sched.canonical_tick)
        clock_far.update(sched.canonical_tick)

    # After same canonical time, near clock has fewer ticks
    assert clock_near.clock_ticks < clock_far.clock_ticks

    # Ratio should roughly match f ratio
    ratio_clocks = clock_near.clock_ticks / clock_far.clock_ticks
    ratio_f = lat.f[50, 55] / lat.f[50, 90]
    assert abs(ratio_clocks - ratio_f) < 0.1

def test_multiple_clocks_at_different_distances():
    """Clock tick rate should correlate with f(x) at clock position."""
    # Place 5 clocks at r=5, 10, 20, 30, 40 from source
    # Run engine
    # Check that tick_rate increases monotonically with distance
    ...
```

**Milestone:** Gravitational time dilation demonstrated. Clocks in "deep potential wells" (low f regions) run slow.

---

## Phase 4: Analysis Layer and Visualization

### Goal
Compute derived field ϕ_engine from f(x). Visualize f(x) and ϕ_engine as heatmaps.
Set up Poisson solver for theoretical comparison.

### Tasks
- [ ] `analysis/phi_field.py`:
  - [ ] `PhiFromF` class with `compute(f) -> phi_engine`
  - [ ] Support modes: "linear" (ϕ = α(f-1)), "log" (ϕ = α·log(f))

- [ ] `analysis/poisson.py`:
  - [ ] `PoissonSolver` class
  - [ ] Build sparse Laplacian matrix for 2D grid
  - [ ] `solve(rho) -> phi_poisson` using scipy.sparse.linalg
  - [ ] Support periodic and absorbing boundaries

- [ ] `analysis/comparison.py`:
  - [ ] `compare_phi_engine_vs_poisson(lat, source_map)`:
    - Compute ϕ_engine from f
    - Compute ϕ_Poisson from source_map.rates
    - Return comparison metrics (correlation, max error, etc.)

- [ ] `viz/fields.py`:
  - [ ] `plot_field(field, title, cmap)` → heatmap
  - [ ] `plot_f_and_phi(lat)` → side-by-side f and ϕ_engine
  - [ ] `plot_radial_profile(field, center)` → 1D radial plot

### Validation
```python
# test_phi_from_f.py
def test_phi_linear_mode():
    f = np.ones((10, 10)) * 0.8
    phi = PhiFromF(mode="linear", alpha=1.0).compute(f)
    assert np.allclose(phi, -0.2)  # ϕ = f - 1 = -0.2

def test_phi_deeper_where_f_lower():
    f = np.ones((50, 50))
    f[25, 25] = 0.5  # Low f at center
    phi = PhiFromF(mode="linear").compute(f)
    assert phi[25, 25] < phi[0, 0]  # Deeper potential at center

# test_poisson.py
def test_poisson_point_source_radial_falloff():
    """Poisson solution should fall off ~log(r) in 2D."""
    solver = PoissonSolver(boundary="absorbing")
    rho = np.zeros((101, 101))
    rho[50, 50] = 100.0
    phi = solver.solve(rho)

    # Check radial profile decreases with distance
    assert phi[50, 50] < phi[50, 60] < phi[50, 70]

# test_comparison.py
def test_phi_engine_matches_poisson_qualitatively():
    """Engine's ϕ should correlate with Poisson prediction."""
    source_map = SourceMap(100, 100, background_rate=0.1)
    source_map.add_gaussian_source(50, 50, peak_rate=50.0, sigma=5)

    lat, sched = run_engine(source_map, ticks=500)

    phi_engine = PhiFromF().compute(lat.f)
    phi_poisson = PoissonSolver().solve(source_map.rates)

    # Normalize both to [0, 1] and check correlation
    corr = np.corrcoef(phi_engine.flatten(), phi_poisson.flatten())[0, 1]
    assert corr > 0.8  # Strong correlation
```

**Manual validation:** Run notebook, visually inspect that:
- f(x) shows dip around source
- ϕ_engine shows potential well shape
- Radial profiles look reasonable

---

## Phase 5: Test Particles and Free Fall

### Goal
Place test particles in the f(x) gradient. Show they drift toward the source (free fall).

### Tasks
- [ ] `patterns/particle.py`:
  - [ ] `GeodesicParticle` class (geodesic-approximate mode):
    - Position integrated using ∇f (or ∇ϕ_engine)
    - Time step modulated by local f(x)
    - Records trajectory
  - [ ] Later: `NativeWavePacket` for engine-native mode (requires WaveKernel)

- [ ] `analysis/geodesic.py`:
  - [ ] `compute_f_gradient(f, x, y)` → (df/dx, df/dy)
  - [ ] Trajectory integration utilities

- [ ] `viz/trajectories.py`:
  - [ ] `plot_trajectories(particles, background_field)` → trajectory lines over heatmap

### Validation
```python
# test_free_fall.py
def test_particle_drifts_toward_source():
    """Particle released near source should drift inward."""
    source_map = SourceMap(100, 100, background_rate=0.1)
    source_map.add_gaussian_source(50, 50, peak_rate=50.0, sigma=5)

    lat, sched = run_engine(source_map, ticks=300)  # Establish f field

    # Release particle at (70, 50), no initial velocity
    particle = GeodesicParticle(center=(70, 50), velocity=(0, 0), lattice=lat)

    for _ in range(200):
        sched.tick()
        particle.update(sched.canonical_tick)

    final_x, final_y = particle.get_center()
    assert final_x < 70  # Moved toward source (at x=50)

def test_particle_trajectory_curves_around_source():
    """Particle with tangential velocity should curve toward source."""
    # Release particle at (70, 50) with velocity (0, 1) (tangential)
    # Should curve inward, not travel in straight line
    ...

def test_particle_speed_varies_with_f():
    """Particle should move slower in low-f regions."""
    # Track particle speed as it passes through different f regions
    ...
```

**Milestone:** Free fall demonstrated. Particles follow "geodesics" in the emergent geometry.

---

## Phase 6: Light-like Patterns and Lensing

### Goal
Propagate light rays across the lattice. Show deflection (lensing) near sources.

### Tasks
- [ ] `patterns/lightray.py`:
  - [ ] `GeodesicLightRay` class:
    - Propagates at speed ∝ f(x)
    - Bends toward high-f regions (refraction from ∇f)
    - Records trajectory

- [ ] `experiments/lensing.py`:
  - [ ] Setup: source in center, light rays from left edge aimed at detector on right
  - [ ] Measure deflection angles vs impact parameter

- [ ] `viz/lensing.py`:
  - [ ] `plot_lensing_rays(rays, source_position)` → ray diagram

### Validation
```python
# test_lensing.py
def test_light_bends_toward_source():
    """Light ray passing near source should deflect toward it."""
    source_map = SourceMap(100, 100, background_rate=0.1)
    source_map.add_gaussian_source(50, 50, peak_rate=50.0, sigma=5)

    lat, sched = run_engine(source_map, ticks=300)

    # Light ray starting at (10, 60), moving right (+x direction)
    # Passes above the source at y=50
    ray = GeodesicLightRay(center=(10, 60), direction=(1, 0), lattice=lat)

    for _ in range(100):
        ray.update(...)

    final_x, final_y = ray.get_center()
    # Ray should have bent downward (toward source at y=50)
    assert final_y < 60

def test_deflection_increases_with_source_strength():
    """Stronger source should cause more deflection."""
    ...

def test_symmetric_lensing():
    """Rays above and below source should bend symmetrically."""
    ...
```

**Milestone:** Gravitational lensing demonstrated.

---

## Phase 7: Multi-World and Dark Matter

### Goal
Set up multiple "worlds" with separate source maps. Show that hidden-world activity contributes to f(x) and ϕ_engine, creating a "dark matter" effect.

### Tasks
- [ ] `worlds/tags.py`:
  - [ ] `WorldConfig` dataclass (world_id, name, color, source_map)
  - [ ] `WorldManager` class:
    - Register worlds
    - `get_engine_source_map()` → combined
    - `get_rho_visible(world_id)` → single world's rates

- [ ] `analysis/dark_matter.py`:
  - [ ] `DarkMatterAnalyzer` class:
    - `analyze(lat, observer_world)` → comparison dict
    - Computes ρ_visible, ρ_engine, ρ_dark
    - Computes ϕ_engine, ϕ_expected_visible, ϕ_dark

- [ ] `experiments/dark_matter.py`:
  - [ ] Standard setup: visible galaxy + hidden halo
  - [ ] Rotation curve analysis (if time permits)

- [ ] `viz/dark_matter.py`:
  - [ ] `plot_dark_matter_comparison(results)` → multi-panel figure

### Validation
```python
# test_dark_matter.py
def test_hidden_world_affects_f():
    """Activity in hidden world should contribute to f(x)."""
    wm = WorldManager(100, 100)

    # Visible world: central source
    source_A = SourceMap(100, 100, background_rate=0.1)
    source_A.add_gaussian_source(50, 50, peak_rate=20.0, sigma=5)
    wm.register_world(WorldConfig(0, "visible", "blue", source_A))

    # Hidden world: extended halo
    source_B = SourceMap(100, 100, background_rate=0.1)
    source_B.add_ring_source(50, 50, radius=25, width=8, rate=10.0)
    wm.register_world(WorldConfig(1, "hidden", "gray", source_B))

    # Run engine on combined sources
    engine_source = wm.get_engine_source_map()
    lat, sched = run_engine(engine_source, ticks=500)

    # f should be affected by hidden halo
    f_at_halo = lat.f[50, 75]  # In the halo region

    # Compare to visible-only run
    lat_visible, _ = run_engine(source_A, ticks=500)
    f_visible_only = lat_visible.f[50, 75]

    assert f_at_halo < f_visible_only  # Hidden world slowed things down

def test_dark_matter_analysis():
    """DarkMatterAnalyzer should detect hidden contribution."""
    # Setup as above
    analyzer = DarkMatterAnalyzer(wm, PoissonSolver())
    results = analyzer.analyze(lat, observer_world=0)

    # ϕ_engine should be deeper than ϕ_expected_visible
    phi_gap = results["phi_engine"] - results["phi_expected_visible"]
    assert np.mean(phi_gap) < 0  # Engine shows more "gravity" than visible predicts

    # Dark fraction should be significant
    assert results["dark_mass_fraction"] > 0.2
```

**Milestone:** Dark matter effect demonstrated. Hidden activity gravitates.

---

## Phase 8: Horizons and Extreme Regimes

### Goal
Create regions of extreme congestion where f(x) → 0. Show horizon-like behavior: proper time freezes, signals can't escape.

### Tasks
- [ ] `experiments/horizons.py`:
  - [ ] Setup: very high activity core
  - [ ] Place clocks inside and outside
  - [ ] Send light rays toward core, measure if they escape

- [ ] Tune parameters to achieve f(x) ≈ 0 in core region

### Validation
```python
# test_horizons.py
def test_extreme_source_freezes_proper_time():
    """Very high activity should drive f(x) → 0."""
    source_map = SourceMap(100, 100, background_rate=0.1)
    source_map.add_gaussian_source(50, 50, peak_rate=500.0, sigma=3)  # Extreme

    lat, sched = run_engine(source_map, ticks=500, link_capacity=5.0)

    f_core = lat.f[50, 50]
    assert f_core < 0.1  # Nearly frozen

def test_clock_in_horizon_region_barely_ticks():
    """Clock inside extreme region should tick very slowly."""
    # Place clock at source center
    # Compare to clock far away
    # Ratio should be very small
    ...

def test_light_trapped_near_horizon():
    """Light ray aimed at extreme source should slow dramatically."""
    # Track light ray approaching the "horizon"
    # It should slow down and potentially never reach center
    ...
```

**Milestone:** Horizon-like behavior demonstrated.

---

## Phase 9: Experiment Harness and Scenarios

### Goal
Clean API for running experiments. Pre-built scenarios for all the standard tests.

### Tasks
- [ ] `experiments/harness.py`:
  - [ ] `ExperimentConfig` dataclass
  - [ ] `ExperimentRunner` class with `setup()`, `run()`, `export_results()`

- [ ] `experiments/scenarios.py`:
  - [ ] `Scenarios.time_dilation_test()` → config
  - [ ] `Scenarios.free_fall_test()` → config
  - [ ] `Scenarios.lensing_test()` → config
  - [ ] `Scenarios.dark_matter_test()` → config
  - [ ] `Scenarios.horizon_test()` → config

- [ ] `experiments/observables.py`:
  - [ ] Standard measurement utilities
  - [ ] Export to CSV/JSON

### Validation
```python
# test_harness.py
def test_time_dilation_scenario():
    config = Scenarios.time_dilation_test(grid_size=100, source_strength=50.0)
    runner = ExperimentRunner(config)
    runner.setup()
    runner.run()
    results = runner.export_results()

    assert "clock_readings" in results
    assert len(results["clock_readings"]) > 0
```

---

## Phase 10: Notebooks and Documentation

### Goal
Jupyter notebooks demonstrating each phenomenon. Polish for presentation.

### Tasks
- [ ] `notebooks/01_basic_demo.ipynb`:
  - Engine setup, run, visualize f(x)

- [ ] `notebooks/02_time_dilation.ipynb`:
  - Clock experiment, plot τ vs T_can for multiple clocks

- [ ] `notebooks/03_free_fall.ipynb`:
  - Particle trajectories, comparison with Newtonian prediction

- [ ] `notebooks/04_lensing.ipynb`:
  - Light ray deflection, ray diagrams

- [ ] `notebooks/05_dark_matter.ipynb`:
  - Multi-world setup, dark matter analysis, rotation curves

- [ ] `notebooks/06_horizons.ipynb`:
  - Extreme regime, frozen clocks, trapped light

- [ ] Update `README.md` with quickstart

### Validation
- All notebooks run without errors
- Figures look reasonable and match paper predictions
- Someone unfamiliar with the code can follow along

---

## Summary: Milestone Checklist

| Phase | Milestone | Key Validation |
|-------|-----------|----------------|
| 0 | Project runs | `import manyworld` works |
| 1 | Lattice + messages | Queue capacity tests pass |
| 2 | **Congestion → f(x)** | f drops near sources |
| 3 | **Time dilation** | Clocks near source tick slow |
| 4 | Analysis + viz | ϕ_engine correlates with Poisson |
| 5 | **Free fall** | Particles drift toward source |
| 6 | **Lensing** | Light rays bend near source |
| 7 | **Dark matter** | Hidden world affects ϕ_engine |
| 8 | Horizons | f → 0 in extreme regions |
| 9 | Harness | Scenarios API works |
| 10 | Notebooks | Demos ready for presentation |

**Phases 2, 3, 5, 6, 7 are the physics milestones.** Everything else is infrastructure.

---

## Estimated Effort

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| 0 | 1 hour | None |
| 1 | 2-3 hours | Phase 0 |
| 2 | 3-4 hours | Phase 1 |
| 3 | 2 hours | Phase 2 |
| 4 | 2-3 hours | Phase 2 |
| 5 | 2 hours | Phase 4 |
| 6 | 2 hours | Phase 4 |
| 7 | 3 hours | Phase 2 |
| 8 | 2 hours | Phase 3 |
| 9 | 2-3 hours | Phases 3-8 |
| 10 | 3-4 hours | Phase 9 |

**Total: ~25-30 hours for complete implementation**

Critical path: 0 → 1 → 2 → 3 (time dilation demo in ~8 hours)
