# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: Read the Research First

**Before making any changes to this codebase, you MUST read and understand the foundational documents:**

### Research Papers
Document: papers/emergent_gravity_from_finite_bandwidth.pdf
**"Emergent Gravity from Finite Bandwidth in a Message-Passing Lattice Universe Engine"** 
   - Defines the gravity mechanism:
   - Finite bandwidth on links → congestion → waiting → slowdown
   - **f(x) = realized_updates / possible_updates** — THIS IS GRAVITY
   - Time-sag field ϕ(x) derived from f(x) via Poisson-like relation: ∇²ϕ = −κρ_act
   - Clocks tick slow where f is low; particles drift toward low-f regions

### Design Documents (in specs/ directory)
- **requirements.md** - Functional requirements for the simulator
- **design.md** - Technical architecture with explicit layer separation
- **tasks.md** - Development phases and milestones

**The code must stay faithful to the papers' logic. This is a physics simulator, not a generic gravity solver.**

## Core Concept

```
Activity → Message traffic → Link congestion → Waiting → Low f(x) → Time dilation
```

**f(x) IS gravity.** Clocks tick slow where f is low. Particles drift toward low-f regions.
No Poisson equations in the engine — just messages and queues. ϕ(x) is a diagnostic tool only.

## Build and Test Commands

```bash
# Install (with dev dependencies)
poetry install

# Run all tests
poetry run pytest

# Run single test file
poetry run pytest tests/unit/test_scheduler.py

# Run single test function
poetry run pytest tests/unit/test_scheduler.py::test_function_name -v

# Run integration tests only
poetry run pytest tests/integration/

# Run demos
poetry run python demo/demo_free_fall.py
poetry run python demo/demo_flyby.py
```

## Architecture (from design.md)

### Layer Structure (information flows ONE WAY: engine → analysis)

**Layer 1: Engine Primitives** (`src/mpl-universe-simulator/core/`)
- `lattice.py` - Lattice, LatticeConfig; contains `f` and `f_smooth` fields
- `scheduler.py` - Tick loop, update eligibility, f(x) computation
- `kernel.py` - LoadGeneratorKernel for message traffic generation
- `source_map.py` - SourceMap defines activity (mass) distributions
- `messages.py` - Message queues and capacity accounting

**Layer 2: Patterns** (`src/mpl-universe-simulator/patterns/`)
- `particle.py` - GeodesicParticle: test particles that move along f gradients
- `clock.py` - ClockPattern: tracks proper time at a location
- `base.py` - Pattern protocol and registry

**Layer 3: Analysis** (`src/mpl-universe-simulator/analysis/`)
- `phi_field.py` - Derive ϕ(x) from measured f(x)
- `poisson.py` - Poisson solver for theoretical comparison
- `comparison.py` - Tools for comparing engine vs Newtonian predictions

**Layer 4: Visualization** (`src/mpl-universe-simulator  /viz/`)
- `fields.py` - Field heatmaps
- `trajectories.py` - Particle trajectory plots

### Key Architectural Boundary

**The `core/` and `patterns/` modules NEVER import from `analysis/`.**

The engine computes f(x) from bandwidth/waiting dynamics. ϕ(x) and Poisson equations are diagnostic tools only — they do NOT feed back into the engine.

### Primitives vs Derived (from design.md section 0.1)

| Primitive (Engine Computes) | Definition |
|-----------------------------|------------|
| `f(x)` | realized_updates / possible_updates — THIS IS GRAVITY |
| `proper_time` τ(x) | Accumulated realized updates at node x |
| `SourceMap` | Defines activity (mass) via output rates |

| Derived (Analysis Tools) | Definition |
|--------------------------|------------|
| `ϕ(x)` | Time-sag field derived from f(x) — diagnostic only |
| `λ(x)` | Slowness = 1 - f(x) |
| `ρ_act(x)` | Activity density for Poisson comparison |

### Two f Fields

- `lattice.f` - Temporally smoothed (EMA) update fraction
- `lattice.f_smooth` - Spatially + temporally smoothed (for physics: gradients, trajectories)

Particles use `f_smooth` for gradient computation to get continuous geodesics.

## Key Physics to Implement Correctly

From the papers, these phenomena must emerge from bandwidth limits:

1. **Time Dilation** - Clocks near sources tick slower (f(x) < 1)
2. **Free Fall** - Particles drift toward low-f regions (∝ ∇f)
3. **Lensing** - Light-like patterns bend toward mass (speed ∝ f(x))
4. **Horizons** - Extreme congestion → f(x) → 0 → frozen proper time
5. **Dark Matter** - Hidden activity contributes to engine-level f(x)

## 2D vs 3D Gravity: Why No Kepler Orbits

**This simulator is 2D.** The dimensional difference has profound implications:

### 2D Gravity (What We Have)
- Poisson equation: ∇²ϕ = ρ → ϕ(r) ∝ **log(r)**
- Force: g(r) = -dϕ/dr ∝ **1/r** (not 1/r²)
- Orbital periods scale as **T² ∝ R²** (not R³)
- Orbits don't close: rosettes instead of ellipses

### 3D Gravity (Kepler's Laws)
- Poisson equation: ∇²ϕ = ρ → ϕ(r) ∝ **1/r**
- Force: g(r) = -dϕ/dr ∝ **1/r²**
- Kepler's Third Law: **T² ∝ R³**
- Closed elliptical orbits

### Implications
- **Do NOT expect Keplerian orbits** from this 2D simulator
- The simulator correctly implements 2D emergent gravity
- To test "Kepler-like" behavior, verify T² ∝ R² scaling instead
- True Kepler orbits would require:
  - Full 3D simulation (computationally expensive), or
  - 2D slice through a 3D potential (non-trivial mapping)

### What IS Valid in 2D
- Time dilation near sources (f(x) < 1)
- Free fall toward mass (particles drift to low-f)
- Gravitational lensing (path bending)
- Horizon formation (f → 0)
- The emergent gravity mechanism itself

The 2D physics is self-consistent and interesting — it just follows 2D laws, not 3D Kepler.

---

## Testing

Integration tests in `tests/integration/`:
- `test_time_dilation.py` - Clocks near sources tick slower
- `test_free_fall.py` - Particles drift toward mass; flyby deflection

Demos produce visualizations in `output/`:
- `free_fall_radial.png`, `free_fall_orbital.png` - Particle trajectories
- `flyby.png` - Gravitational deflection
