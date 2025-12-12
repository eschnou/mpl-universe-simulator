# A Message-Passing Lattice Universe Simulator

A 2D simulator exploring emergent gravity in a message-passing lattice universe engine.

---

## The Big Picture

What if we asked: *what's the cheapest way to run a universe?*

This project explores a computational approach to physics foundations. We start with a 
simple substrate—a lattice of nodes exchanging messages under finite bandwidth constraints—and 
ask what phenomena emerge when we don't prevent them.

A companion paper develop the idea:

**[Emergent Gravity from Finite Bandwidth in a Message-Passing Lattice](papers/emergent_gravity_from_finite_bandwidth.pdf)**
> Add bandwidth limits to the lattice. Busy regions saturate their links, forcing nodes to wait. 
> This creates a position-dependent slowdown—a "time-sag field"—from which gravitational phenomenology 
> emerges: time dilation, free fall and horizons like behavior.

---

## Core Idea

```
Activity → Message traffic → Link congestion → Waiting → Low f(x) → Gravity
```

Nodes on a 2D lattice exchange messages. Links have finite capacity. High-activity regions saturate their links, causing neighbors to wait. This waiting shows up as:

- **f(x) < 1** — the fraction of "possible updates" that actually happen
- **Time dilation** — clocks in congested regions tick slower
- **Geodesic motion** — particles drift toward low-f regions (free fall)

No Poisson equations in the engine. Gravity emerges from queuing dynamics.

---

## Current Status

**Completed (Phases 0–5):**

| Phase | What Works |
|-------|------------|
| Lattice infrastructure | 2D grid, periodic boundaries, message queues |
| Congestion dynamics | f(x) drops near high-activity sources |
| Time dilation | Clocks near sources tick slower |
| Analysis tools | ϕ-field derivation, Poisson comparison |
| Free fall | Particles drift toward mass, trajectories curve |
| Visualization | Field heatmaps, trajectory plots |

---

## Installation

```bash
cd simulator
poetry install        # or: pip install -e ".[dev]"
```

## Running the Demos

```bash
# Free fall: particles released around a central mass
poetry run python demo/demo_free_fall.py

# Flyby: gravitational deflection of passing particles
poetry run python demo/demo_flyby.py

# Poisson emergence verification
poetry run python demo/demo_emergence.py
```

## Running the Tests

```bash
# All tests
poetry run pytest

# Just unit tests
poetry run pytest tests/unit/

# Just integration tests (physics validation)
poetry run pytest tests/integration/

# Specific test file
poetry run pytest tests/integration/test_free_fall.py -v
```

---

## Architecture

```
src/mplsim/
├── core/                       # Engine primitives (THE physics)
│   ├── lattice.py              # 2D grid with f(x) field
│   ├── theoretical_scheduler.py # TheoreticalScheduler (λ = γa + β⟨λ⟩)
│   ├── kernel.py               # LoadGeneratorKernel (message traffic)
│   ├── source_map.py           # Activity distributions (mass)
│   └── messages.py             # Queue and capacity management
│
├── patterns/                   # Things that live in the engine
│   ├── particle.py             # GeodesicParticle (test masses)
│   └── clock.py                # ClockPattern (proper time measurement)
│
├── analysis/                   # Diagnostic tools (DO NOT feed back to engine)
│   ├── phi_field.py            # Derive ϕ from f(x)
│   ├── emergence.py            # Poisson emergence verification
│   └── comparison.py           # Engine vs Newtonian metrics
│
└── viz/                        # Visualization
    ├── fields.py               # Heatmaps
    └── trajectories.py         # Particle paths
```

**Key rule:** `core/` and `patterns/` never import from `analysis/`. The engine computes f(x) from bandwidth dynamics. ϕ(x) is a diagnostic, not a driver.

---

## 2D Limitations

This simulator is 2D, which has important implications:

| 2D (What We Have) | 3D (Kepler's Laws) |
|-------------------|---------------------|
| ϕ(r) ∝ log(r) | ϕ(r) ∝ 1/r |
| Force ∝ 1/r | Force ∝ 1/r² |
| T² ∝ R² | T² ∝ R³ (Kepler) |
| Orbits don't close (rosettes) | Closed ellipses |

**Do not expect Keplerian orbits.** The 2D physics is self-consistent but follows 2D laws.

---

## Documentation

- `CLAUDE.md` — Guidance for AI assistants working on this code
- `design.md` — Detailed technical architecture (~2000 lines)
- `requirements.md` — Functional requirements
- `tasks.md` — Development phases and milestones

---

## References

[Emergent Gravity from Finite Bandwidth in a Message-Passing Lattice](papers/emergent_gravity_from_finite_bandwidth.pdf)
