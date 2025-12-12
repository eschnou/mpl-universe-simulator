# 2D Message-Passing Gravity Engine — Requirements Specification

## 0. Purpose and Scope

**Goal.** Build a 2D simulator of a bandwidth-limited message-passing universe engine that:

- Implements the **message-passing lattice** / universe engine.
- Implements **finite link capacity**, **congestion**, and **time dilation** via local update rate.
- Derives an effective **time-sag / gravitational potential** field ϕ(x) from activity.
- Allows **patterns** (clocks, test particles, light-like packets, agents) to move and experience:
  - Free fall (geodesic-like trajectories).
  - Gravitational time dilation.
  - Gravitational lensing (bending of light-like patterns).
  - Horizon-like trapped regions.
  - A “dark” gravitational component from unseen activity in other patterns/worlds.

The simulator is a **toy engine**: it need not be physically accurate, but must stay faithful to the logic of the two papers and expose all the knobs needed to explore that logic.

No implementation language, UI, or performance strategy is fixed here; this document defines **concepts, modules, and observable behaviour only**.

---

## 1. Conceptual Model

### 1.1 Universe Engine Objects

The simulator must implement the following conceptual entities:

1. **Node**
   - Represents a lattice site in the 2D engine.
   - Carries a small local state and per-node bookkeeping (activity, slowness, proper time, etc.).

2. **Link**
   - Represents a directed/undirected communication channel between neighbouring nodes.
   - Has finite capacity per canonical tick.

3. **Message**
   - Represents information transmitted along a link in one direction.
   - Has an associated “information size” used for capacity accounting.

4. **Local Kernel**
   - A uniform local rule that maps a node’s current state + incoming messages to:
     - Updated node state.
     - Outgoing message deltas on each incident link.

5. **Patterns**
   - Structured configurations of node states/messages (e.g. wave packets, clocks, agents).
   - Used to probe free fall, time dilation, lensing, horizons, etc.

6. **World Tags / Pattern Channels**
   - Optional labels used to distinguish contributions from different quasi-classical patterns (“worlds”) while sharing the same engine.
   - Some patterns are **visible** to a chosen world; others are **hidden** but still contribute to congestion.

7. **Time Fields**
   - **Canonical time**: tick counter for potential update cycles.
   - **Proper time**: accumulated count of *realized* local updates along a pattern’s worldline.
   - **Slowness field λ(x)** and **update fraction f(x)** per node.
   - **Time-sag / gravitational potential ϕ(x)** defined on nodes.

---

## 2. Lattice and Geometry

### 2.1 Dimension and Topology

- The engine runs on a **2D lattice**:
  - Default: regular square grid with size `Nx × Ny`.
  - Neighbourhood:
    - Default: 4-neighbour (Von Neumann) connectivity.
    - Must support configuration of 8-neighbour or custom stencils for experiments.

- **Boundary conditions**:
  - Configurable per run:
    - Periodic (torus).
    - Reflective (no-flux).
    - Open / absorbing.

### 2.2 Coordinate System

- Every node has integer coordinates `(i, j)` with `0 ≤ i < Nx`, `0 ≤ j < Ny`.
- Distance measures:
  - Hop distance (lattice steps).
  - Optional Euclidean distance for diagnostics and plotting.

---

## 3. Time and Scheduling

### 3.1 Canonical Time

- Global **canonical tick index** `T_can ∈ ℕ`:
  - Counts how many “update opportunities” the hardware has offered.
  - Does **not** represent physical time directly for internal observers.

### 3.2 Local Update Cycles

The simulator must implement the notion of:

- **Canonical update opportunity** at node `x` at tick `T_can`.
- **Realized update** at node `x` when:
  - All required incoming messages for this kernel application have been received.
  - All outgoing messages produced by the previous update have been successfully pushed within link capacity constraints.

For each node `x`, the simulator tracks over time:

- `n_possible(x, window)` – number of canonical opportunities in a time window.
- `n_real(x, window)` – number of *completed* local update cycles.
- **Update fraction** `f(x) = n_real(x) / n_possible(x)` (smoothed over configurable window).
- **Slowness** `λ(x) = 1 − f(x)`.

### 3.3 Proper Time

- Each node `x` maintains a **proper time counter** `τ(x)`:
  - Incremented by `+1` whenever a *realized* local update cycle occurs at `x`.
- Patterns anchored at a node (e.g. clocks) use `τ(x)` as their proper time parameter.

---

## 4. Local State, Messages, and Activity

### 4.1 Node State Structure (Minimum Requirements)

Each node must carry at least:

- `state_local`:
  - Abstract local state (could later be refined to a small complex vector or a few scalar channels).
  - Must be extensible for:
    - Visible matter/activity fields.
    - Hidden “other-world” activity channels.
    - Pattern IDs for special structures (clocks, test particles, agents).

- `activity_stats`:
  - Rolling estimate of local activity (used to build ρ_act):
    - e.g. average magnitude of state change per realized update.
    - and/or average message entropy per incident link.

- `f(x)`, `λ(x)`, `ϕ(x)`:
  - Current smoothed update fraction, slowness, and time-sag potential.

- `τ(x)`:
  - Proper time accumulator.

### 4.2 Link State Structure

Each link (directed edge `x → y`) must track:

- `capacity_C_xy`:
  - Max information units per canonical tick.

- `queue_xy`:
  - FIFO or similar queue of outbound messages awaiting transmission.

- `load_stats`:
  - Rolling statistics:
    - Average utilization of capacity.
    - Fraction of canonical ticks where queue is non-empty (proxy for congestion).

### 4.3 Message Model

- A **message** has:
  - `payload_change`:
    - Encodes the **delta** in neighbour’s view of `state_local` (not necessarily full state).
  - `size_bits`:
    - Abstract measure of information size used against link capacity.

- The simulator must allow the **message size function** to be configurable:
  - `size_bits = f_size(state_before, state_after, neighbour)`.

### 4.4 Local Kernel

The **local kernel** must be:

- **Uniform**: same code / rule at every node.
- **Local**: only reads `state_local(x)` and incoming messages from neighbours.
- **Reversible-ish** / information-preserving in spirit (no explicit erasure required).

The simulator must:

- Provide an interface to plug different kernels.
- For v1, support a simple kernel class that:
  - Conserves some scalar “activity” quantity.
  - Produces state changes whose magnitude is controllable and predictable (useful for unit tests).

---

## 5. Capacity Limits, Waiting, and Activity Density

### 5.1 Capacity Constraints

For each canonical tick and each link `x → y`:

- The total size of messages actually transmitted must satisfy:
  - `Σ size_bits(messages sent in this tick) ≤ capacity_C_xy`.

- If the kernel wants to send more than capacity allows:
  - The excess stays in the local outgoing queue.
  - This contributes to **waiting** and **congestion** in future ticks.

### 5.2 Waiting for Inputs

A node **may not** perform a realized update at tick `T_can` if:

- One or more required incoming messages (from neighbour nodes needed by the kernel) have not yet arrived due to upstream congestion.

In such cases:

- The canonical opportunity at `T_can` is counted in `n_possible(x)` but **not** in `n_real(x)`.

### 5.3 Activity Density ρ_act(x)

The simulator must define a configurable **activity density** per node, e.g.:

- `ρ_act(x)` built from:
  - Local state change magnitude, and/or
  - Total information transmitted on incident links, and/or
  - Number of realized updates per window.

It must be possible to:

- Include contributions from:
  - **Visible** patterns (e.g. world A).
  - **Hidden / other-world** patterns (worlds B, C, …).
- Toggle which contributions are:
  - Used in computing engine-level ϕ(x).
  - Visible to observers in a given world (for dark-matter experiments).

---

## 6. Time-Sag Field ϕ(x)

### 6.1 Definition and Storage

Each node `x` must store a scalar **time-sag / potential** value `ϕ(x)`.

The simulator must support at least two modes for computing ϕ:

1. **Empirical mode**:
   - Define ϕ(x) as a function of measured slowness:
     - e.g. `ϕ(x) = F(f(x))` with user-defined F (e.g. linear, logarithmic).

2. **Poisson-solver mode**:
   - ϕ(x) is computed from ρ_act(x) via a discrete Poisson-like relation:
     - `Δ_graph ϕ(x) ≈ κ · ρ_act(x)`
   - Implemented via iterative relaxation or other standard solvers on the 2D grid.

### 6.2 Update Schedule

- The simulator must support:
  - **Synchronous** update of ϕ(x) every N canonical ticks.
  - **Asynchronous**/incremental updates (e.g. a few relaxation sweeps per tick).

- Both ϕ(x) modes must be usable in the same simulation to compare:
  - Direct f(x)-based time dilation.
  - Smoothed potential field from activity statistics.

---

## 7. Pattern Layer (Clocks, Particles, Light, Agents)

The simulator must provide a **pattern abstraction** built on top of node states.

### 7.1 Pattern Representation

Each pattern instance must have:

- A unique ID.
- A **world tag** (e.g. “visible world A”, “other-world B”).
- A definition of:
  - Which nodes / channels it occupies.
  - How to compute its **center** (for trajectory tracking).
  - Its “mass” or “activity contribution” profile.

### 7.2 Clock Patterns

Requirements:

- Ability to define a localised **clock pattern** anchored at specific node(s).
- Clock tick logic:
  - Clock’s internal phase increments by 1 on each *realized* update of its host node or small node cluster.
- Observables:
  - Proper time `τ_clock` recorded as function of canonical time `T_can`.
  - Must support multiple clocks in different ϕ regions for time-dilation unit tests.

### 7.3 Massive Test Particles

Requirements:

- Define patterns that behave as **massive test particles**:
  - Localised wave-packet-like structures that:
    - Move through the lattice.
    - Contribute to activity ρ_act(x) according to a configurable profile.

- Motion:
  - Default motion should approximate:
    - Straight lines in flat (ϕ ≈ 0) background.
    - Curved paths along ∇ϕ in regions with gradients.

- Implementation options:
  - Directly via underlying kernel (patterns are emergent).
  - Or via an effective “particle integrator” that:
    - Updates particle position using local ϕ(x) as an effective potential.
    - Still accounts for its activity load on nearby nodes.

### 7.4 Light-like Patterns (for Lensing)

Requirements:

- Define **massless / light-like** patterns:
  - Travel at maximal effective speed allowed by the engine.
  - Trajectories sensitive to local propagation speed `∝ f(x)` or equivalently to an effective index `n_eff(x)` derived from ϕ(x).

- Observables:
  - Deflection angles when passing near regions of deep ϕ.
  - Arrival times at distant detectors.

### 7.5 Horizon-like Regions

Requirements:

- The simulator must support scenarios where:
  - In a region, `f(x)` is driven close to zero by extremely high ρ_act.
  - ϕ(x) becomes very deep.

- Behaviour to observe:
  - Patterns entering this region see proper time almost freeze.
  - Signals from inside take arbitrarily long (in outside proper time) to escape, if at all.

Implementation:
- Configurable sources that force persistent high activity in a finite region.
- Diagnostics to show:
  - `f(x)` and `ϕ(x)` profile radially around the source.
  - Inward versus outward trajectories of test patterns.

### 7.6 Agents (Optional for v1, Desirable for v2)

- Provide a simple agent abstraction:
  - Localised pattern with:
    - Internal memory.
    - Decision rules that depend on local records (e.g., local clock, local ϕ estimate).
- Purpose:
  - Explore how agent-like structures experience “apparent collapse” and branch structure relative to the engine.

---

## 8. Many-Worlds and Dark-Matter-Like Behaviour

### 8.1 World Tags and Engine-Level Gravity

Requirements:

- Every pattern can carry a **world tag**:
  - World A (visible to our observer).
  - Other worlds (B, C, …) that share the same lattice.

- Engine-level ϕ(x) must be computed from **all** activity:
  - The engine does not know about worlds.
  - Only cares about ρ_act(x) built from full state.

### 8.2 Visible vs Hidden Activity

Requirements:

- For any given **observer world** (e.g. World A):
  - There is a notion of **visible mass/activity**:
    - ρ_visible(x) from patterns tagged as that world.
  - There is a total **engine-level activity**:
    - ρ_engine(x) including all world tags.

- The simulator must:
  - Allow plotting/comparing ϕ(x) inferred from ρ_engine(x) vs ϕ(x) expected from ρ_visible(x) alone.
  - Make it possible to design scenarios where:
    - ρ_visible explains only part of the potential.
    - The rest is due to hidden activity → “dark matter” analogue.

---

## 9. Experiment / Unit-Test Harness

The simulator must ship with an **experiment framework** that makes it easy to define and run scenarios.

### 9.1 Core API (Conceptual)

The harness should expose functions (conceptually):

- `setup_lattice(config)`
- `place_pattern(pattern_spec)`
- `run_simulation(T_can_max, output_schedule)`
- `measure(observable_spec)`
- `export_results(format)`

No specific technology is required, but these capabilities must exist.

### 9.2 Required Experiments

At minimum, the following benchmark experiments must be expressible:

1. **Static Point Source — Time Dilation**
   - Setup:
     - Single compact region with high persistent activity (source).
     - Two clocks: one near the source, one far away.
   - Measurements:
     - τ_near(T_can) vs τ_far(T_can).
     - f(x) and ϕ(x) radial profile.
   - Expected qualitative result:
     - τ_near grows more slowly than τ_far.
     - ϕ(x) ~ 1/r falloff in 2D analogue (logarithmic) or at least monotonic.

2. **Free-Fall Trajectories**
   - Setup:
     - Same time-sag field as above.
     - Massive test particles released with initial horizontal velocity at various impact parameters.
   - Measurements:
     - Trajectories in (x, y) over τ.
   - Expected qualitative result:
     - Curvature toward the source, trajectories approximating motion in potential −ϕ(x).

3. **Gravitational Lensing**
   - Setup:
     - Light-like patterns propagating across the lattice toward a detector line.
     - One or more mass/ activity wells in between.
   - Measurements:
     - Arrival positions along the detector line.
     - Angular deflection vs impact parameter.
   - Expected qualitative result:
     - Distorted arrival profile compared to flat-space control.

4. **Horizon-like Region**
   - Setup:
     - Very high activity in a compact core region.
     - ϕ(x) deep enough that f(x) ≪ 1 in core.
   - Measurements:
     - Proper time rate τ_inside / τ_far over T_can.
     - Escape vs trapping of test particles launched from near the core.
   - Expected qualitative result:
     - Strong slow-down of τ inside.
     - Threshold behaviour (impact-parameter-like) between escape and trapped trajectories.

5. **Dark-Matter Analogue**
   - Setup:
     - World A (visible) has a ring or cluster of matter.
     - Hidden world B has additional ring(s) offset or more spread out.
   - Measurements:
     - ϕ(x) computed from full ρ_engine(x).
     - ϕ_visible(x) computed from ρ_visible(x) of A only.
     - Test-particle rotation curves or lensing using engine-level ϕ(x).
   - Expected qualitative result:
     - Observers in world A, if they only account for visible matter, infer a “missing mass” component.

---

## 10. Observability and Instrumentation

The engine must support structured access to:

- Field snapshots:
  - `f(x)`, `λ(x)`, `ϕ(x)`, `ρ_act(x)`, ρ_visible(x), ρ_engine(x).
- Pattern data:
  - Positions, velocities, world tags, proper times.
- Link metrics:
  - Capacity usage per link.
  - Queue lengths and congestion hotspots.

The simulator must allow:

- Temporal sampling at configurable intervals.
- Export in standard data formats (e.g., CSV/JSON/NetCDF) for external plotting and analysis.

---

## 11. Configurability and Extensibility

The design must allow:

- Changing lattice size and topology.
- Changing link capacities and their spatial dependence.
- Swapping local kernels.
- Adding new pattern types without modifying the core engine.
- Plugging in alternative ϕ(x) solvers and activity definitions.

Non-goals for v1:

- Full Lorentz-invariant dynamics.
- Full-blown complex Hilbert space dynamics with interference (can be approximated or left for v2).
- Detailed modelling of realistic astrophysical systems.

---

## 12. Validation and Sanity Checks

The simulator must provide basic checks:

1. **Flat background sanity test**
   - With no sources and low activity:
     - f(x) ≈ 1 everywhere.
     - ϕ(x) ~ constant.
     - Clocks tick at the same rate.
     - Particles move in straight lines.

2. **Reversibility/information-preservation diagnostics (qualitative)**
   - No hard erasure operations in the core kernel.
   - Ability to approximate time reversal on toy scenarios (optional).

3. **Scaling checks**
   - Confirm that total memory and compute per tick scale ∝ number of nodes (for fixed local-state size and degree).

These ensure the engine behaves as a genuine local, resource-limited universe engine rather than a bespoke gravity solver.

---

