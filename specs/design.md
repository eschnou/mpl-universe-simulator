# 2D Message-Passing Gravity Engine — Design Document

## 0. Overview

This document specifies the technical design for a 2D simulator of a bandwidth-limited, message-passing universe engine as described in the companion papers on many-worlds interpretations and emergent gravity from time dilation.

**Primary Goals:**
- Faithful implementation of the conceptual model from the papers
- Clean separation of concerns (engine core, patterns, experiments)
- Extensibility for future experiments and alternative kernels
- Performance sufficient for interactive exploration on grids up to ~500×500 nodes

---

## 0.1 Conceptual Hierarchy: Primitives vs Derived Quantities

**This distinction is fundamental to the design.**

The papers establish a clear causal chain:

```
PRIMITIVES (what the engine actually computes):
┌─────────────────────────────────────────────────────────────────┐
│  Local activity → Message sizes → Link congestion → Waiting    │
│                                                                 │
│  Waiting + bandwidth limits → Fraction of updates realized     │
│                                                                 │
│  f(x) = realized_updates / possible_updates                    │
│                                                                 │
│  Proper time τ(x) = accumulated realized updates               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    (derived / emergent)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  DERIVED QUANTITIES (analysis tools, not engine primitives):   │
│                                                                 │
│  • Time-sag field ϕ(x) — a smooth summary of f(x) patterns     │
│  • Poisson equation ∇²ϕ = κρ — an approximation that holds     │
│    in quasi-static, smoothly-varying regimes                   │
│  • "Gravitational potential" language — useful analogy         │
└─────────────────────────────────────────────────────────────────┘
```

**What this means for the implementation:**

1. **The core engine knows nothing about ϕ or Poisson equations.** It only knows:
   - Node states and how to apply the local kernel
   - Messages, queues, and link capacities
   - Whether a node *can* update this tick (inputs arrived? queue cleared?)
   - Counting realized vs possible updates → f(x)
   - Incrementing proper time τ(x) on realized updates

2. **Patterns experience f(x) directly, not ϕ.** A clock ticks when its host node completes a realized update. A particle moves through regions where f(x) varies and naturally slows down or speeds up. The "gravitational" effects are not computed from a potential field — they emerge from the bandwidth-limited dynamics.

3. **ϕ(x) and Poisson solvers are diagnostic/analysis tools.** They sit in a separate layer and are used for:
   - Visualizing the emergent "gravity-like" structure
   - Comparing engine behavior to Newtonian predictions
   - Providing geodesic-style trajectory integration as an *approximation* (useful for fast experiments, but not the "true" engine dynamics)

4. **Two modes of pattern motion:**
   - **Engine-native:** Pattern state lives in the lattice; motion emerges from message-passing dynamics and local f(x). This is the "real" behavior.
   - **Geodesic-approximate:** Pattern position integrated using ∇ϕ as an effective force. Faster to compute, useful for quick exploration, but an approximation.

This separation keeps the philosophy of the papers intact: gravity *is* time dilation from bandwidth limits, not a force field that happens to cause time dilation.

---

## 1. Technology Stack

### 1.1 Language: Python 3.11+

**Rationale:**
- Rapid prototyping and iteration
- Rich scientific ecosystem (NumPy, SciPy, Matplotlib)
- Familiar to physics/simulation researchers
- Easy to extend with Cython/Numba for hot paths if needed

### 1.2 Core Dependencies

| Library | Purpose | Version |
|---------|---------|---------|
| `numpy` | Array operations, field storage | ≥1.24 |
| `scipy` | Poisson solver (sparse linalg), spatial algorithms | ≥1.11 |
| `matplotlib` | Visualization, field plots, trajectories | ≥3.7 |
| `dataclasses` | Structured configuration and state objects | stdlib |
| `typing` | Type hints for clarity | stdlib |
| `pytest` | Unit and integration testing | ≥7.0 |
| `pydantic` | Configuration validation (optional but recommended) | ≥2.0 |
| `h5py` or `netCDF4` | Large dataset export (optional) | latest |

### 1.3 Optional Performance Extensions

- **Numba**: JIT compilation for hot loops (kernel application, message routing)
- **Joblib/multiprocessing**: Parallel field updates for large grids

---

## 2. Package Structure

The package structure reflects the **primitives vs derived** distinction:

```
simulator/
├── requirements.md          # Functional requirements (existing)
├── design.md                 # This document
├── pyproject.toml            # Project metadata and dependencies
├── src/
│   └── manyworld/
│       ├── __init__.py
│       │
│       │   ═══════════════════════════════════════════════════════
│       │   LAYER 1: ENGINE PRIMITIVES
│       │   The "universe hardware" — knows nothing about gravity
│       │   ═══════════════════════════════════════════════════════
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── lattice.py        # Lattice, Node, Link structures
│       │   ├── messages.py       # Message, MessageQueue, capacity accounting
│       │   ├── kernel.py         # Kernel protocol and implementations
│       │   ├── scheduler.py      # Update scheduling: eligibility, waiting, f(x)
│       │   └── proper_time.py    # τ(x) accumulation (THE primitive clock)
│       │
│       │   ═══════════════════════════════════════════════════════
│       │   LAYER 2: PATTERNS (live in the engine)
│       │   Structured configurations that experience f(x) directly
│       │   ═══════════════════════════════════════════════════════
│       │
│       ├── patterns/
│       │   ├── __init__.py
│       │   ├── base.py           # Pattern protocol and registry
│       │   ├── clock.py          # Clock: ticks on realized updates (native)
│       │   ├── wavepacket.py     # Emergent particle: state in lattice (native)
│       │   └── agent.py          # Agent pattern (v2)
│       │
│       ├── worlds/
│       │   ├── __init__.py
│       │   └── tags.py           # World tagging, visible/hidden activity split
│       │
│       │   ═══════════════════════════════════════════════════════
│       │   LAYER 3: ANALYSIS & DIAGNOSTICS
│       │   Derived quantities — NOT seen by the engine
│       │   ═══════════════════════════════════════════════════════
│       │
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── phi_field.py      # ϕ(x) computation (empirical or Poisson)
│       │   ├── poisson.py        # Discrete Poisson solver (diagnostic tool)
│       │   ├── activity.py       # ρ_act density (for Poisson, visualization)
│       │   ├── geodesic.py       # Geodesic integrator using ∇ϕ (APPROXIMATE)
│       │   └── dark_matter.py    # Visible vs engine-level ϕ comparison
│       │
│       │   ═══════════════════════════════════════════════════════
│       │   LAYER 4: EXPERIMENT HARNESS & VISUALIZATION
│       │   ═══════════════════════════════════════════════════════
│       │
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── harness.py        # Experiment runner
│       │   ├── scenarios.py      # Pre-defined scenario builders
│       │   └── observables.py    # Measurement and export utilities
│       │
│       ├── viz/
│       │   ├── __init__.py
│       │   ├── fields.py         # Field heatmaps (f, τ, and derived ϕ)
│       │   ├── trajectories.py   # Pattern trajectory plots
│       │   └── animation.py      # Time evolution animations
│       │
│       └── config.py             # Global configuration dataclasses
│
├── tests/
│   ├── conftest.py               # Pytest fixtures
│   ├── unit/
│   │   ├── test_lattice.py
│   │   ├── test_kernel.py
│   │   ├── test_scheduler.py     # Core: f(x), waiting, proper time
│   │   ├── test_patterns.py
│   │   └── test_poisson.py       # Analysis layer
│   └── integration/
│       ├── test_time_dilation.py
│       ├── test_free_fall.py
│       ├── test_lensing.py
│       ├── test_horizons.py
│       └── test_dark_matter.py
│
└── notebooks/
    ├── 01_basic_demo.ipynb
    ├── 02_time_dilation.ipynb
    ├── 03_free_fall.ipynb
    ├── 04_lensing.ipynb
    └── 05_dark_matter.ipynb
```

**Key architectural boundary:** The `core/` and `patterns/` modules never import from `analysis/`. Information flows one way: engine primitives → analysis tools.

---

## 3. Core Engine Architecture

### 3.1 Lattice Representation

The lattice is a 2D grid of nodes connected by links. We use array-based storage for efficiency.

```python
@dataclass
class LatticeConfig:
    nx: int                          # Grid width
    ny: int                          # Grid height
    neighborhood: Literal["von_neumann", "moore"]  # 4 or 8 neighbors
    boundary: Literal["periodic", "reflective", "absorbing"]
    link_capacity: float             # Default capacity C per link per tick
    capacity_map: np.ndarray | None  # Optional per-link capacity overrides

@dataclass
class Lattice:
    """
    The universe engine's state.

    IMPORTANT: This class contains ONLY engine primitives.
    ϕ(x), ρ_act(x), and other derived quantities live in the analysis layer.
    """
    config: LatticeConfig

    # ═══════════════════════════════════════════════════════════════
    # PRIMITIVE STATE (what the engine actually tracks)
    # ═══════════════════════════════════════════════════════════════

    # Node state arrays (all shape: [ny, nx] or [ny, nx, channels])
    state: np.ndarray          # Local state per node, shape [ny, nx, n_channels]
    proper_time: np.ndarray    # τ(x) accumulator, shape [ny, nx], dtype int64
                               # THE fundamental clock — increments on realized updates

    # Update statistics (primitives from which f(x) is computed)
    realized_updates: np.ndarray   # Count of realized updates, shape [ny, nx]
    possible_updates: np.ndarray   # Count of canonical opportunities, shape [ny, nx]
    f: np.ndarray                  # Update fraction = realized/possible, shape [ny, nx]
                                   # This IS the "gravity" — not derived from anything else

    # Link state (for each direction)
    # Using separate arrays for each cardinal direction for efficiency
    queues: dict[str, MessageQueue]  # "N", "S", "E", "W" (and diagonals if Moore)

    # ═══════════════════════════════════════════════════════════════
    # NOTE: The following are NOT stored in Lattice:
    #   - phi (ϕ) — derived, lives in analysis.phi_field
    #   - rho_act — derived, lives in analysis.activity
    #   - slowness (λ) — trivially λ = 1-f, computed on demand
    # ═══════════════════════════════════════════════════════════════
```

**Design Decision: Dense Arrays vs Sparse Structures**

We use dense NumPy arrays because:
- The lattice is regular and fully occupied
- Vectorized operations are much faster than Python loops
- Memory is acceptable: 500×500 × 8 fields × 8 bytes ≈ 16 MB

### 3.2 Node State Model

Each node carries a small local state vector. For v1, we use a simple multi-channel scalar model:

```python
# State channels per node
N_CHANNELS = 4  # Configurable

# Channel semantics (example):
# Channel 0: "mass/activity" - contributes to ρ_act
# Channel 1: "momentum_x" - directional flow
# Channel 2: "momentum_y" - directional flow
# Channel 3: "aux" - pattern markers, world tags encoded here

# state array shape: [ny, nx, N_CHANNELS]
# Each channel is a float64
```

**Extensibility:** The kernel and pattern layer reference channels by name via a `ChannelMap` configuration, allowing different kernels to use different semantics.

### 3.3 Link and Message Model

Links connect adjacent nodes. Each directed link has a capacity and a queue.

```python
@dataclass
class Message:
    source: tuple[int, int]    # (i, j) of sender
    delta: np.ndarray          # State change vector, shape [n_channels]
    size_bits: float           # Information size for capacity accounting
    tick_created: int          # Canonical tick when generated

@dataclass
class MessageQueue:
    """FIFO queue for one direction of links across the entire lattice."""
    # For efficiency, store as arrays rather than per-link lists
    # pending[j, i] = list of messages waiting to be sent from (i,j) in this direction
    pending: list[list[list[Message]]]  # [ny][nx] -> list of Message

    # Capacity tracking
    sent_this_tick: np.ndarray   # [ny, nx] - bits sent this tick
    capacity: np.ndarray         # [ny, nx] - capacity per link
```

**Alternative Design:** For very large grids, we could use a sparse representation where only links with non-empty queues are stored. For v1, the simpler dense approach suffices.

### 3.4 Message Size Function

The information size of a message is configurable:

```python
class MessageSizer(Protocol):
    def compute_size(
        self,
        state_before: np.ndarray,
        state_after: np.ndarray,
        direction: str
    ) -> float:
        """Return size in bits for the delta message."""
        ...

class SimpleDeltaSizer:
    """Size proportional to L2 norm of state change."""
    def __init__(self, bits_per_unit: float = 1.0):
        self.bits_per_unit = bits_per_unit

    def compute_size(self, state_before, state_after, direction) -> float:
        delta = state_after - state_before
        return np.linalg.norm(delta) * self.bits_per_unit
```

---

## 4. Local Kernel Design

### 4.1 Kernel Protocol

```python
class Kernel(Protocol):
    """Protocol for local update kernels."""

    def compute_update(
        self,
        node_state: np.ndarray,           # [n_channels]
        incoming_messages: dict[str, np.ndarray],  # direction -> delta
        position: tuple[int, int],
        lattice: "Lattice"
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Compute one local update cycle.

        Returns:
            new_state: Updated node state
            outgoing_deltas: Direction -> delta to send to that neighbor
        """
        ...

    @property
    def required_inputs(self) -> set[str]:
        """Which neighbor directions must have delivered messages."""
        ...
```

### 4.2 v1 Kernel: Load Generator (For Gravity Experiments)

**Design principle:** For the gravity experiments, the kernel's job is simple:
- **Sources** = nodes that try to send lots of messages
- **Background** = nodes that send minimal messages
- Congestion arises when message volume exceeds link capacity

The detailed form of a "wave equation" is irrelevant for demonstrating:
`high activity → congestion → waiting → low f(x) → time dilation`

```python
class LoadGeneratorKernel:
    """
    Minimal kernel whose only job is to generate configurable message traffic.

    NOT physically realistic — just creates the load patterns needed to
    demonstrate bandwidth-limited gravity.

    Usage:
    - Mark some nodes as "sources" with high output_rate
    - Background nodes use low output_rate
    - Observe f(x) drop near sources due to congestion
    """

    def __init__(
        self,
        background_rate: float = 0.1,    # Messages per tick for normal nodes
        message_size: float = 1.0,        # Bits per message
    ):
        self.background_rate = background_rate
        self.message_size = message_size

    def compute_update(
        self,
        node_state: np.ndarray,
        incoming_messages: dict[str, np.ndarray],
        position: tuple[int, int],
        source_map: np.ndarray,  # [ny, nx] array of output rates (0 = use background)
    ) -> tuple[np.ndarray, dict[str, Message]]:
        """
        Generate outgoing messages based on this node's configured load.

        The state update is trivial — what matters is message generation.
        """
        i, j = position

        # Determine this node's output rate
        output_rate = source_map[j, i]
        if output_rate == 0:
            output_rate = self.background_rate

        # Trivial state update: just increment a counter (for diagnostics)
        new_state = node_state.copy()
        new_state[0] += 1  # tick counter

        # Generate outgoing messages
        # Each message is just "I updated" with configurable size
        outgoing = {}
        for direction in ["N", "S", "E", "W"]:
            outgoing[direction] = Message(
                source=position,
                delta=np.array([output_rate]),  # Content doesn't matter much
                size_bits=output_rate * self.message_size,  # THIS is what matters
                tick_created=int(new_state[0])
            )

        return new_state, outgoing

    @property
    def required_inputs(self) -> set[str]:
        # For v1: don't require inputs, just generate load
        # This means nodes won't block waiting for neighbors
        # Congestion comes purely from output queue saturation
        return set()  # No required inputs


class SourceMap:
    """
    Defines which nodes are "sources" (high message output) vs background.

    This is how we create "mass" in the engine:
    - Place a Gaussian blob of high output_rate somewhere
    - That region generates lots of messages
    - Links saturate → queues build up → neighbors wait → f(x) drops
    - Time dilation emerges around the "mass"
    """

    def __init__(self, ny: int, nx: int, background_rate: float = 0.1):
        self.rates = np.full((ny, nx), background_rate)

    def add_point_source(self, x: int, y: int, rate: float):
        """Add a point source at (x, y) with given output rate."""
        self.rates[y, x] = rate

    def add_gaussian_source(
        self,
        cx: int, cy: int,
        peak_rate: float,
        sigma: float
    ):
        """Add a Gaussian blob of activity centered at (cx, cy)."""
        ny, nx = self.rates.shape
        yy, xx = np.ogrid[:ny, :nx]
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        self.rates += peak_rate * np.exp(-dist_sq / (2 * sigma**2))

    def add_ring_source(
        self,
        cx: int, cy: int,
        radius: float,
        width: float,
        rate: float
    ):
        """Add a ring of activity (for galaxy rotation curve experiments)."""
        ny, nx = self.rates.shape
        yy, xx = np.ogrid[:ny, :nx]
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        ring_mask = np.abs(dist - radius) < width
        self.rates[ring_mask] = rate
```

**Why this works for gravity experiments:**

1. **Time dilation test:** Place a Gaussian source. Clocks near it tick slower because f(x) drops.

2. **Free fall test:** Place a source. Release a particle nearby. It drifts toward the source because f(x) gradient affects its update rate.

3. **Lensing test:** Place a source. Send light rays past it. They bend toward the source because effective propagation speed ∝ f(x).

4. **Dark matter test:** Place visible sources in world A, hidden sources in world B. Engine-level f(x) sees both; observers in A see only their sources.

---

### 4.3 v2 Kernel: Wave-like Dynamics (Future)

For later experiments requiring actual wave-packet propagation:

```python
class WaveKernel:
    """
    Kernel with wave-like dynamics for propagating coherent patterns.

    Key requirements:
    - Stable, non-dispersive wave packets
    - Conserves total "mass" (L2 norm of state)
    - Message size scales with state change magnitude

    Deferred to v2 — LoadGeneratorKernel is sufficient for gravity experiments.
    """
    pass  # TODO: implement after v1 gravity demos work
```

**When to upgrade to WaveKernel:**
- When we need actual wave packets that hold together and propagate
- When testing interference effects
- When pattern motion must emerge from dynamics (not geodesic approximation)

---

## 5. Scheduling and Update Logic

### 5.1 Canonical Tick Loop

```python
class Scheduler:
    def __init__(self, lattice: Lattice, kernel: Kernel, sizer: MessageSizer):
        self.lattice = lattice
        self.kernel = kernel
        self.sizer = sizer
        self.canonical_tick = 0

        # Statistics windows for f(x) computation
        self.window_size = 100
        self.possible_updates = np.zeros((lattice.config.ny, lattice.config.nx))
        self.realized_updates = np.zeros((lattice.config.ny, lattice.config.nx))

    def tick(self):
        """Execute one canonical tick."""
        self.canonical_tick += 1

        # Phase 1: Determine which nodes CAN update
        can_update = self._check_update_eligibility()

        # Phase 2: For eligible nodes, compute kernel and generate messages
        new_states, outgoing = self._apply_kernel_vectorized(can_update)

        # Phase 3: Attempt to send messages, respecting capacity
        self._route_messages(outgoing, can_update)

        # Phase 4: Update statistics
        self._update_statistics(can_update)

        # Phase 5: Deliver messages that fit in capacity
        self._deliver_messages()

        # Phase 6: Commit new states for nodes that updated
        self._commit_updates(new_states, can_update)

    def _check_update_eligibility(self) -> np.ndarray:
        """
        Returns boolean mask [ny, nx] of nodes that can update this tick.

        A node can update if:
        1. All required incoming messages have arrived
        2. Its outgoing queues from last tick have cleared
        """
        # Check incoming message availability
        has_inputs = np.ones((self.lattice.config.ny, self.lattice.config.nx), dtype=bool)
        for direction in self.kernel.required_inputs:
            opposite = self._opposite_direction(direction)
            # Node needs message from neighbor in 'direction'
            # That means neighbor in 'direction' must have sent to us (opposite)
            has_inputs &= self._check_message_arrived(direction)

        # Check outgoing queue cleared
        queues_clear = self._check_queues_cleared()

        return has_inputs & queues_clear
```

### 5.2 Synchronous vs Asynchronous Modes

**Synchronous Mode (Default for v1):**
- All eligible nodes update in parallel each tick
- Simpler to reason about and debug
- Matches the "logical round" abstraction from the papers

**Asynchronous Mode (Future):**
- Nodes update when their local conditions are met
- More realistic but harder to analyze
- Could be implemented with event-driven scheduling

### 5.3 Update Fraction and Slowness Computation

```python
def _update_statistics(self, updated_mask: np.ndarray):
    """Update rolling statistics for f(x) computation."""
    # Increment possible for all nodes
    self.possible_updates += 1

    # Increment realized only for nodes that updated
    self.realized_updates += updated_mask.astype(float)

    # Compute f(x) with smoothing
    # Use exponential moving average for online computation
    alpha = 2.0 / (self.window_size + 1)
    self.lattice.f = alpha * updated_mask + (1 - alpha) * self.lattice.f

    # Slowness
    self.lattice.slowness = 1.0 - self.lattice.f
```

---

## 6. Analysis Layer: Derived Quantities

**IMPORTANT:** Everything in this section is a *diagnostic tool*, not an engine primitive. The engine computes f(x) and τ(x) from bandwidth/waiting dynamics. The quantities below are derived for analysis, visualization, and comparison with Newtonian gravity.

---

### 6.0 Two Modes: Engine vs Analytic Comparison

**This distinction prevents circular reasoning.**

There are two ways to get a "gravitational potential" field, and they serve different purposes:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MODE 1: ENGINE MODE (The "Real" Physics)                               │
│  ════════════════════════════════════════                               │
│                                                                         │
│  Causal chain (ONE DIRECTION ONLY):                                     │
│                                                                         │
│    SourceMap (output rates)                                             │
│         ↓                                                               │
│    LoadGeneratorKernel generates messages                               │
│         ↓                                                               │
│    Link congestion, queue buildup                                       │
│         ↓                                                               │
│    Nodes wait → fewer realized updates                                  │
│         ↓                                                               │
│    f(x) = realized / possible  ← MEASURED from engine                   │
│         ↓                                                               │
│    ϕ_engine(x) = F(f(x))  ← DERIVED diagnostic (e.g., log f, or 1-f)   │
│         ↓                                                               │
│    Used for: visualization, test particle steering, lensing plots       │
│                                                                         │
│  ⚠️  ϕ_engine does NOT feed back into engine dynamics.                  │
│  ⚠️  The engine knows nothing about ϕ. Only messages and queues.        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  MODE 2: ANALYTIC COMPARISON MODE (Theoretical Reference)               │
│  ═════════════════════════════════════════════════════════              │
│                                                                         │
│  Purpose: "What SHOULD we see if engine matches Newtonian gravity?"     │
│                                                                         │
│    ρ_specified(x)  ← ANALYST specifies (e.g., point source, Gaussian)  │
│         ↓                                                               │
│    Solve Poisson: ∇²ϕ = κ·ρ_specified                                   │
│         ↓                                                               │
│    ϕ_Poisson(x)  ← Theoretical Newtonian prediction                     │
│         ↓                                                               │
│    Compare: ϕ_engine(x) vs ϕ_Poisson(x)                                 │
│                                                                         │
│  ⚠️  ρ_specified is NOT derived from the engine.                        │
│  ⚠️  It's an input for theoretical comparison, not a computed field.    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why this matters:**

In Engine Mode, the source of gravity is **measured slowdown f(x)**, not Poisson.
The Poisson equation is a theoretical tool for comparison, not a computational step.

If we solved Poisson on some ρ_act derived from f, and then used that ϕ to move patterns,
and those patterns affected activity, which affected f... we'd have a feedback loop that
doesn't match the papers' story.

**The papers' story is:**
- Activity → congestion → f(x) drops → that IS gravity
- Poisson is an approximation that summarizes this in quasi-static regimes

**Implementation rule:**
- `analysis/phi_field.py` computes ϕ_engine from f(x) — one-way derivation
- `analysis/poisson.py` solves for ϕ_Poisson from analyst-specified ρ — separate tool
- These two never mix in the causal chain

---

### 6.1 Activity Density ρ: Where Does It Come From?

**Clarification:** There's only ONE source of truth for activity: the `SourceMap`.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SINGLE SOURCE OF TRUTH                                                 │
│                                                                         │
│  SourceMap (per world)                                                  │
│       │                                                                 │
│       ├──→ LoadGeneratorKernel  →  congestion  →  f(x)  →  ϕ_engine    │
│       │    (engine runs on this)   (measured)     (derived)            │
│       │                                                                 │
│       └──→ ρ for Poisson comparison (directly from SourceMap rates)    │
│            (analyst uses this)                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**The lattice does NOT compute ρ.** It only measures f(x) from congestion.
**Patterns don't write into a separate ρ_act field.** They configure `SourceMap`, which drives the kernel.

---

### 6.1.1 ρ for Dark Matter Experiments

For multi-world experiments, each world has its own `SourceMap`:

```python
# World A (visible): galaxy with central bulge
source_map_A = SourceMap(ny, nx)
source_map_A.add_gaussian_source(cx=50, cy=50, peak_rate=10.0, sigma=5)

# World B (hidden): extended halo
source_map_B = SourceMap(ny, nx)
source_map_B.add_ring_source(cx=50, cy=50, radius=30, width=10, rate=3.0)

# Engine sees BOTH (doesn't know about worlds)
source_map_engine = source_map_A + source_map_B  # element-wise sum
```

**Activity densities for comparison:**

| Quantity | Definition | Purpose |
|----------|------------|---------|
| ρ_engine | `source_map_A.rates + source_map_B.rates` | What actually drives f(x) |
| ρ_visible | `source_map_A.rates` | What observers in world A "see" |
| ρ_dark | `ρ_engine - ρ_visible` | The "missing mass" |

**Key point:** These ρ values come directly from `SourceMap` configurations, not from measuring the engine. They're used for:
1. Poisson comparison (does ϕ_engine match ϕ_Poisson(ρ_engine)?)
2. Dark matter analysis (what's the gap between ϕ_Poisson(ρ_visible) and ϕ_engine?)

---

### 6.1.2 ρ for Analytic Comparison

```python
class AnalyticSourceDistribution:
    """
    Activity/mass distribution for Poisson comparison.

    Constructed FROM SourceMap configurations, not measured from engine.
    """

    @classmethod
    def from_source_map(cls, source_map: SourceMap) -> "AnalyticSourceDistribution":
        """Create ρ directly from a SourceMap."""
        dist = cls(source_map.rates.shape[0], source_map.rates.shape[1])
        dist.rho = source_map.rates.copy()
        return dist

    @classmethod
    def from_world_manager(
        cls,
        world_manager: "WorldManager",
        world_filter: set[int] | None = None
    ) -> "AnalyticSourceDistribution":
        """
        Create ρ by summing SourceMaps from selected worlds.

        Args:
            world_filter: If None, sum all worlds (ρ_engine).
                         If specified, sum only those worlds (e.g., ρ_visible).
        """
        # Sum source_map.rates from each world
        ...
```

**Typical workflow:**
1. Configure `SourceMap` for each world
2. Run engine with combined source map → measure f(x) → derive ϕ_engine
3. For comparison:
   - ρ_engine = sum of all SourceMaps → ϕ_Poisson(ρ_engine)
   - ρ_visible = SourceMap of visible world only → ϕ_Poisson(ρ_visible)
4. Compare: ϕ_engine ≈ ϕ_Poisson(ρ_engine)? (validates engine)
5. Dark matter: gap between ϕ_engine and ϕ_Poisson(ρ_visible)

### 6.2 Time-Sag Field ϕ(x) (Diagnostic)

**There are TWO distinct ϕ fields. Don't confuse them:**

| Field | Source | Purpose |
|-------|--------|---------|
| **ϕ_engine** | Derived from measured f(x) | The "real" emergent potential from engine dynamics |
| **ϕ_Poisson** | Solved from analyst-specified ρ | Theoretical Newtonian prediction for comparison |

**ϕ_engine** is what the engine produces. **ϕ_Poisson** is what theory predicts.
Comparing them validates that bandwidth-limited dynamics reproduce Newtonian gravity.

---

#### 6.2.1 ϕ_engine: Derived from f(x) (Engine Mode)

This is the "physical" potential — derived ONE-WAY from the engine's measured f(x).

**Causal direction:** `engine dynamics → f(x) → ϕ_engine` (no feedback)

Simple direct mapping from update fraction to potential:

```python
class PhiFromF:
    """
    Compute ϕ_engine directly from measured f(x).

    This is the ENGINE MODE potential — the "real" emergent gravity.
    No Poisson equation involved. Just a function of f(x).
    """

    def __init__(self, mode: Literal["linear", "log", "custom"], alpha: float = 1.0):
        self.mode = mode
        self.alpha = alpha

    def compute(self, f: np.ndarray) -> np.ndarray:
        """
        Derive ϕ_engine from the engine's measured update fraction f(x).

        Args:
            f: Update fraction field from lattice, shape [ny, nx]

        Returns:
            phi_engine: Derived potential, shape [ny, nx]
        """
        if self.mode == "linear":
            # ϕ = α * (f - 1), so ϕ < 0 where f < 1 (deep potential = slow region)
            return self.alpha * (f - 1.0)

        elif self.mode == "log":
            # ϕ = α * log(f), more sensitive near f → 0 (horizon-like)
            return self.alpha * np.log(np.clip(f, 1e-10, 1.0))

        else:
            # Custom function F: f → ϕ
            return self.custom_fn(f)
```

**Choice of F:** The papers suggest f(x) ≈ 1 + αϕ in weak-field limit, so `ϕ ∝ (f-1)` is natural.
For horizon-like regions where f → 0, `ϕ ∝ log(f)` captures the divergence.

---

#### 6.2.2 ϕ_Poisson: Theoretical Reference (Analytic Comparison Mode)

This is NOT the engine's potential. It's a theoretical prediction:
"If activity distribution is ρ, Newtonian gravity predicts this ϕ."

**Used for:** Validating that ϕ_engine matches Newtonian expectations.

**Note:** The Poisson equation ∇²ϕ = κρ is an *approximation* valid in quasi-static regimes.
It is NOT fundamental to the engine — the engine doesn't solve Poisson.

```python
class PoissonSolver:
    """
    Solve discrete Poisson equation: ∇²ϕ = κ·ρ

    PURPOSE: Compute ϕ_Poisson for ANALYTIC COMPARISON only.
    This is a theoretical reference, not the engine's dynamics.

    Typical use:
        rho = AnalyticSourceDistribution(...)  # Analyst specifies
        phi_poisson = solver.solve(rho)        # Theoretical prediction
        phi_engine = PhiFromF().compute(lattice.f)  # Engine's actual result
        compare(phi_engine, phi_poisson)       # Do they match?

    Uses scipy.sparse for efficient solution on 2D grid.
    """

    def __init__(self, kappa: float = 1.0, boundary: str = "periodic"):
        self.kappa = kappa
        self.boundary = boundary
        self._laplacian_matrix = None  # Cached sparse Laplacian

    def _build_laplacian(self, ny: int, nx: int) -> sp.sparse.csr_matrix:
        """Build sparse 2D discrete Laplacian matrix."""
        n = ny * nx

        # 5-point stencil Laplacian
        diagonals = []
        offsets = []

        # Main diagonal: -4
        diagonals.append(-4 * np.ones(n))
        offsets.append(0)

        # Horizontal neighbors: +1
        diag_h = np.ones(n)
        if self.boundary != "periodic":
            # Zero out connections at boundaries
            for j in range(ny):
                diag_h[j * nx] = 0  # Left edge, no left neighbor
        diagonals.append(diag_h[:-1])
        offsets.append(1)
        diagonals.append(diag_h[:-1])
        offsets.append(-1)

        # Vertical neighbors: +1
        diag_v = np.ones(n - nx)
        diagonals.append(diag_v)
        offsets.append(nx)
        diagonals.append(diag_v)
        offsets.append(-nx)

        L = sp.sparse.diags(diagonals, offsets, shape=(n, n), format="csr")

        # Handle periodic boundaries
        if self.boundary == "periodic":
            # Add wrap-around connections
            L = self._add_periodic_connections(L, ny, nx)

        return L

    def solve(self, rho_act: np.ndarray) -> np.ndarray:
        """
        Solve Δϕ = κ * ρ_act for ϕ.

        Returns:
            phi: Solution field, shape [ny, nx]
        """
        ny, nx = rho_act.shape

        # Build or retrieve cached Laplacian
        if self._laplacian_matrix is None:
            self._laplacian_matrix = self._build_laplacian(ny, nx)

        # Flatten RHS
        b = self.kappa * rho_act.flatten()

        # Solve using conjugate gradient (Laplacian is symmetric)
        # Note: Poisson with periodic BC is singular; fix one point or use regularization
        if self.boundary == "periodic":
            # Regularize: subtract mean to ensure solution exists
            b = b - b.mean()

        phi_flat, info = sp.sparse.linalg.cg(self._laplacian_matrix, b, tol=1e-8)

        if info != 0:
            # Fall back to direct solve
            phi_flat = sp.sparse.linalg.spsolve(self._laplacian_matrix, b)

        return phi_flat.reshape((ny, nx))

    def solve_iterative(
        self,
        phi_current: np.ndarray,
        rho_act: np.ndarray,
        n_iterations: int = 10
    ) -> np.ndarray:
        """
        Incremental Jacobi/Gauss-Seidel relaxation for real-time update.

        Useful for running a few relaxation sweeps per tick instead of
        full solve.
        """
        phi = phi_current.copy()
        ny, nx = phi.shape

        for _ in range(n_iterations):
            # Jacobi update: phi_new[i,j] = (sum of neighbors + kappa*rho) / 4
            phi_new = np.zeros_like(phi)
            phi_new[1:-1, 1:-1] = (
                phi[:-2, 1:-1] + phi[2:, 1:-1] +  # vertical neighbors
                phi[1:-1, :-2] + phi[1:-1, 2:] +  # horizontal neighbors
                self.kappa * rho_act[1:-1, 1:-1]
            ) / 4.0

            # Handle boundaries according to config
            phi_new = self._apply_boundary(phi_new, phi, ny, nx)

            phi = phi_new

        return phi
```

---

## 7. Pattern Layer

Patterns are structured configurations that live in the engine and experience f(x) directly.

**Two modes of pattern dynamics:**

| Mode | Description | Use Case |
|------|-------------|----------|
| **Engine-Native** | Pattern state lives in lattice; motion emerges from message-passing and local f(x) | "True" behavior, faithful to papers |
| **Geodesic-Approximate** | Position integrated using ∇ϕ as effective force | Fast exploration, NOT the real dynamics |

For clocks, only engine-native makes sense (they tick on realized updates).
For particles, both modes are available; geodesic-approximate is faster but less faithful.

---

### 7.1 Pattern Base Class

```python
@dataclass
class PatternConfig:
    pattern_id: str
    world_tag: int              # 0 = default visible world
    center: tuple[int, int]     # Initial center position
    extent: int                 # Radius in lattice units
    activity_profile: Literal["point", "gaussian", "uniform"]
    activity_strength: float

class Pattern(ABC):
    """Base class for all patterns in the engine."""

    def __init__(self, config: PatternConfig, lattice: Lattice):
        self.config = config
        self.lattice = lattice
        self.proper_time = 0
        self.trajectory: list[tuple[int, float, float]] = []  # (tick, x, y)

    @abstractmethod
    def initialize(self):
        """Set up initial state in the lattice."""
        ...

    @abstractmethod
    def update(self, tick: int):
        """Called after each tick to update pattern state."""
        ...

    @abstractmethod
    def get_center(self) -> tuple[float, float]:
        """Return current center of mass / position."""
        ...

    def contribute_activity(self) -> np.ndarray:
        """
        Return activity contribution of this pattern.
        Shape [ny, nx], added to lattice rho_act.
        """
        ny, nx = self.lattice.config.ny, self.lattice.config.nx
        contrib = np.zeros((ny, nx))

        cx, cy = self.config.center
        r = self.config.extent

        if self.config.activity_profile == "point":
            contrib[cy, cx] = self.config.activity_strength

        elif self.config.activity_profile == "gaussian":
            y, x = np.ogrid[:ny, :nx]
            dist_sq = (x - cx)**2 + (y - cy)**2
            contrib = self.config.activity_strength * np.exp(-dist_sq / (2 * r**2))

        elif self.config.activity_profile == "uniform":
            y, x = np.ogrid[:ny, :nx]
            mask = (np.abs(x - cx) <= r) & (np.abs(y - cy) <= r)
            contrib[mask] = self.config.activity_strength

        return contrib
```

### 7.2 Clock Pattern

```python
class ClockPattern(Pattern):
    """
    A localized clock that ticks with the local proper time.

    Used to measure gravitational time dilation.
    """

    def __init__(self, config: PatternConfig, lattice: Lattice):
        super().__init__(config, lattice)
        self.phase = 0
        self.clock_ticks = 0
        self.readings: list[tuple[int, int, int]] = []  # (canonical_tick, clock_tick, phase)

    def initialize(self):
        # Mark clock location in lattice aux channel
        cx, cy = self.config.center
        self.lattice.state[cy, cx, 3] = self.config.pattern_id_numeric

    def update(self, tick: int):
        cx, cy = self.config.center

        # Clock ticks when its host node completes a realized update
        # We detect this by checking if proper_time at this node increased
        current_tau = self.lattice.proper_time[cy, cx]

        if current_tau > self.proper_time:
            # Node updated this tick
            ticks_elapsed = current_tau - self.proper_time
            self.clock_ticks += ticks_elapsed
            self.proper_time = current_tau

            # Update phase (e.g., oscillator with period P)
            self.phase = (self.phase + ticks_elapsed) % self.config.get("period", 10)

        # Record reading
        self.readings.append((tick, self.clock_ticks, self.phase))

    def get_center(self) -> tuple[float, float]:
        return float(self.config.center[0]), float(self.config.center[1])
```

### 7.3 Massive Test Particle

Two implementations available:

#### 7.3.1 Engine-Native Particle (Faithful to Papers)

```python
class NativeWavePacket(Pattern):
    """
    A localized wave-packet whose state lives IN the lattice.

    Motion emerges from message-passing dynamics:
    - Packet is a structured perturbation in lattice state
    - Propagation happens through kernel updates
    - In regions with low f(x), updates are less frequent → packet slows down
    - No explicit ∇ϕ computation — "gravity" emerges from bandwidth limits

    This is the TRUE dynamics of the papers.
    """

    def __init__(self, config: PatternConfig, lattice: Lattice,
                 initial_momentum: tuple[float, float] = (0.0, 0.0)):
        super().__init__(config, lattice)
        self.momentum = np.array(initial_momentum, dtype=float)
        self._last_center = np.array(config.center, dtype=float)

    def initialize(self):
        # Inject initial wave packet into lattice state channels
        # The packet is a localized bump in state[..., 0] (activity/mass channel)
        # with momentum encoded in state[..., 1:3]
        self._deposit_packet_to_lattice()

    def update(self, tick: int):
        # The packet moves because the KERNEL propagates state through the lattice
        # We just observe where the packet IS by finding its center of mass

        # Find current center from lattice state (not from integrating ∇ϕ!)
        center = self._find_center_of_mass()

        # Record trajectory
        self.trajectory.append((tick, center[0], center[1]))
        self._last_center = center

    def _find_center_of_mass(self) -> np.ndarray:
        """Locate packet center from lattice state (channel 0 = mass/activity)."""
        mass_field = self.lattice.state[:, :, 0]
        # Apply mask around expected location to avoid noise
        # ... center-of-mass calculation ...
        return center

    def get_center(self) -> tuple[float, float]:
        return self._last_center[0], self._last_center[1]
```

#### 7.3.2 Geodesic-Approximate Particle (Fast but Approximate)

```python
class GeodesicParticle(Pattern):
    """
    A test particle whose position is integrated using ∇ϕ.

    ⚠️  WARNING: This is an APPROXIMATION for fast exploration.
    It uses the derived ϕ field, not the true engine dynamics.
    Use NativeWavePacket for faithful behavior.

    Useful for:
    - Quick trajectory visualization
    - Comparison with Newtonian predictions
    - Large-scale surveys where native dynamics would be slow
    """

    def __init__(
        self,
        config: PatternConfig,
        lattice: Lattice,
        phi_field: np.ndarray,   # NOTE: requires pre-computed ϕ from analysis layer
        initial_velocity: tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__(config, lattice)
        self.phi = phi_field  # Reference to analysis-layer ϕ
        self.position = np.array(config.center, dtype=float)
        self.velocity = np.array(initial_velocity, dtype=float)

    def update(self, tick: int):
        # Compute effective force from ϕ gradient (DERIVED quantity!)
        grad_phi = self._compute_phi_gradient()
        acceleration = -grad_phi

        # Time step modulated by local f(x)
        dt = self.lattice.f[int(self.position[1]), int(self.position[0])]

        # Semi-implicit Euler
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        self._apply_boundaries()
        self.trajectory.append((tick, self.position[0], self.position[1]))

    def _compute_phi_gradient(self) -> np.ndarray:
        """Compute ∇ϕ at current position using central differences."""
        x, y = int(self.position[0]), int(self.position[1])
        ny, nx = self.phi.shape
        x, y = np.clip(x, 1, nx - 2), np.clip(y, 1, ny - 2)

        grad_x = (self.phi[y, x + 1] - self.phi[y, x - 1]) / 2.0
        grad_y = (self.phi[y + 1, x] - self.phi[y - 1, x]) / 2.0
        return np.array([grad_x, grad_y])

    def get_center(self) -> tuple[float, float]:
        return self.position[0], self.position[1]
```

### 7.4 Light-like Pattern

Like particles, light-like patterns can be implemented in two modes:

#### 7.4.1 Engine-Native Light (Faithful)

```python
class NativeLightPulse(Pattern):
    """
    A light-like disturbance that propagates through the lattice at max speed.

    The pulse is a structured perturbation in lattice state that propagates
    via kernel dynamics. In regions with low f(x), updates happen less often,
    so the pulse effectively slows down → gravitational lensing emerges.

    No explicit n_eff or ∇ϕ — lensing is a consequence of bandwidth limits.
    """

    def __init__(self, config: PatternConfig, lattice: Lattice,
                 initial_direction: tuple[float, float]):
        super().__init__(config, lattice)
        self.direction = np.array(initial_direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)

    def initialize(self):
        # Inject pulse into lattice — a sharp, localized perturbation
        # designed to propagate at maximum kernel speed
        self._inject_pulse()

    def update(self, tick: int):
        # Track pulse by finding its peak in lattice state
        # The pulse moves because the kernel propagates it
        center = self._find_pulse_peak()
        self.trajectory.append((tick, center[0], center[1]))

    def _find_pulse_peak(self) -> np.ndarray:
        """Locate pulse center from lattice state."""
        # Find peak of relevant state channel
        # ... implementation ...
        return center
```

#### 7.4.2 Geodesic-Approximate Light (Fast)

```python
class GeodesicLightRay(Pattern):
    """
    A light ray whose path is integrated using effective refractive index.

    ⚠️  APPROXIMATION: Uses n_eff(x) ∝ 1/f(x), derived from the
    engine-primitive f(x). Bending computed via Snell's law analogy.

    Faithful to the EMERGENT behavior but not the underlying mechanism.
    """

    def __init__(
        self,
        config: PatternConfig,
        lattice: Lattice,
        direction: tuple[float, float]
    ):
        super().__init__(config, lattice)
        self.position = np.array(config.center, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)
        self.base_speed = 1.0  # Max speed in lattice units

    def update(self, tick: int):
        x, y = int(self.position[0]), int(self.position[1])

        # Effective speed from f(x) — this IS using a primitive
        f_local = self.lattice.f[y, x]
        effective_speed = self.base_speed * f_local

        # Refraction from ∇f (NOT ∇ϕ — staying closer to primitives)
        self._apply_refraction_from_f()

        # Move
        self.position += self.direction * effective_speed
        self.trajectory.append((tick, self.position[0], self.position[1]))

    def _apply_refraction_from_f(self):
        """
        Bend ray toward regions of higher f(x).

        n_eff ∝ 1/f, so light bends toward high-f (fast) regions.
        Equivalent to bending toward lower ϕ in the Poisson approximation.
        """
        x, y = int(self.position[0]), int(self.position[1])
        f = self.lattice.f
        ny, nx = f.shape
        x, y = np.clip(x, 1, nx - 2), np.clip(y, 1, ny - 2)

        # Gradient of f (not ϕ!)
        grad_f = np.array([
            (f[y, x + 1] - f[y, x - 1]) / 2.0,
            (f[y + 1, x] - f[y - 1, x]) / 2.0
        ])

        # Bend toward higher f (perpendicular component)
        perp = grad_f - np.dot(grad_f, self.direction) * self.direction
        self.direction += 0.1 * perp
        self.direction /= np.linalg.norm(self.direction)
```

---

## 8. World Tags and Dark Matter

### 8.1 World Tag System

Each world is defined by its `SourceMap`. The engine doesn't know about worlds — it just runs on the combined source map.

```python
@dataclass
class WorldConfig:
    world_id: int
    name: str
    color: str  # For visualization
    source_map: SourceMap  # THIS WORLD'S activity sources

class WorldManager:
    """
    Manage multiple quasi-classical worlds sharing the same engine.

    Key insight: The engine runs on the SUM of all SourceMaps.
    It doesn't know which sources belong to which world.
    World tags are bookkeeping for the analyst, not for the engine.
    """

    def __init__(self, ny: int, nx: int):
        self.ny = ny
        self.nx = nx
        self.worlds: dict[int, WorldConfig] = {}

    def register_world(self, config: WorldConfig):
        self.worlds[config.world_id] = config

    def get_engine_source_map(self) -> SourceMap:
        """
        Combine all worlds' SourceMaps into one.

        This is what the engine actually runs on.
        The engine doesn't know about world tags.
        """
        combined = SourceMap(self.ny, self.nx, background_rate=0)
        for world in self.worlds.values():
            combined.rates += world.source_map.rates
        return combined

    def get_world_source_map(self, world_id: int) -> SourceMap:
        """Get a single world's SourceMap (for ρ_visible computation)."""
        return self.worlds[world_id].source_map

    def get_rho_visible(self, observer_world: int) -> np.ndarray:
        """
        Activity density visible to an observer in a given world.

        For v1: observers only see their own world's sources.
        """
        return self.worlds[observer_world].source_map.rates.copy()

    def get_rho_engine(self) -> np.ndarray:
        """Total activity density from ALL worlds (what drives f(x))."""
        return self.get_engine_source_map().rates
```

### 8.2 Dark Matter Experiment Support

```python
class DarkMatterAnalyzer:
    """
    Analyze dark matter-like effects from hidden world activity.

    Compares:
    - ϕ_engine: What the engine actually produces (from measured f(x))
    - ϕ_Poisson(ρ_visible): What observers EXPECT from visible matter
    - Gap = "dark matter" effect
    """

    def __init__(self, world_manager: WorldManager, poisson: PoissonSolver):
        self.world_manager = world_manager
        self.poisson = poisson
        self.phi_from_f = PhiFromF(mode="linear")

    def analyze(
        self,
        lattice: Lattice,
        observer_world: int
    ) -> dict:
        """
        Compare visible-matter prediction with actual engine behavior.

        Returns dict with:
            - rho_visible: SourceMap rates for observer's world
            - rho_engine: Combined SourceMap rates (all worlds)
            - rho_dark: rho_engine - rho_visible
            - phi_engine: ACTUAL potential from engine (derived from f(x))
            - phi_expected_visible: What observer expects from visible matter
            - phi_expected_total: What Poisson predicts from all sources
            - phi_dark: Apparent "missing" potential
        """
        # Activity densities (from SourceMap configurations, not measurements)
        rho_visible = self.world_manager.get_rho_visible(observer_world)
        rho_engine = self.world_manager.get_rho_engine()
        rho_dark = rho_engine - rho_visible

        # The ACTUAL potential (from engine's measured f(x))
        phi_engine = self.phi_from_f.compute(lattice.f)

        # Theoretical predictions (from Poisson)
        phi_expected_visible = self.poisson.solve(rho_visible)
        phi_expected_total = self.poisson.solve(rho_engine)

        # Dark component: what observers can't explain from visible matter
        # This is the gap between what they measure (phi_engine)
        # and what they expect (phi_expected_visible)
        phi_dark = phi_engine - phi_expected_visible

        return {
            # Activity densities (from SourceMap)
            "rho_visible": rho_visible,
            "rho_engine": rho_engine,
            "rho_dark": rho_dark,

            # Potentials
            "phi_engine": phi_engine,              # What engine produces
            "phi_expected_visible": phi_expected_visible,  # What observer expects
            "phi_expected_total": phi_expected_total,      # Theoretical (all sources)
            "phi_dark": phi_dark,                  # The "missing" potential

            # Summary stats
            "dark_mass_fraction": np.abs(rho_dark).sum() / (np.abs(rho_engine).sum() + 1e-10),
        }
```

### 8.3 Dark Matter Experiment Workflow

```python
# 1. Set up worlds
world_manager = WorldManager(ny=100, nx=100)

# World A: visible galaxy
source_A = SourceMap(100, 100)
source_A.add_gaussian_source(50, 50, peak_rate=10.0, sigma=5)
world_manager.register_world(WorldConfig(0, "visible", "blue", source_A))

# World B: hidden halo (dark matter analogue)
source_B = SourceMap(100, 100)
source_B.add_ring_source(50, 50, radius=25, width=8, rate=5.0)
world_manager.register_world(WorldConfig(1, "hidden", "gray", source_B))

# 2. Run engine on combined sources
engine_source_map = world_manager.get_engine_source_map()
# ... run engine with engine_source_map ...
# ... engine measures f(x) from congestion ...

# 3. Analyze dark matter effect
analyzer = DarkMatterAnalyzer(world_manager, PoissonSolver())
results = analyzer.analyze(lattice, observer_world=0)

# 4. Observer in world A sees:
#    - Their visible matter (source_A)
#    - But ϕ_engine is deeper than phi_expected_visible
#    - The gap (phi_dark) is "dark matter"
```

---

## 9. Experiment Harness

### 9.1 Experiment Configuration

```python
@dataclass
class ExperimentConfig:
    name: str
    description: str

    # Lattice setup
    lattice: LatticeConfig

    # Kernel selection
    kernel_type: str
    kernel_params: dict

    # Patterns to create
    patterns: list[PatternConfig]

    # Simulation parameters
    canonical_ticks: int
    output_interval: int

    # Analysis options
    compute_phi_mode: Literal["empirical", "poisson", "both"]
    track_trajectories: bool
    measure_time_dilation: bool

    # Output
    output_dir: Path
    export_format: Literal["csv", "json", "hdf5"]
```

### 9.2 Experiment Runner

```python
class ExperimentRunner:
    """Run and manage simulation experiments."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.lattice = None
        self.scheduler = None
        self.patterns = []
        self.measurements = []

    def setup(self):
        """Initialize lattice, kernel, and patterns."""
        self.lattice = Lattice(self.config.lattice)

        kernel = self._create_kernel()
        sizer = SimpleDeltaSizer()
        self.scheduler = Scheduler(self.lattice, kernel, sizer)

        for pattern_config in self.config.patterns:
            pattern = self._create_pattern(pattern_config)
            pattern.initialize()
            self.patterns.append(pattern)

    def run(self, progress_callback: Callable[[int, int], None] | None = None):
        """Execute the simulation."""
        for tick in range(self.config.canonical_ticks):
            # Advance engine
            self.scheduler.tick()

            # Update patterns
            for pattern in self.patterns:
                pattern.update(tick)

            # Compute ϕ if needed
            if tick % self.config.output_interval == 0:
                self._update_phi()
                self._record_measurements(tick)

            # Progress callback
            if progress_callback:
                progress_callback(tick, self.config.canonical_ticks)

    def export_results(self):
        """Export measurements and final state."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export field snapshots
        self._export_fields(output_dir)

        # Export pattern trajectories
        if self.config.track_trajectories:
            self._export_trajectories(output_dir)

        # Export time dilation data
        if self.config.measure_time_dilation:
            self._export_clock_readings(output_dir)
```

### 9.3 Pre-built Scenarios

```python
class Scenarios:
    """Factory for standard experiment configurations."""

    @staticmethod
    def time_dilation_test(
        grid_size: int = 100,
        source_strength: float = 10.0,
        source_position: tuple[int, int] = (50, 50),
        clock_positions: list[tuple[int, int]] = [(50, 50), (10, 10), (90, 90)]
    ) -> ExperimentConfig:
        """Create time dilation experiment with one source and multiple clocks."""
        patterns = []

        # Static source
        patterns.append(PatternConfig(
            pattern_id="source",
            world_tag=0,
            center=source_position,
            extent=3,
            activity_profile="gaussian",
            activity_strength=source_strength
        ))

        # Clocks at various distances
        for i, pos in enumerate(clock_positions):
            patterns.append(PatternConfig(
                pattern_id=f"clock_{i}",
                world_tag=0,
                center=pos,
                extent=1,
                activity_profile="point",
                activity_strength=0.1  # Clocks have minimal activity
            ))

        return ExperimentConfig(
            name="time_dilation",
            description="Measure gravitational time dilation around a static source",
            lattice=LatticeConfig(nx=grid_size, ny=grid_size, ...),
            patterns=patterns,
            canonical_ticks=1000,
            measure_time_dilation=True,
            ...
        )

    @staticmethod
    def free_fall_test(...) -> ExperimentConfig:
        """Create free-fall trajectory experiment."""
        ...

    @staticmethod
    def lensing_test(...) -> ExperimentConfig:
        """Create gravitational lensing experiment."""
        ...

    @staticmethod
    def dark_matter_test(...) -> ExperimentConfig:
        """Create dark matter analogue experiment."""
        ...
```

---

## 10. Visualization

### 10.1 Field Visualization

```python
class FieldVisualizer:
    """Visualize scalar fields on the lattice."""

    def plot_field(
        self,
        field: np.ndarray,
        title: str,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        overlay_patterns: list[Pattern] | None = None
    ) -> plt.Figure:
        """Create a heatmap of a 2D field."""
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        plt.colorbar(im, ax=ax, label=title)

        if overlay_patterns:
            for pattern in overlay_patterns:
                x, y = pattern.get_center()
                ax.plot(x, y, 'o', markersize=8, label=pattern.config.pattern_id)
            ax.legend()

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)

        return fig

    def plot_phi_comparison(
        self,
        phi_engine: np.ndarray,
        phi_visible: np.ndarray,
        phi_dark: np.ndarray
    ) -> plt.Figure:
        """Compare engine-level, visible, and dark potentials."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, field, title in zip(
            axes,
            [phi_engine, phi_visible, phi_dark],
            ["Engine ϕ (total)", "Visible ϕ", "Dark ϕ"]
        ):
            im = ax.imshow(field, cmap="RdBu", origin="lower")
            plt.colorbar(im, ax=ax)
            ax.set_title(title)

        fig.tight_layout()
        return fig
```

### 10.2 Trajectory Visualization

```python
class TrajectoryVisualizer:
    """Visualize pattern trajectories and geodesics."""

    def plot_trajectories(
        self,
        patterns: list[Pattern],
        phi: np.ndarray,
        show_phi_background: bool = True
    ) -> plt.Figure:
        """Plot particle/light trajectories over the potential field."""
        fig, ax = plt.subplots(figsize=(10, 10))

        if show_phi_background:
            im = ax.imshow(phi, cmap="Greys", origin="lower", alpha=0.5)

        colors = plt.cm.tab10.colors
        for i, pattern in enumerate(patterns):
            if pattern.trajectory:
                ticks, xs, ys = zip(*pattern.trajectory)
                ax.plot(xs, ys, '-', color=colors[i % 10],
                       label=pattern.config.pattern_id, linewidth=2)
                # Mark start and end
                ax.plot(xs[0], ys[0], 'o', color=colors[i % 10], markersize=10)
                ax.plot(xs[-1], ys[-1], 's', color=colors[i % 10], markersize=10)

        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Pattern Trajectories")

        return fig
```

---

## 11. Testing Strategy

### 11.1 Unit Tests

**Core Components:**

| Component | Test Focus |
|-----------|-----------|
| `Lattice` | Array initialization, boundary handling, state access |
| `MessageQueue` | FIFO ordering, capacity accounting |
| `Kernel` | Conservation laws, reversibility, output shape |
| `PoissonSolver` | Known analytical solutions, convergence |
| `Pattern` | Initialization, update mechanics, trajectory recording |

**Example Unit Test:**

```python
def test_poisson_point_source():
    """Poisson solution for point source should approximate 1/r in 2D."""
    solver = PoissonSolver(kappa=1.0, boundary="absorbing")

    rho = np.zeros((101, 101))
    rho[50, 50] = 100.0  # Point source at center

    phi = solver.solve(rho)

    # Check radial profile
    for r in range(5, 45, 5):
        phi_at_r = phi[50, 50 + r]
        phi_at_2r = phi[50, 50 + 2*r] if 50 + 2*r < 101 else None

        if phi_at_2r is not None:
            # In 2D, phi ~ log(r), so phi(r) - phi(2r) ~ log(2)
            diff = phi_at_r - phi_at_2r
            assert np.abs(diff - np.log(2)) < 0.5, f"Failed at r={r}"
```

### 11.2 Integration Tests

**Experiment Validation:**

```python
class TestTimeDilation:
    """Integration tests for time dilation behavior."""

    def test_clock_near_source_runs_slow(self):
        """Clock near high-activity source should tick slower than distant clock."""
        config = Scenarios.time_dilation_test(
            grid_size=100,
            source_strength=50.0,
            source_position=(50, 50),
            clock_positions=[(55, 50), (90, 50)]  # Near and far
        )

        runner = ExperimentRunner(config)
        runner.setup()
        runner.run()

        clock_near = runner.patterns[1]  # clock at (55, 50)
        clock_far = runner.patterns[2]   # clock at (90, 50)

        # After same canonical time, near clock should have fewer ticks
        assert clock_near.clock_ticks < clock_far.clock_ticks

        # Ratio should depend on phi difference
        phi_near = runner.lattice.phi[50, 55]
        phi_far = runner.lattice.phi[50, 90]
        expected_ratio = (1 + phi_near) / (1 + phi_far)  # Approximate
        actual_ratio = clock_near.clock_ticks / clock_far.clock_ticks

        assert np.abs(actual_ratio - expected_ratio) < 0.1

class TestFreeFall:
    """Integration tests for free-fall trajectories."""

    def test_particle_curves_toward_source(self):
        """Particle with tangential velocity should curve toward source."""
        ...

class TestDarkMatter:
    """Integration tests for dark matter analogue behavior."""

    def test_hidden_world_contributes_to_phi(self):
        """Activity in hidden world should affect engine-level phi."""
        ...
```

### 11.3 Sanity Checks

```python
class TestSanityChecks:
    """Basic sanity checks as specified in requirements."""

    def test_flat_background(self):
        """With no sources, f(x) ≈ 1, phi ~ constant, straight trajectories."""
        config = ExperimentConfig(
            lattice=LatticeConfig(nx=50, ny=50, ...),
            patterns=[],  # No activity sources
            canonical_ticks=100,
            ...
        )

        runner = ExperimentRunner(config)
        runner.setup()
        runner.run()

        # f should be close to 1 everywhere
        assert np.allclose(runner.lattice.f, 1.0, atol=0.05)

        # phi should be nearly constant
        phi_std = runner.lattice.phi.std()
        assert phi_std < 0.01

    def test_memory_scaling(self):
        """Memory should scale O(N) with number of nodes."""
        import tracemalloc

        sizes = [50, 100, 200]
        memories = []

        for size in sizes:
            tracemalloc.start()
            lattice = Lattice(LatticeConfig(nx=size, ny=size, ...))
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memories.append(peak)

        # Check scaling is roughly linear
        # mem(200x200) / mem(50x50) should be ~ (200*200) / (50*50) = 16
        ratio = memories[2] / memories[0]
        expected_ratio = (sizes[2]**2) / (sizes[0]**2)

        assert 0.5 * expected_ratio < ratio < 2.0 * expected_ratio
```

---

## 12. Performance Considerations

### 12.1 Vectorization Strategy

All field operations use NumPy vectorization:

```python
# BAD: Python loop
for i in range(ny):
    for j in range(nx):
        f[i, j] = realized[i, j] / possible[i, j]

# GOOD: Vectorized
f = realized / possible
```

### 12.2 Hot Path Optimization

If performance becomes an issue, these paths are candidates for Numba JIT:

1. `Scheduler.tick()` — main update loop
2. `Kernel.compute_update()` — kernel application
3. `PoissonSolver.solve_iterative()` — relaxation sweeps
4. `Pattern.update()` — trajectory integration

```python
from numba import njit

@njit
def apply_kernel_batch(states, incoming, D, c, gamma):
    """JIT-compiled kernel application across all nodes."""
    ny, nx, nc = states.shape
    new_states = np.empty_like(states)

    for j in range(ny):
        for i in range(nx):
            # Kernel logic here
            ...

    return new_states
```

### 12.3 Memory Optimization

For very large grids, consider:

- **Sparse message queues**: Only store non-empty queues
- **Reduced precision**: Use `float32` instead of `float64`
- **Tiled updates**: Process grid in tiles to improve cache locality

---

## 13. Future Extensions (v2+)

### 13.1 Complex Hilbert Space Dynamics

Upgrade node state to true complex vectors with interference:

```python
@dataclass
class QuantumNodeState:
    amplitudes: np.ndarray  # Complex, shape [n_basis_states]

    def norm(self) -> float:
        return np.sum(np.abs(self.amplitudes)**2)
```

### 13.2 Agent Patterns

```python
class AgentPattern(Pattern):
    """Agent with internal memory and decision dynamics."""

    def __init__(self, ...):
        self.memory: list[Any] = []
        self.policy: Callable = default_policy

    def update(self, tick: int):
        # Perceive local environment
        observation = self._observe()

        # Update memory
        self.memory.append(observation)

        # Decide action based on policy
        action = self.policy(self.memory)

        # Execute action (move, interact, etc.)
        self._execute(action)
```

### 13.3 Dynamic Topology

Allow links to be created/destroyed, enabling:
- Emergent geometry
- Topology changes near horizon regions

---

## 14. Glossary

### Primitives (What the Engine Actually Computes)

| Term | Definition |
|------|-----------|
| **Canonical time** T_can | Engine-level tick counter; counts update *opportunities* |
| **Proper time** τ(x) | Accumulated count of *realized* updates at node x — THE fundamental clock |
| **Update fraction** f(x) | realized_updates / possible_updates — THIS IS "GRAVITY" |
| **Link capacity** C | Max bits per link per canonical tick |
| **Message queue** | Pending messages waiting for capacity |
| **Waiting** | Node blocked because inputs not arrived or queue not cleared |

### Derived Quantities (Analysis Tools, NOT Engine Primitives)

| Term | Definition |
|------|-----------|
| **Slowness** λ(x) | 1 - f(x); trivially derived, not separately stored |
| **Time-sag** ϕ(x) | Smooth summary of f(x) pattern; useful for visualization and Newtonian comparison |
| **Activity density** ρ_act(x) | Aggregate activity measure used to source Poisson equation for ϕ |
| **Poisson equation** | ∇²ϕ = κρ_act — an *approximation* valid in quasi-static regimes, NOT fundamental |
| **Effective refractive index** n_eff | 1/f(x) — useful for geodesic-approximate light propagation |

### Other Terms

| Term | Definition |
|------|-----------|
| **World tag** | Label distinguishing quasi-classical patterns (for dark matter experiments) |
| **Engine-level** | Computed from full lattice state, shared across all worlds |
| **Engine-native** | Pattern dynamics from actual message-passing, not from derived fields |
| **Geodesic-approximate** | Pattern dynamics from integrating ∇ϕ — faster but not faithful |

---

## 15. References

- [Paper 1] "The Universe as a Message Passing Network" — Conceptual substrate
- [Paper 2] "Gravity as Bandwidth-Limited Synchronization" — Time-sag phenomenology
- NumPy documentation: https://numpy.org/doc/
- SciPy sparse solvers: https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
