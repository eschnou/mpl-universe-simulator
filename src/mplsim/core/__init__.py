"""
Core engine primitives.

This layer knows NOTHING about gravity, potentials, or Poisson equations.
It only knows:
- Nodes with local state
- Links with capacity limits
- Messages and queues
- Whether a node can update (eligibility)
- Counting realized vs possible updates → f(x)
- Proper time τ(x) accumulation

Two schedulers are available:
- BandwidthScheduler: TRUE causal emergence from bandwidth + sync waiting
- TheoreticalScheduler: Direct computation of λ = γa + β⟨λ⟩ (fast, for testing)
"""

from mplsim.core.lattice import Lattice, LatticeConfig
from mplsim.core.messages import Message, LinkQueues
from mplsim.core.source_map import SourceMap
from mplsim.core.kernel import Kernel, LoadGeneratorKernel, create_default_kernel
from mplsim.core.bandwidth_scheduler import BandwidthScheduler, BandwidthSchedulerConfig
from mplsim.core.theoretical_scheduler import TheoreticalScheduler, TheoreticalSchedulerConfig

__all__ = [
    "Lattice",
    "LatticeConfig",
    "Message",
    "LinkQueues",
    "SourceMap",
    "Kernel",
    "LoadGeneratorKernel",
    "create_default_kernel",
    "BandwidthScheduler",
    "BandwidthSchedulerConfig",
    "TheoreticalScheduler",
    "TheoreticalSchedulerConfig",
]
