"""
Patterns: structured configurations that live in the engine.

Patterns experience f(x) directly. They don't know about ϕ or Poisson.
- ClockPattern: ticks on realized updates (measures time dilation)
- GeodesicParticle: position integrated using ∇f (approximate free fall)
- GeodesicLightRay: propagates at speed ∝ f(x) (lensing)
"""

from mplsim.patterns.base import Pattern, PatternConfig
from mplsim.patterns.clock import ClockPattern, ClockConfig, create_clock
from mplsim.patterns.particle import GeodesicParticle, ParticleConfig, create_particle

__all__ = [
    "Pattern",
    "PatternConfig",
    "ClockPattern",
    "ClockConfig",
    "create_clock",
    "GeodesicParticle",
    "ParticleConfig",
    "create_particle",
]
