"""
mplsim: 2D Message-Passing Gravity Engine Simulator

A simulator demonstrating how gravity emerges from bandwidth-limited
message passing in a universe engine.

Core concepts:
- Activity creates message traffic
- Traffic creates congestion on links
- Congestion creates waiting (nodes can't update)
- Waiting reduces f(x) = realized_updates / possible_updates
- f(x) IS gravity: clocks tick slow where f is low

See design.md and requirements.md for full details.
"""

__version__ = "0.1.0"
