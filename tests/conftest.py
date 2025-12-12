"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np


@pytest.fixture
def small_grid_config():
    """Configuration for a small 50x50 test grid."""
    from mplsim.core import LatticeConfig
    return LatticeConfig(
        nx=50,
        ny=50,
        neighborhood="von_neumann",
        boundary="periodic",
        link_capacity=10.0,
    )


@pytest.fixture
def medium_grid_config():
    """Configuration for a medium 100x100 test grid."""
    from mplsim.core import LatticeConfig
    return LatticeConfig(
        nx=100,
        ny=100,
        neighborhood="von_neumann",
        boundary="periodic",
        link_capacity=10.0,
    )


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(seed=42)
