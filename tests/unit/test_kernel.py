"""Unit tests for Kernel."""

import pytest
import numpy as np

from mplsim.core.kernel import LoadGeneratorKernel, create_default_kernel
from mplsim.core.source_map import SourceMap


class TestLoadGeneratorKernel:
    """Tests for LoadGeneratorKernel."""

    def test_creation(self):
        kernel = LoadGeneratorKernel(message_size=2.0)
        assert kernel.message_size == 2.0

    def test_default_kernel(self):
        kernel = create_default_kernel()
        assert isinstance(kernel, LoadGeneratorKernel)
        assert kernel.message_size == 1.0

    def test_required_inputs_empty(self):
        kernel = LoadGeneratorKernel()
        assert kernel.required_inputs == set()

    def test_compute_output_bits_background(self):
        kernel = LoadGeneratorKernel(message_size=1.0)
        source_map = SourceMap(ny=10, nx=10, background_rate=0.5)

        bits = kernel.compute_output_bits(y=5, x=5, source_map=source_map)
        assert bits == 0.5  # rate * message_size = 0.5 * 1.0

    def test_compute_output_bits_with_source(self):
        kernel = LoadGeneratorKernel(message_size=2.0)
        source_map = SourceMap(ny=10, nx=10, background_rate=0.1)
        source_map.add_point_source(x=5, y=5, rate=10.0)

        # At the source
        bits_at_source = kernel.compute_output_bits(y=5, x=5, source_map=source_map)
        assert bits_at_source == 20.0  # 10.0 * 2.0

        # Away from source
        bits_away = kernel.compute_output_bits(y=0, x=0, source_map=source_map)
        assert bits_away == 0.2  # 0.1 * 2.0

    def test_compute_output_bits_gaussian_source(self):
        kernel = LoadGeneratorKernel(message_size=1.0)
        source_map = SourceMap(ny=50, nx=50, background_rate=0.1)
        source_map.add_gaussian_source(cx=25, cy=25, peak_rate=10.0, sigma=5.0)

        # At center should be high
        bits_center = kernel.compute_output_bits(y=25, x=25, source_map=source_map)

        # Far away should be near background
        bits_far = kernel.compute_output_bits(y=0, x=0, source_map=source_map)

        assert bits_center > bits_far
        assert bits_center > 10.0  # background + peak
        assert bits_far < 0.5  # Near background
