"""Unit tests for Message and LinkQueues."""

import numpy as np
import pytest

from mplsim.core.messages import Message, LinkQueues


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        msg = Message(source_x=10, source_y=20, size_bits=5.5, tick_created=42)
        assert msg.source_x == 10
        assert msg.source_y == 20
        assert msg.size_bits == 5.5
        assert msg.tick_created == 42


class TestLinkQueues:
    """Tests for LinkQueues."""

    def test_creation(self):
        queues = LinkQueues(ny=10, nx=10, directions=["N", "S", "E", "W"], capacity=10.0)
        assert queues.ny == 10
        assert queues.nx == 10
        assert queues.capacity == 10.0
        assert set(queues.directions) == {"N", "S", "E", "W"}

    def test_enqueue_and_get_pending(self):
        queues = LinkQueues(ny=10, nx=10, directions=["N", "S", "E", "W"], capacity=10.0)

        queues.enqueue("N", 5, 5, 3.0)
        queues.enqueue("N", 5, 5, 2.0)  # Add more to same link

        assert queues.get_pending("N", 5, 5) == 5.0
        assert queues.get_pending("S", 5, 5) == 0.0  # Other direction empty

    def test_total_pending(self):
        queues = LinkQueues(ny=10, nx=10, directions=["N", "S", "E", "W"], capacity=10.0)

        queues.enqueue("N", 5, 5, 3.0)
        queues.enqueue("S", 5, 5, 2.0)
        queues.enqueue("E", 5, 5, 1.0)

        assert queues.get_total_pending(5, 5) == 6.0

    def test_can_send_within_capacity(self):
        queues = LinkQueues(ny=10, nx=10, directions=["N", "S", "E", "W"], capacity=10.0)

        assert queues.can_send("N", 5, 5, 8.0) == True
        assert queues.can_send("N", 5, 5, 10.0) == True
        assert queues.can_send("N", 5, 5, 11.0) == False

    def test_can_send_accounts_for_already_sent(self):
        queues = LinkQueues(ny=10, nx=10, directions=["N", "S", "E", "W"], capacity=10.0)

        # Send 7 bits
        queues.send("N", 5, 5, 7.0)

        # Can only send 3 more
        assert queues.can_send("N", 5, 5, 3.0) == True
        assert queues.can_send("N", 5, 5, 4.0) == False

    def test_available_capacity(self):
        queues = LinkQueues(ny=10, nx=10, directions=["N", "S", "E", "W"], capacity=10.0)

        assert queues.available_capacity("N", 5, 5) == 10.0

        queues.send("N", 5, 5, 7.0)
        assert queues.available_capacity("N", 5, 5) == 3.0

    def test_send_returns_actual_sent(self):
        queues = LinkQueues(ny=10, nx=10, directions=["N", "S", "E", "W"], capacity=10.0)

        # Add to pending
        queues.enqueue("N", 5, 5, 15.0)

        # Try to send 12, but capacity is 10
        actual = queues.send("N", 5, 5, 12.0)
        assert actual == 10.0

        # Pending should be reduced by what was sent
        assert queues.get_pending("N", 5, 5) == 5.0

    def test_process_queues_clears_within_capacity(self):
        queues = LinkQueues(ny=10, nx=10, directions=["N", "S", "E", "W"], capacity=10.0)

        # Add small amounts that fit in capacity
        queues.enqueue("N", 5, 5, 3.0)
        queues.enqueue("S", 5, 5, 4.0)

        cleared = queues.process_queues()

        # This node should be cleared
        assert cleared[5, 5] == True
        assert queues.get_total_pending(5, 5) < 1e-9

    def test_process_queues_partial_clear(self):
        queues = LinkQueues(ny=10, nx=10, directions=["N", "S", "E", "W"], capacity=10.0)

        # Add more than capacity
        queues.enqueue("N", 5, 5, 15.0)

        cleared = queues.process_queues()

        # This node should NOT be cleared
        assert cleared[5, 5] == False
        assert queues.get_pending("N", 5, 5) == 5.0  # 15 - 10 remaining

    def test_start_new_tick_resets_sent(self):
        queues = LinkQueues(ny=10, nx=10, directions=["N", "S", "E", "W"], capacity=10.0)

        queues.send("N", 5, 5, 7.0)
        assert queues.available_capacity("N", 5, 5) == 3.0

        queues.start_new_tick()

        # Capacity should be restored
        assert queues.available_capacity("N", 5, 5) == 10.0

    def test_queues_cleared_check(self):
        queues = LinkQueues(ny=10, nx=10, directions=["N", "S", "E", "W"], capacity=10.0)

        assert queues.queues_cleared(5, 5) is True

        queues.enqueue("N", 5, 5, 3.0)
        assert queues.queues_cleared(5, 5) is False

        queues.process_queues()
        assert queues.queues_cleared(5, 5) is True

    def test_congestion_map(self):
        queues = LinkQueues(ny=10, nx=10, directions=["N", "S", "E", "W"], capacity=10.0)

        queues.enqueue("N", 5, 5, 3.0)
        queues.enqueue("S", 5, 5, 4.0)
        queues.enqueue("N", 7, 7, 10.0)

        congestion = queues.get_congestion_map()

        assert congestion[5, 5] == 7.0
        assert congestion[7, 7] == 10.0
        assert congestion[0, 0] == 0.0
