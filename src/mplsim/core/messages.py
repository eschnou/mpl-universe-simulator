"""
Messages and link queues for the message-passing engine.

Each link has finite capacity. When a node generates more message traffic
than capacity allows, the excess queues up. This creates congestion,
which creates waiting, which reduces f(x).

This is the mechanism that creates "gravity".

Additionally, nodes must wait for inputs from neighbors before updating.
This creates synchronization pressure: when a neighbor is slow, we wait longer.
This is the β·⟨λ⟩_x term in the paper's self-consistency equation.
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np


# Direction mappings for message delivery
OPPOSITE_DIRECTION = {
    "N": "S", "S": "N", "E": "W", "W": "E",
    "NE": "SW", "NW": "SE", "SE": "NW", "SW": "NE",
}

DIRECTION_DELTAS = {
    "N": (0, -1),   # North: y decreases
    "S": (0, 1),    # South: y increases
    "E": (1, 0),    # East: x increases
    "W": (-1, 0),   # West: x decreases
    "NE": (1, -1),
    "NW": (-1, -1),
    "SE": (1, 1),
    "SW": (-1, 1),
}


@dataclass
class Message:
    """
    A message sent from one node to a neighbor.

    The content doesn't matter much for gravity experiments —
    what matters is size_bits for capacity accounting.
    """

    source_x: int
    source_y: int
    size_bits: float  # Information size — this is what saturates links
    tick_created: int  # Canonical tick when generated


class LinkQueues:
    """
    Manages outbound message queues for all links in all directions.

    Each node has one outbound queue per direction.
    Queues have capacity limits per tick.

    Structure:
    - For each direction (N, S, E, W, ...), we track:
      - Pending message sizes per link: how much is waiting to be sent
      - Sent this tick: how much has been sent this tick
      - Received this tick: whether inputs arrived from each direction

    The "received" tracking enables synchronization pressure:
    - A node can only update after receiving inputs from all neighbors
    - Slow neighbors delay us → our f drops → synchronization propagates
    """

    def __init__(
        self,
        ny: int,
        nx: int,
        directions: list[str],
        capacity: float,
    ):
        self.ny = ny
        self.nx = nx
        self.directions = directions
        self.capacity = capacity

        # Pending bits to send per node per direction
        # Shape: {direction: [ny, nx]}
        self.pending: dict[str, np.ndarray] = {
            d: np.zeros((ny, nx), dtype=np.float64)
            for d in directions
        }

        # Bits sent this tick per node per direction
        self.sent_this_tick: dict[str, np.ndarray] = {
            d: np.zeros((ny, nx), dtype=np.float64)
            for d in directions
        }

        # Track whether each node received input from each direction this tick
        # Key is the direction FROM which input was received (opposite of sender's direction)
        # Shape: {direction: [ny, nx]} where True = received input from that neighbor
        self.received_this_tick: dict[str, np.ndarray] = {
            d: np.zeros((ny, nx), dtype=bool)
            for d in directions
        }

    def enqueue(self, direction: str, x: int, y: int, size_bits: float):
        """Add message to outbound queue for a link."""
        self.pending[direction][y, x] += size_bits

    def get_pending(self, direction: str, x: int, y: int) -> float:
        """Get pending bits for a specific link."""
        return self.pending[direction][y, x]

    def get_total_pending(self, x: int, y: int) -> float:
        """Get total pending bits across all outbound links for a node."""
        return sum(self.pending[d][y, x] for d in self.directions)

    def can_send(self, direction: str, x: int, y: int, size_bits: float) -> bool:
        """Check if there's enough capacity to send this tick."""
        already_sent = self.sent_this_tick[direction][y, x]
        return already_sent + size_bits <= self.capacity

    def available_capacity(self, direction: str, x: int, y: int) -> float:
        """Get remaining capacity for this tick."""
        return max(0.0, self.capacity - self.sent_this_tick[direction][y, x])

    def send(self, direction: str, x: int, y: int, size_bits: float) -> float:
        """
        Attempt to send bits on a link.

        Returns: actual bits sent (may be less than requested if capacity hit)
        """
        available = self.available_capacity(direction, x, y)
        actual_sent = min(size_bits, available)

        self.sent_this_tick[direction][y, x] += actual_sent
        self.pending[direction][y, x] -= actual_sent

        return actual_sent

    def process_queues(self) -> np.ndarray:
        """
        Process all queues: send as much as capacity allows.

        Also marks receiving nodes as having received input from the sending direction.
        This enables synchronization pressure: nodes wait for neighbor inputs.

        Returns:
            cleared: [ny, nx] bool array — True where ALL queues cleared
        """
        ny, nx = self.ny, self.nx

        for direction in self.directions:
            pending = self.pending[direction]
            sent = self.sent_this_tick[direction]

            # How much can we send?
            can_send = np.minimum(pending, self.capacity - sent)

            # Send it
            sent += can_send
            pending -= can_send

            # Mark receiving nodes as having received input
            # When node (y,x) sends in direction D, the receiver gets input from opposite(D)
            self._mark_receivers(direction, can_send > 0)

        # Check which nodes have cleared all their queues
        cleared = np.ones((ny, nx), dtype=bool)
        for direction in self.directions:
            cleared &= (self.pending[direction] < 1e-10)

        return cleared

    def _mark_receivers(self, send_direction: str, sent_mask: np.ndarray):
        """
        Mark receiving nodes as having received input.

        When a node sends in direction D, the receiving node gets input from opposite(D).
        Uses periodic boundary conditions.
        """
        ny, nx = self.ny, self.nx
        dx, dy = DIRECTION_DELTAS[send_direction]
        receive_direction = OPPOSITE_DIRECTION[send_direction]

        # Shift the sent_mask to receiver positions (periodic wrapping)
        # If sender sends North (dy=-1), receiver is at y-1, receiving from South
        receiver_mask = np.roll(np.roll(sent_mask, dy, axis=0), dx, axis=1)

        # Mark receivers
        self.received_this_tick[receive_direction] |= receiver_mask

    def start_new_tick(self):
        """Reset per-tick sent counters. Call at start of each canonical tick."""
        for direction in self.directions:
            self.sent_this_tick[direction].fill(0.0)
        # NOTE: received_this_tick is NOT reset here.
        # It persists until nodes consume their inputs (via reset_received_for_nodes)

    def reset_received_for_nodes(self, updated_mask: np.ndarray):
        """
        Clear received tracking for nodes that successfully updated.

        When a node updates, it has "consumed" its inputs and must wait
        for fresh inputs from neighbors before updating again.

        Args:
            updated_mask: [ny, nx] bool array where True = node updated
        """
        for direction in self.directions:
            # Clear received flag for updated nodes
            self.received_this_tick[direction] &= ~updated_mask

    def get_congestion_map(self) -> np.ndarray:
        """
        Get total pending bits per node (summed over all directions).

        High values = congested nodes.
        """
        congestion = np.zeros((self.ny, self.nx), dtype=np.float64)
        for direction in self.directions:
            congestion += self.pending[direction]
        return congestion

    def queues_cleared(self, x: int, y: int) -> bool:
        """Check if all outbound queues for a node are empty."""
        return all(
            self.pending[d][y, x] < 1e-10
            for d in self.directions
        )

    def has_inputs_from_directions(
        self, y: int, x: int, required_directions: set[str]
    ) -> bool:
        """
        Check if a node has received input from all required directions this tick.

        Args:
            y, x: Node position
            required_directions: Set of directions from which input is required
                                 (e.g., {"N", "S", "E", "W"} for all von Neumann neighbors)

        Returns:
            True if input has been received from ALL required directions
        """
        return all(
            self.received_this_tick[d][y, x]
            for d in required_directions
            if d in self.received_this_tick
        )

    def get_received_mask(self) -> np.ndarray:
        """
        Get mask of nodes that received input from ALL directions this tick.

        Returns:
            [ny, nx] bool array — True where node received from all neighbors
        """
        received_all = np.ones((self.ny, self.nx), dtype=bool)
        for direction in self.directions:
            received_all &= self.received_this_tick[direction]
        return received_all
