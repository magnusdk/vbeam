"""Interface for sampling the speed of sound on a line, typically the line between a 
:term:`sender` and a :term:`point`, and a :term:`point` and a :term:`receiver`. """

from abc import abstractmethod

from spekk import Module, ops


class SpeedOfSound(Module):
    @abstractmethod
    def average(
        self,
        sender_position: ops.array,
        sender: ops.array,
        receiver_position: ops.array,
    ) -> float:
        """Sample the speed of sound between the sender, the point position, and the
        receiver, and return the average.

        All positions are arrays of three elements: (x, y, z)."""
