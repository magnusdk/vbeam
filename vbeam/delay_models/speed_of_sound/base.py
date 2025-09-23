from abc import abstractmethod

from spekk import Module, ops


class SpeedOfSound(Module):
    @abstractmethod
    def get_delay_between(self, point1: ops.array, point2: ops.array, /) -> ops.array:
        """Return the elapsed time when traveling from point1 to point2.

        Both points must be in cartesian coordinates and have a dimension named "xyz"
        that indexes the coordinate components.
        """
