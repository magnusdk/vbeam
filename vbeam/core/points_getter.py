from abc import abstractmethod

from spekk import Module, ops


class PointsGetter(Module):
    "Something that can get 3D point positions, e.g.: a :class:`~vbeam.scan.Scan`."

    @abstractmethod
    def get_points(self) -> ops.array:
        "Get the 3D point positions."
