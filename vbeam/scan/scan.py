
from abc import abstractmethod

from spekk import ops, Module, field
from vbeam.core.scan import TScan
from vbeam.geometry import as_cartesian
from vbeam.geometry import cartesian_to_polar

class Scan(Module):
    points: ops.array


class LinearScanGeometry(Module):
    x: ops.array
    y: ops.array
    z: ops.array

    def __post_init__(self):
        self._cartesian_points = ops.stack([self.x, self.y, self.z], axis="xyz")
    
    @property
    def points(self):
        return self._cartesian_points


class ScanConvetable(Module):

    @abstractmethod
    def from_cartesian_to_local_coordinates(self, cartesian_points) -> ops.array:
        """Convert Cartesian points to local coordinate system."""
        pass
    
    @abstractmethod
    def calculate_cartesian_bounds(self)-> tuple[float, float, float, float, float, float]:
        """Calculate the bounds of the cartesian coordinates."""
        pass
    
class SectorScanGeometry(ScanConvetable):
    azimuths: ops.array
    elevations: ops.array
    depths: ops.array

    def __post_init__(self):
        p = ops.stack([self.azimuths, self.elevations, self.depths], axis="az_el_depth")
        self._cartesian_points = as_cartesian(p)
    
    @property
    def points(self):
        return self._cartesian_points

    def from_cartesian_to_local_coordinates(self, cartesian_points) -> ops.array:
        return cartesian_to_polar(cartesian_points)

    def calculate_cartesian_bounds(self)-> tuple[float, float, float, float, float, float]:
        """
        Calculate the bounds of the cartesian coordinates.
        Note: Should be updated to calculate the bounds analytically
        """

        min_x = self.points["xyz", 0].min()
        max_x = self.points["xyz", 0].max()
        min_y = self.points["xyz", 1].min()
        max_y = self.points["xyz", 1].max()
        min_z = self.points["xyz", 2].min()
        max_z = self.points["xyz", 2].max()
        return (min_x, max_x, min_y, max_y, min_z, max_z)

class ScanLineSetup(Module):
    """
    Attributes:
        n_tx_azimuth (int): Number of transmitted waves across the azimuth axis.
        n_mla_azimuth (int): Number of beamformed (rx) lines per tx.
        n_mla_azimuth_after_bf (int): Number of beamformed lines per tx after RTB.
    """

    n_tx_azimuth: int = field(static=True)
    n_mla_azimuth: int = field(static=True)
    n_mla_azimuth_after_bf: int = field(static=True)

    @property
    def full_n_azimuths(self) -> int:
        """
        Return number of azimuth directions in the full grid, before removing MLA lines
        without full overlap in finalize."""
        return self.n_azimuths + self.n_mla_azimuth - self.n_mla_azimuth_after_bf

    @property
    def n_azimuths(self) -> int:
        """The number of azimuth directions with full overlap, i.e.: after finalize."""
        return self.n_tx_azimuth * self.n_mla_azimuth_after_bf

class SectorScanGeometryRTB(SectorScanGeometry):
    scan_line_setup: ScanLineSetup

