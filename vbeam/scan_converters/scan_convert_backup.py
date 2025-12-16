
from abc import abstractmethod
from typing import Type
from spekk import Module, ops, field
from vbeam.core import NDInterpolator, Coordinate
from vbeam.geometry.coordinate_systems import polar_to_cartesian
from vbeam.scan import SectorScanGeometry, SectorScanGeometryRTB, ScanConvetable
from vbeam.interpolation import LinearNDInterpolator, LinearCoordinate, IrregularSampledCoordinate
from vbeam.geometry import as_cartesian, as_polar, cartesian_to_polar

def scan_convert_impl(
    beamspace_data: ops.array,
    beamspace_coordinates: dict["str", Type[Coordinate]],
    interpolated_coordinates: dict["str", ops.array],
    fill_value: float | None,
    interpolator_type: Type[NDInterpolator],
):
    interpolator = interpolator_type(beamspace_coordinates, beamspace_data, fill_value=fill_value)
    interpolated_data = interpolator(interpolated_coordinates)
    return interpolated_data


class ScanConverter(Module):

    @abstractmethod
    def scan_convert(self, 
                     beamspace_data,
                     cartesian_points: ops.array | None = None,
                     ):
        pass


class SectorScanConverter(ScanConverter):
    scan: ScanConvetable

    # azimuths: ops.array | LinearCoordinate | IrregularSampledCoordinate
    # elevations: ops.array | LinearCoordinate | IrregularSampledCoordinate
    # depths: ops.array | LinearCoordinate | IrregularSampledCoordinate
    # apex: ops.array
    fill_value: float | None = float("nan")
    interpolator_type: Type[NDInterpolator] = LinearNDInterpolator
    sampling_type: IrregularSampledCoordinate | LinearCoordinate = IrregularSampledCoordinate
    shape: tuple | None = field(default=None, static=True)
    _azimuth_scale: float = field(default=None, static=True)
    _elevation_scale: float = field(default=None, static=True)
    _is_3d: bool = field(default=True, static=True)

    def __post_init__(self):
        if self._azimuth_scale is None:
            self._azimuth_scale = self.get_azimuth_scale()
        if self._is_3d and self._elevation_scale is None:      
            self._elevation_scale = self.get_elevation_scale()

    def scan_convert(self, 
                beamspace_data, 
                cartesian_points: ops.array | None = None,
                ):
        
        azimuths = self.azimuths
        if isinstance(self.azimuths, ops.array):
            azimuths = IrregularSampledCoordinate(x_data=self.azimuths, dim="azimuths")

        elevations = self.elevations
        if isinstance(self.elevations, ops.array):
            elevations = IrregularSampledCoordinate(x_data=self.elevations, dim="elevations")

        depths = self.depths
        if isinstance(self.depths, ops.array):
            depths = IrregularSampledCoordinate(x_data=self.depths, dim="depths")

        if cartesian_points is None:
            if self.shape is None:
                shape = self.calculate_cartesian_shape(n_zs = beamspace_data.dim_sizes["depths"])
            else:
                shape = self.shape
            cartesian_points = self.get_cartesian_points(shape)

        polar_points = self.scan.from_cartesian_to_local_coordinates(cartesian_points)

        # beamspace_coordinates = self.scan.coordinates
        # beamspace_coordinates = {
        #     "azimuths": azimuths,
        #     "depths": depths,
        # }

        interpolated_coordinates = {
            "azimuths": polar_points["az_el_depth", 0],
            "depths": polar_points["az_el_depth", 2]
            }
        if self._is_3d:
            # beamspace_coordinates["elevations"] = elevations
            interpolated_coordinates["elevations"] = polar_points["az_el_depth", 1]

        return scan_convert_impl(
                    beamspace_data,
                    self.scan.coordinates,
                    interpolated_coordinates,
                    fill_value=self.fill_value,
                    interpolator_type=self.interpolator_type,
                    )

    def get_azimuth_scale(self) -> float:
        min_x, max_x, min_y, max_y, min_z, max_z = self.calculate_cartesian_bounds()
        azimuth_scale = ops.abs((max_x - min_x) / (max_z - min_z))
        return float(azimuth_scale)

    def get_elevation_scale(self) -> float:
        min_x, max_x, min_y, max_y, min_z, max_z = self.calculate_cartesian_bounds()
        elevation_scale = ops.abs((max_y - min_y) / (max_z - min_z))
        return float(elevation_scale)

    def calculate_cartesian_bounds(self):
        """ 
        Calculate the bounds of the cartesian coordinates. 
        Note: Should be updated to calculate the bounds analytically
        """

        polar_point = ops.stack([self.azimuths, self.elevations, self.depths], axis="az_el_depth")

        polar_point["az_el_depth", 2] -= self.apex["xyz", 2] / (ops.cos(polar_point["az_el_depth", 0])*ops.cos(-polar_point["az_el_depth", 1]))

        # cartesian_points = as_cartesian(polar_point)
        cartesian_points = polar_to_cartesian(polar_point)
        cartesian_points["xyz", 2] += self.apex["xyz", 2]

        min_x = cartesian_points["xyz", 0].min()
        max_x = cartesian_points["xyz", 0].max()
        min_y = cartesian_points["xyz", 1].min() if self._is_3d else 0
        max_y = cartesian_points["xyz", 1].max() if self._is_3d else 0
        min_z = cartesian_points["xyz", 2].min()
        max_z = cartesian_points["xyz", 2].max()
        return (min_x, max_x, min_y, max_y, min_z, max_z)

    def calculate_cartesian_shape(self, n_zs: int) -> tuple[int, int]:
        """ 
        The function calculates the cartesian shape based on the beamspace data. 
        The shape is calculated to keep the dx, dy, dz equal for zs, ys, and zs.
        """

        n_xs = int(n_zs * self._azimuth_scale)
        n_ys = int(n_zs * self._elevation_scale) if self._is_3d else 0
        
        return (n_xs, n_ys, n_zs)

    def get_cartesian_points(self, shape: ops.array) -> ops.array:
        min_x, max_x, min_y, max_y, min_z, max_z = self.calculate_cartesian_bounds()

        x_axis = ops.linspace(min_x, max_x, shape[0], dim="xs")
        y_axis = ops.linspace(min_y, max_y, shape[1], dim="ys") if self._is_3d else 0
        z_axis = ops.linspace(min_z, max_z, shape[2], dim="zs")

        return ops.stack([x_axis, y_axis, z_axis], axis="xyz")

class SectorScanConverter2D(SectorScanConverter):
    def __init__(self,
            azimuths: ops.array | LinearCoordinate | IrregularSampledCoordinate,
            depths: ops.array | LinearCoordinate | IrregularSampledCoordinate,
            apex: ops.array = ops.array([0,0,0,], dims=["xyz"]),
            fill_value: float | None = float("nan"),
            interpolator_type: Type[NDInterpolator] = LinearNDInterpolator,
            shape: tuple | None=None,
            ):
        shape = shape if shape is None else (shape[0], 0, shape[1])

        scan = SectorScanGeometry(azimuths=azimuths, elevations=0, depths=depths)

        super().__init__(azimuths, 0, depths, apex, fill_value, interpolator_type, shape=shape ,_is_3d=False)

    def from_scan(scan: SectorScanGeometry, 
                  fill_value: float | None = float("nan"), 
                  interpolator_type: Type[NDInterpolator] = LinearNDInterpolator,
                  shape: tuple | None=None,
                  ):
        return SectorScanConverter2D(
            azimuths=scan.azimuths,
            depths=scan.depths,
            apex=scan.apex,
            fill_value=fill_value,
            interpolator_type=interpolator_type,
            shape=shape,
        )

    def from_dicom(scan: SectorScanGeometry | SectorScanGeometryRTB,
                  fill_value: float | None = float("nan"), 
                  interpolator_type: Type[NDInterpolator] = LinearNDInterpolator,
                  shape: tuple | None=None,
                  ):
        return SectorScanConverter2D(
            azimuths=scan.azimuths,
            depths=scan.depths,
            apex=scan.apex,
            fill_value=fill_value,
            interpolator_type=interpolator_type,
            shape=shape,
        )        

class SectorScanConverter3D(SectorScanConverter):
    def __init__(self, 
            azimuths: ops.array | LinearCoordinate | IrregularSampledCoordinate,
            elevations: ops.array | LinearCoordinate | IrregularSampledCoordinate,
            depths: ops.array | LinearCoordinate | IrregularSampledCoordinate,
            apex: ops.array = ops.array([0,0,0,], dims=["xyz"]),
            fill_value: float | None = float("nan"),
            interpolator_type: Type[NDInterpolator] = LinearNDInterpolator,
            shape: tuple | None = field(default=None, static=True),
            ):
        super().__init__(azimuths, elevations, depths, apex, fill_value, interpolator_type, shape=shape,_is_3d=True)

    def from_scan(scan: SectorScanGeometry, 
                  fill_value: float | None = float("nan"), 
                  interpolator_type: Type[NDInterpolator] = LinearNDInterpolator,
                  shape: tuple | None=None,
                  ):

        return SectorScanConverter3D(
            azimuths=scan.azimuths,
            elevations =scan.elevations,
            depths=scan.depths,
            apex=scan.apex,
            fill_value=fill_value,
            interpolator_type=interpolator_type,
            shape=shape,
        )        

