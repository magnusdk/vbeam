from abc import abstractmethod
from typing import Type
from spekk import Module, ops, field
from vbeam.core import NDInterpolator, Coordinate
from vbeam.scan import SectorScanGeometry, ScanConvetable
from vbeam.interpolation import (
    LinearNDInterpolator,
    LinearCoordinate,
    IrregularSampledCoordinate,
)


def scan_convert_impl(
    beamspace_data: ops.array,
    beamspace_coordinates: dict["str", Type[Coordinate]],
    interpolated_coordinates: dict["str", ops.array],
    fill_value: float | None,
    interpolator_type: Type[NDInterpolator],
):
    interpolator = interpolator_type(
        beamspace_coordinates, beamspace_data, fill_value=fill_value
    )
    interpolated_data = interpolator(interpolated_coordinates)
    return interpolated_data


class ScanConverter(Module):
    @abstractmethod
    def scan_convert(
        self,
        beamspace_data,
        cartesian_points: ops.array | None = None,
    ):
        pass


class SectorScanConverter(ScanConverter):
    scan: ScanConvetable
    fill_value: float | None
    interpolator_type: Type[NDInterpolator]
    sampling_types: dict[str, Type[Coordinate]]
    output_shape: tuple | None = field(default=None, static=True)
    _azimuth_scale: float = field(default=None, static=True)
    _elevation_scale: float = field(default=None, static=True)
    _is_3d: bool = field(default=True, static=True)

    def __post_init__(self):
        if self._azimuth_scale is None:
            self._azimuth_scale = self.get_azimuth_scale()
        if self._is_3d and self._elevation_scale is None:
            self._elevation_scale = self.get_elevation_scale()

    def scan_convert(
        self,
        beamspace_data,
        cartesian_points: ops.array | None = None,
    ):
        if cartesian_points is None:
            if self.output_shape is None:
                output_shape = self.calculate_cartesian_shape(
                    n_zs=beamspace_data.dim_sizes["depths"]
                )
            else:
                output_shape = self.output_shape
            cartesian_points = self.get_cartesian_points(output_shape)

        polar_points = self.scan.from_cartesian_to_local_coordinates(cartesian_points)

        interpolated_coordinates = {
            "azimuths": polar_points["az_el_depth", 0],
            "depths": polar_points["az_el_depth", 2],
            ** ({"elevations": polar_points["az_el_depth", 1]} if self._is_3d else {}),
        }

        beamspace_coordinates = {
            "azimuths": self.scan.azimuths,
            "elevations": self.scan.elevations,
            "depths": self.scan.depths,
        }

        for key, sampling_type in self.sampling_types.items():
            if sampling_type==IrregularSampledCoordinate:
                beamspace_coordinates[key] = IrregularSampledCoordinate(
                    x_data=beamspace_coordinates[key], dim=key
                )
            elif sampling_type==LinearCoordinate:
                beamspace_coordinates[key] = LinearCoordinate(
                    start=beamspace_coordinates[key][0], 
                    stop=beamspace_coordinates[key][-1], 
                    num=len(beamspace_coordinates[key])
                )
            else:
                raise ValueError(
                    f"Unsupported sampling type: {sampling_type} for coordinate {key}"
                )

        return scan_convert_impl(
            beamspace_data,
            beamspace_coordinates,
            interpolated_coordinates,
            fill_value=self.fill_value,
            interpolator_type=self.interpolator_type,
        )

    def get_azimuth_scale(self) -> float:
        min_x, max_x, min_y, max_y, min_z, max_z = self.scan.calculate_cartesian_bounds()
        azimuth_scale = ops.abs((max_x - min_x) / (max_z - min_z))
        return float(azimuth_scale)

    def get_elevation_scale(self) -> float:
        min_x, max_x, min_y, max_y, min_z, max_z = self.scan.calculate_cartesian_bounds()
        elevation_scale = ops.abs((max_y - min_y) / (max_z - min_z))
        return float(elevation_scale)

    def calculate_cartesian_shape(self, n_zs: int) -> tuple[int, int]:
        """
        The function calculates the cartesian shape based on the beamspace data.
        The shape is calculated to keep the dx, dy, dz equal for zs, ys, and zs.
        """

        n_xs = int(n_zs * self._azimuth_scale)
        n_ys = int(n_zs * self._elevation_scale) if self._is_3d else 0

        return (n_xs, n_ys, n_zs)

    def get_cartesian_points(self, output_shape: ops.array) -> ops.array:
        min_x, max_x, min_y, max_y, min_z, max_z = self.scan.calculate_cartesian_bounds()

        x_axis = ops.linspace(min_x, max_x, output_shape[0], dim="xs")
        y_axis = ops.linspace(min_y, max_y, output_shape[1], dim="ys") if self._is_3d else 0
        z_axis = ops.linspace(min_z, max_z, output_shape[2], dim="zs")

        return ops.stack([x_axis, y_axis, z_axis], axis="xyz")


class SectorScanConverter2D(SectorScanConverter):
    def __init__(
        self,
        azimuths: ops.array,
        depths: ops.array,
        fill_value: float | None = float("nan"),
        interpolator_type: Type[NDInterpolator] = LinearNDInterpolator,
        sampling_types: dict[str, Type[Coordinate]] = {
            "azimuths": IrregularSampledCoordinate,
            "depths": IrregularSampledCoordinate,
        },
        output_shape: tuple | None = field(default=None, static=True), 
    ):
        output_shape = output_shape if output_shape is None else (output_shape[0], 0, output_shape[1])
        scan = SectorScanGeometry(azimuths=azimuths, elevations=0, depths=depths)
        super().__init__(
            scan,
            fill_value,
            interpolator_type,
            sampling_types,
            output_shape=output_shape,
            _is_3d=False,
        )

    def from_scan(
        scan: SectorScanGeometry,
        fill_value: float | None = float("nan"),
        interpolator_type: Type[NDInterpolator] = LinearNDInterpolator,
        sampling_types: dict[str, Type[Coordinate]] = {
            "azimuths": IrregularSampledCoordinate,
            "depths": IrregularSampledCoordinate,
        },
        output_shape: tuple | None = None, 
    ):
        return SectorScanConverter2D(
            azimuths=scan.azimuths,
            depths=scan.depths,
            fill_value=fill_value,
            interpolator_type=interpolator_type,
            sampling_types=sampling_types,
            output_shape=output_shape,
        )


class SectorScanConverter3D(SectorScanConverter):
    def __init__(
        self,
        azimuths: ops.array,
        elevations: ops.array,
        depths: ops.array,
        fill_value: float | None = float("nan"),
        interpolator_type: Type[NDInterpolator] = LinearNDInterpolator,
        sampling_types: dict[str, Type[Coordinate]] = {
            "azimuths": IrregularSampledCoordinate,
            "elevations": IrregularSampledCoordinate,
            "depths": IrregularSampledCoordinate,
        },
        output_shape: tuple | None = field(default=None, static=True), 
    ):
        scan = SectorScanGeometry(
            azimuths=azimuths, elevations=elevations, depths=depths
        )
        super().__init__(
            scan, fill_value, interpolator_type, sampling_types, output_shape, _is_3d=True
        )

    def from_scan(
        scan: SectorScanGeometry,
        fill_value: float | None = float("nan"),
        interpolator_type: Type[NDInterpolator] = LinearNDInterpolator,
        sampling_types: dict[str, Type[Coordinate]] = {
            "azimuths": IrregularSampledCoordinate,
            "elevations": IrregularSampledCoordinate,
            "depths": IrregularSampledCoordinate,
        },
        output_shape: tuple | None = None, 
    ):
        return SectorScanConverter3D(
            azimuths=scan.azimuths,
            elevations=scan.elevations,
            depths=scan.depths,
            fill_value=fill_value,
            interpolator_type=interpolator_type,
            sampling_types=sampling_types,
            output_shape=output_shape,
        )
