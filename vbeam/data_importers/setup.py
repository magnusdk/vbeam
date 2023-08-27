import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Sequence, Union

from spekk import Spec, trees
from spekk.util.slicing import IndicesT, slice_data, slice_spec

from vbeam.apodization.util import get_apodization_values
from vbeam.core import SignalForPointData
from vbeam.fastmath import numpy as np
from vbeam.scan import Scan
from vbeam.scan.advanced import ExtraDimsScanMixin
from vbeam.util.transformations import *


@dataclass
class SignalForPointSetup(SignalForPointData):
    spec: Spec
    scan: Optional[Scan] = None

    @property
    def slice(self):
        class Slicer:
            def __getitem__(
                _, slices: Sequence[Union[str, IndicesT]]
            ) -> "SignalForPointSetup":
                sliced_data = slice_data(self.data, self.spec, slices)
                sliced_spec = slice_spec(self.spec, slices)
                return SignalForPointSetup(
                    **sliced_data, scan=self.scan, spec=sliced_spec
                )

        return Slicer()

    def __setattr__(self, name: str, value) -> None:
        # Handle cases where scan and point_position are set at the same time
        # Note: scan always takes precedence
        if name == "point_position" and self.scan is not None:
            raise AttributeError(
                "You may not set point_position when a scan has been defined. Try \
updating the scan instead."
            )
        if name == "scan":
            if self.point_position is not None:
                warnings.warn("point_position will be overwritten by the scan.")

        return super().__setattr__(name, value)

    def __getattribute__(self, name: str):
        scan: Scan = object.__getattribute__(self, "scan")
        if name == "point_position" and scan is not None:
            point_position = object.__getattribute__(self, "point_position")
            if point_position is not None:
                warnings.warn(
                    "Both point_position and scan are set. Scan will be used."
                )
            value = scan.get_points()
        elif name == "spec":
            spec: Spec = object.__getattribute__(self, "spec")
            point_position = object.__getattribute__(self, "point_position")
            if scan is None and point_position is None:
                spec = spec.remove_subtree(["point_position"])
            elif isinstance(scan, ExtraDimsScanMixin):
                spec = spec.at["point_position"].set(scan.flattened_points_dimensions)
            value = spec
        else:
            value = object.__getattribute__(self, name)
        return value

    def size(
        self, dimension: Optional[Union[str, trees.Tree]] = None
    ) -> Union[int, Dict[str, int]]:
        """Return the size of the various dimensions of the data.

        ``dimension`` may be None, a string, or a tree of strings. If it is None, then
        a dict of all the dimensions and their sizes is returned. If it is a string,
        then the size of that dimension is returned. If it is a tree of strings, then
        the leaves of the tree is updated to be the size of those dimensions (assuming
        all leaves are dimensions that exists in the spec).

        See Spec.size for details."""
        data = self.data
        if dimension is None:
            # Returns a dict with all dimensions in the spec as keys and sizes as values
            return self.spec.size(data)
        # dimension may be a tree of dimensions. Update the leaves of the tree to get
        # the size of each dimension.
        return trees.update_leaves(
            dimension,
            lambda x: isinstance(x, str),
            lambda dimension: self.spec.size(data, dimension),
        )

    @property
    def data(self) -> dict:
        kernel_data_fields = SignalForPointData.__dataclass_fields__
        data = {k: getattr(self, k) for k in kernel_data_fields}
        return data

    def copy(self) -> "SignalForPointSetup":
        return SignalForPointSetup(**self.data, spec=self.spec, scan=self.scan)

    def get_apodization_values(
        self,
        dimensions: Sequence[str],
        sum_fn: Union[
            Literal["sum"], Callable[[np.ndarray, Sequence[int]], np.ndarray]
        ] = "sum",
    ):
        """Return the apodization values for the dimensions. All other relevant
        dimensions are (by default) summed over.

        If you want the apodization values for each transmit and point, you'd call
        setup.get_apodization_values(["transmits", "points"]). Likewise, if you only
        want the points, you'd call setup.get_apodization_values(["points"]). All other
        dimensions are summed over.

        By default, all dimensions not specified in the list are summed over. You can
        override this by passing in a sum_fn."""
        return get_apodization_values(
            self.apodization,
            self.sender,
            self.point_position,
            self.receiver,
            self.wave_data,
            self.spec,
            dimensions,
            sum_fn,
        )
