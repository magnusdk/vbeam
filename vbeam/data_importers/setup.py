import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Sequence, Union

from spekk import Spec, trees
from spekk.util.slicing import IndicesT, slice_data, slice_spec

from vbeam.core import SignalForPointData
from vbeam.fastmath import numpy as np
from vbeam.scan import PointOptimizer, Scan
from vbeam.util.transformations import *


@dataclass
class SignalForPointSetup(SignalForPointData):
    spec: Spec
    scan: Optional[Scan] = None
    points_optimizer: Optional[PointOptimizer] = None

    @property
    def slice(self):
        class Slicer:
            def __getitem__(
                _, slices: Sequence[Union[str, IndicesT]]
            ) -> "SignalForPointSetup":
                sliced_data = slice_data(self.data, self.spec, slices)
                sliced_spec = slice_spec(self.spec, slices)
                return SignalForPointSetup(**sliced_data, spec=sliced_spec)

        return Slicer()

    def __setattr__(self, name: str, value) -> None:
        # Handle cases where scan and point_pos are set at the same time
        # Note: scan always takes precedence
        if name == "point_pos" and self.scan is not None:
            raise AttributeError(
                "You may not set point_pos when a scan has been defined. Try updating \
the scan instead."
            )
        if name == "scan":
            if self.point_pos is not None:
                warnings.warn("point_pos will be overwritten by the scan.")

        return super().__setattr__(name, value)

    def __getattribute__(self, name: str):
        scan: Scan = object.__getattribute__(self, "scan")
        points_optimizer: Optional[PointOptimizer] = object.__getattribute__(
            self, "points_optimizer"
        )
        if name == "point_pos" and scan is not None:
            point_pos = object.__getattribute__(self, "point_pos")
            if point_pos is not None:
                warnings.warn("Both point_pos and scan are set. Scan will be used.")
            value = scan.get_points()

            if points_optimizer is not None:
                value = points_optimizer.reshape(value, scan)
        elif name == "spec":
            spec: Spec = object.__getattribute__(self, "spec")
            if points_optimizer is not None:
                spec = spec.replace(
                    {"point_pos": points_optimizer.shape_info.after_reshape}
                )
            point_pos = object.__getattribute__(self, "point_pos")
            if scan is None and point_pos is None:
                spec = spec.remove_subtree(["point_pos"])
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
        sum_fn = np.sum if sum_fn == "sum" else sum_fn
        all_dimensions = (
            self.spec["sender"].dimensions
            | self.spec["point_pos"].dimensions
            | self.spec["receiver"].dimensions
            | self.spec["wave_data"].dimensions
        )
        calculate_apodization = compose(
            lambda apodization, *args, **kwargs: apodization(*args, **kwargs),
            *[ForAll(dim) for dim in all_dimensions],
            Apply(sum_fn, [Axis(dim) for dim in all_dimensions - set(dimensions)]),
            # Put the dimensions in the order defined by keep
            Apply(np.transpose, [Axis(dim, keep=True) for dim in dimensions]),
        ).build(self.spec.replace({"point_position": self.spec.get(["point_pos"])}))
        return calculate_apodization(
            apodization=self.apodization,
            sender=self.sender,
            point_position=self.point_pos,
            receiver=self.receiver,
            wave_data=self.wave_data,
        )
