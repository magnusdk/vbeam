"""A datastructure for representing a transducer element (or the full array), including 
the position, orientation, etc.
"""

from typing import Callable, Optional

from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass

identity_fn = lambda x: x  # Just return value as-is


@traceable_dataclass(("position", "theta", "phi", "sub_elements", "parent_element"))
class ElementGeometry:
    """A vectorizable container of element geometry.

    ElementGeometry is vectorizable, meaning that it works with vmap. This way we can
    represent the element geometry for all probe elements in a single object.

    If an ElementGeometry object contains multiple elements then each field will have an
    additional dimension. For example, position may have the shape (64, 3) if there are
    64 elements (each with x, y, and z coordinates)."""

    position: np.ndarray
    theta: Optional[float] = None
    phi: Optional[float] = None
    sub_elements: Optional["ElementGeometry"] = None
    parent_element: Optional["ElementGeometry"] = None

    def __getitem__(self, *args) -> "ElementGeometry":
        """Index the element geometry.

        Note that a ElementGeometry instance may be a container of multiple vectorizable
        ElementGeometry "objects". See ElementGeometry's class docstring.

        >>> element_geometry = ElementGeometry(
        ...     np.array([[0,0,0], [1,1,1]]), np.array([0,1]), np.array([0,1])
        ... )
        >>> element_geometry[1]
        ElementGeometry(position=array([1, 1, 1]), theta=1, phi=1, sub_elements=None)
        """
        _maybe_getitem = (
            lambda attr: attr.__getitem__(*args) if attr is not None else None
        )
        return ElementGeometry(
            _maybe_getitem(self.position),
            _maybe_getitem(self.theta),
            _maybe_getitem(self.phi),
            _maybe_getitem(self.sub_elements),
            _maybe_getitem(self.parent_element),
        )

    @property
    def shape(self) -> tuple:
        return self.position.shape[:-1]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def with_updates_to(
        self,
        *,
        position: Callable[[np.ndarray], np.ndarray] = identity_fn,
        theta: Callable[[float], float] = identity_fn,
        phi: Callable[[float], float] = identity_fn,
        sub_elements: Callable[["ElementGeometry"], "ElementGeometry"] = identity_fn,
        parent_element: Callable[["ElementGeometry"], "ElementGeometry"] = identity_fn,
    ) -> "ElementGeometry":
        """Return a copy with updated values for the given fields.

        If the given value for a field is a function the updated field will be that
        function applied to the current field. Example:
        >>> element_geometry = ElementGeometry(np.array([0, 0, 0]), 1, 2)
        >>> element_geometry.with_updates_to(position=lambda x: x+1)
        ElementGeometry(position=array([1, 1, 1]), theta=1, phi=2, sub_elements=None)

        If the given value for a field is not a function then the field will simply be
        set to that value. Example:
        >>> element_geometry.with_updates_to(theta=5, phi=6)
        ElementGeometry(position=array([0, 0, 0]), theta=5, phi=6, sub_elements=None)
        """
        return ElementGeometry(
            position=position(self.position) if callable(position) else position,
            theta=theta(self.theta) if callable(theta) else theta,
            phi=phi(self.phi) if callable(phi) else phi,
            sub_elements=sub_elements(self.sub_elements)
            if callable(sub_elements)
            else sub_elements,
            parent_element=parent_element(self.parent_element)
            if callable(parent_element)
            else parent_element,
        )

    def copy(self) -> "ElementGeometry":
        return self.with_updates_to()
