from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple, Union

from spekk import Spec, trees


@dataclass
class _DummyContainer:
    """A temporary container that keeps the structure of the original object and can be
    used to get the original object back given a flattened object.

    All values of data are either another _DummyContainer or an integer representing
    the index in the flattened_obj list of the original value. When calling unflatten,
    the index is looked up in the flattened_obj to get back the original value.
    """

    tree: trees.TreeDef
    data: dict

    def unflatten(self, flattened_obj: list):
        return self.tree.create(
            self.data.keys(),
            [
                v.unflatten(flattened_obj)
                if isinstance(v, _DummyContainer)
                else flattened_obj[v]
                for v in self.data.values()
            ],
        )


@dataclass
class _State:
    """Keep track of the list of parts of the flattened object and their corresponding
    axes (calculated given the spec and a dimension).

    Fields:
    - flattened: the list of all parts of the flattened object.
    - in_axes: the axis of a dimension for each part of the flattened object.
    - i: the current index in flattened. It is incremented every time a part is added
        to flattened (and in_axes).
    """

    flattened: list = field(default_factory=list)
    in_axes: list = field(default_factory=list)
    i: int = 0

    def append(self, value: Any, axis: Union[int, None]) -> int:
        """Add value to flattened and the axis of a dimension for the value to in_axes.

        Return the index of the newly added value and axis in the flattened list.
        """
        self.flattened.append(value)
        self.in_axes.append(axis)
        flattened_index = self.i  # Return the value of i before incrementing
        self.i += 1
        return flattened_index


def flatten(
    obj: trees.Tree, spec: Spec, dimension: str
) -> Tuple[List[Any], List[Union[None, int]], Callable[[List[Any]], trees.Tree],]:
    """Flatten/decompose the obj into a flattened list such that parts with the given
    dimension are separate items from those without the dimension.

    This allows us to define objects with fields that may consist of different
    dimensions, but still be able to vmap over them. vmap requires all axes vmapped
    over to have the same size so we need to separate fields with different dimensions
    to different arguments.

    Returns a tuple of:
    - The flattened/decomposed object as a list.
    - The axis of each element in the flattened list that corresponds to the given
        dimension.
    - A function that takes a flattened/decomposed object (same structure as the one
        returned from this function) and returns an object with the same structure as
        the original object.
    """
    state = _State()  # state is updated inside the _flatten function

    def _flatten(obj: trees.Tree, spec: Spec):
        "Recursively flatten obj, mutating the state along the way."

        # Base case 1: the object is not a tree-like structure and can not be
        #   recursively flattened.
        base_case1 = not trees.has_treedef(obj)
        # Base case 2: the spec is a leaf, i.e.: not a tree-like structure, and can not
        #   be used to recursively flatten obj.
        base_case2 = spec.is_leaf(spec)
        if base_case1 or base_case2:
            # Just add the object as-is to the flattened arguments
            flattened_index = state.append(obj, spec.index_for(dimension))
            return flattened_index

        # Else, it is a nested tree-like structure
        tree = trees.treedef(obj)

        # Build up a dummy-object that references the indices in the flattened array
        dummy_data = {}
        for key in tree.keys():
            value = tree.get(key)
            if value is not None:  # None values can be ignored
                sub_spec = Spec(spec.get([key])) if spec.has_subtree([key]) else None
                if sub_spec is not None and sub_spec.has_dimension(dimension):
                    dummy_data[key] = _flatten(value, sub_spec)
                else:
                    flattened_index = state.append(value, None)
                    dummy_data[key] = flattened_index
        return _DummyContainer(tree, dummy_data)

    dummy = _flatten(obj, spec)
    if isinstance(dummy, _DummyContainer):
        unflatten = dummy.unflatten
    else:
        # Special case where the object was not a nested structure. If it was just a
        # single item, then we need to just return that item when unflattening.
        unflatten = lambda x: x[0]

    _validate(state.flattened, state.in_axes)
    return state.flattened, state.in_axes, unflatten


def _validate(args: list, in_axes: List[int]):
    return
    # TODO: validate that args have the same size for the given in_axes
    # sizes = [arg.shape[axis] for arg, axis in zip(args, in_axes)]
