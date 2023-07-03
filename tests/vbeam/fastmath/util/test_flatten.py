from dataclasses import dataclass, field
from typing import Any, Callable, List, Sequence, Tuple, Union

import numpy as np
from spekk import Spec, trees
from vbeam_test_helpers.assertions import assert_equal_treedefs

from vbeam.core import ElementGeometry
from vbeam.fastmath import Backend
from vbeam.fastmath.util.flatten import flatten


def test_flattening_dict():
    foo_x = np.ones((2, 3)) * 4
    foo_y = np.ones((2,)) * 2
    bar = np.ones((3,)) * 3
    baz_z = np.ones((4,))
    obj = {
        "foo": {
            "x": foo_x,
            "y": [foo_y, foo_y],
        },
        "bar": bar,
        "baz": {"z": baz_z},
    }
    spec = Spec(
        {
            "foo": {
                "x": ["a", "b"],
                "y": [["a"], ["a"]],
            }
        }
    )

    flattened, in_axes, unflatten = flatten(obj, spec, "b")
    np.testing.assert_equal(flattened, [foo_x, [foo_y, foo_y], bar, {"z": baz_z}])
    assert in_axes == [1, None, None, None]
    np.testing.assert_equal(obj, unflatten(flattened))

    flattened, in_axes, unflatten = flatten(obj, spec, "a")
    np.testing.assert_equal(flattened, [foo_x, foo_y, foo_y, bar, {"z": baz_z}])
    assert in_axes == [0, 0, 0, None, None]
    np.testing.assert_equal(obj, unflatten(flattened))


def test_flattening_element_geometry():
    obj = {
        "element": ElementGeometry(
            position=np.ones((2, 3)),
            theta=np.ones((3,)) * 2,
            sub_elements=ElementGeometry(
                position=np.ones((2, 4)) * 3,
                theta=np.ones((4, 3, 2)) * 4,
            ),
        )
    }
    spec = Spec(
        {
            "element": {
                "position": ["a", "b"],
                "theta": ["b"],
                "sub_elements": {
                    "position": ["a", "c"],
                    "theta": ["c", "b", "a"],
                },
            }
        }
    )

    flattened, in_axes, unflatten = flatten(obj, spec, "a")
    assert in_axes == [0, None, 0, 2]
    assert_equal_treedefs(obj, unflatten(flattened))

    flattened, in_axes, unflatten = flatten(obj, spec, "b")
    assert in_axes == [1, 0, None, 1]
    assert_equal_treedefs(obj, unflatten(flattened))

    flattened, in_axes, unflatten = flatten(obj, spec, "c")
    assert in_axes == [None, None, 1, 0]
    assert_equal_treedefs(obj, unflatten(flattened))


def test_flattening_single_item():
    obj = np.array([1, 2, 3])
    spec = Spec(["a"])
    flattened, in_axes, unflatten = flatten(obj, spec, "a")
    print(flattened, in_axes, unflatten(flattened))
    np.testing.assert_equal(flattened, [obj])
    assert in_axes == [0]
    np.testing.assert_equal(obj, unflatten(flattened))
