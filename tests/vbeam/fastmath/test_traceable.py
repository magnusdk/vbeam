from abc import ABC, abstractmethod

from vbeam.fastmath import Backend, backend_manager
from vbeam.fastmath import numpy as np
from vbeam.fastmath.traceable import traceable_dataclass


class MyABC(ABC):
    @abstractmethod
    def do_thing(self):
        ...


@traceable_dataclass(data_fields=["a"])
class Item(MyABC):
    a: np.ndarray = 1.0

    def do_thing(self):
        return 2.0


@traceable_dataclass(data_fields=["item"])
class Container(MyABC):
    item: Item

    def do_thing(self):
        return 3.0


def test_tensorflow_traceable_dataclass_with_gradient():
    import tensorflow as tf

    backend_manager.active_backend = "tensorflow"

    @tf.function(jit_compile=True)
    def func(container: Container):
        return container.item.a + container.item.do_thing() + container.do_thing()

    item = Item(a=10.5)
    container = Container(item)
    with tf.GradientTape() as tape:
        tape.watch(container.item.a)
        res = func(container)

    assert res == 15.5
    assert tape.gradient(res, container.item.a) == 1.0


def test_jax_traceable_dataclass_with_gradient():
    import jax

    backend_manager.active_backend = "jax"

    @jax.jit
    @jax.value_and_grad
    def func(container: Container):
        return container.item.a + container.item.do_thing() + container.do_thing()

    item = Item(a=10.5)
    container = Container(item)
    res, container_grad = func(container)

    assert res == 15.5
    assert container_grad.item.a == 1.0


def test_numpy_traceable_dataclass():
    backend_manager.active_backend = "numpy"

    def func(container: Container):
        return container.item.a + container.item.do_thing() + container.do_thing()

    item = Item(a=10.5)
    container = Container(item)
    res = func(container)

    assert res == 15.5


def test_dataclass_fields_float_0d_tensor(np: Backend):
    """There have been some (Tensorflow-specific) problems with creating dataclasses
    with fields of scalar types and constructing them using 0-dimensional (aka scalar)
    values.
    Tensorflow has complained that it "expected float, got <tf.Tensor: shape=(), ...>."
    """

    @traceable_dataclass(data_fields=["a"])
    class Foo:
        a: float

    some_tensor = np.array([0.2, 0.1, 0.3])
    obj = Foo(a=some_tensor.min())
    assert obj.a == 0.1
