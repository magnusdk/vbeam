from dataclasses import dataclass

from spekk import Spec

from vbeam.beamformers.transformations.base import (
    Specced,
    Transformation,
    TransformedFunctionError,
    compose,
)


@dataclass
class T(Transformation):
    "A mock Transformation"
    f: callable
    add_dims: tuple = ()
    remove_dims: tuple = ()

    def transform(
        self, to_be_transformed: callable, input_spec: Spec, output_spec: Spec
    ) -> callable:
        def wrapped(*args, **kwargs):
            res = to_be_transformed(*args, **kwargs)
            return self.f(res)

        return wrapped

    def transform_input_spec(self, spec: Spec) -> Spec:
        for dim in self.add_dims:
            spec = spec.add_dimension(dim)
        for dim in self.remove_dims:
            if not spec.has_dimension(dim):
                raise ValueError(f"Spec does not have dimension {dim}")
            spec = spec.remove_dimension(dim)
        return spec

    def transform_output_spec(self, spec: Spec) -> Spec:
        return spec

    def __repr__(self) -> str:
        return f"T({self.f.__name__}, add_dims={self.add_dims}, remove_dims={self.remove_dims})"


## The following are just unit tests
import pytest


def add1(x):
    return x + 1


# A transformed function can be run without a Spec, if it doesn't require one
# - It will act as just a function composition
def test_transformed_function_without_spec():
    tf = compose(
        add1,
        T(add1),
        T(add1),
    )
    assert tf(0) == 3


# A transformed function can be built with a Spec
# - If the Spec is wrong for the function, it will raise an exception
# - It keeps track of the changes to Spec as it is applied, and we can get the final output Spec
def test_transformed_function_with_spec():
    tf = compose(
        Specced(add1),  # By default, Specced does nothing to the passed spec
        T(add1, remove_dims=("b",)),
        T(add1, add_dims=("c",)),
    ).build(Spec(("a", "b")))

    assert tf(0) == 3
    assert tf.output_spec == Spec(("c", "a"))

    # A function can be rebuilt with a new Spec
    tf2 = tf.build(Spec(("w", "b")))
    assert tf2.output_spec == Spec(("c", "w"))


def test_transformed_function_with_incorrect_spec():
    tf = compose(
        add1,
        T(add1, remove_dims=("b",)),
        T(add1, add_dims=("c",)),
    )
    with pytest.raises(TransformedFunctionError):
        # Missing dimension "b"
        tf.build(Spec(("a",)))


# If an error occurs in a nestedly transformed function, we can get the traceback with the step that caused the error
# - This can happen when calling the function
# - Or when building the function (for example with incorrect spec)
def test_transformed_function_with_incorrect_spec_error_message():
    tf = compose(
        add1,
        T(add1, remove_dims=("b",)),
        T(add1, add_dims=("c",)),
    )
    try:
        tf.build(Spec(("a",)))
    except TransformedFunctionError as e:
        expected_msg = """Spec does not have dimension b
TransformedFunction(
  <compose(
    add1,
⚠   T(add1, add_dims=(), remove_dims=('b',)),
⚠     ↳ This step raised ValueError('Spec does not have dimension b')
    T(add1, add_dims=('c',), remove_dims=()),
  )>
)"""
        assert str(e) == expected_msg


def divide_by_zero(x):
    return x / 0


def test_exception_raising_transformed_function_error_message():
    tf = compose(
        divide_by_zero,
        T(add1, remove_dims=("b",)),
        T(add1, add_dims=("c",)),
    ).build(Spec(("a", "b")))
    try:
        tf(0)
    except TransformedFunctionError as e:
        expected_msg = """division by zero
TransformedFunction(
  <compose(
⚠   divide_by_zero,
⚠     ↳ This step raised ZeroDivisionError('division by zero')
    T(add1, add_dims=(), remove_dims=('b',)),
    T(add1, add_dims=('c',), remove_dims=()),
  )>
)"""
        assert str(e) == expected_msg

    # Let's try raising an error at a different step
    tf = compose(
        add1,
        T(divide_by_zero, remove_dims=("b",)),
        T(add1, add_dims=("c",)),
    ).build(Spec(("a", "b")))
    try:
        tf(0)
    except TransformedFunctionError as e:
        expected_msg = """division by zero
TransformedFunction(
  <compose(
    add1,
⚠   T(divide_by_zero, add_dims=(), remove_dims=('b',)),
⚠     ↳ This step raised ZeroDivisionError('division by zero')
    T(add1, add_dims=('c',), remove_dims=()),
  )>
)"""
        assert str(e) == expected_msg


# We can transform a Transformation, making it partially applied


def test_partially_applied_transformation():
    partial_tf = compose(T(add1), T(add1))
    tf = compose(add1, partial_tf, T(add1))
    assert tf(0) == 4

    partial_tf = compose(T(add1, remove_dims=("c", "a")), T(add1, add_dims=("c",)))
    tf = compose(Specced(add1), partial_tf, T(add1)).build(Spec(("a", "b")))
    assert tf(0) == 4
    assert tf.output_spec == Spec(("b",))


def test_partial_transformations_are_handled_as_separate_steps():
    partial_tf = compose(T(add1), T(add1), T(add1))
    tf = compose(add1, partial_tf, T(add1))
    expected_repr = """TransformedFunction(
  <compose(
    add1,
    T(add1, add_dims=(), remove_dims=()),
    T(add1, add_dims=(), remove_dims=()),
    T(add1, add_dims=(), remove_dims=()),
    T(add1, add_dims=(), remove_dims=()),
  )>
)"""
    assert repr(tf) == expected_repr


# We can transform a transformed function
# - It can be rebuilt with a new Spec
def test_transform_transformed_function():
    tf = compose(
        Specced(add1),
        T(add1, remove_dims=("b",)),
        T(add1, add_dims=("c",)),
    ).build(Spec(("a", "b")))

    # It will not inherit the Spec from the wrapped function
    tf2 = compose(tf, T(add1))
    assert tf2(0) == 4
    assert tf2.output_spec is None

    # But we can rebuild it with the same input Spec as the wrapped function
    tf2 = tf2.build(tf.input_spec)
    assert tf2.output_spec == tf.output_spec
