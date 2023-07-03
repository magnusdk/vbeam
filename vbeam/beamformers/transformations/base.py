"""Classes and functions for building up more complex functions by composing functions.

This module tries to solve two things:
- Having tools for transforming functions in a readable way
- Keeping track of the dimensions of the data after each function transformation

compose is a function that applies a list of functions to a value in order. Say we have 
3 functions, f, g, and h, we can apply them in order like this: compose(x, f, g, h). 
This is equivalent to h(g(f(x))), but is (often) more readable.

Let's look at another example:
>>> import numpy as np
>>> f = lambda a, b, c: np.sum(a) + b + np.sum(c)
>>> transformed_f = compose(
...     f,
...     Apply(np.sum),
...     ForAll("bees"),
... )

In this case, the value f is a function which we transform by composing it with two
Transformation objects, Apply(np.sum) and ForAll("bees"). Apply(np.sum) transforms the
function such that we apply np.sum to the result. ForAll("bees") further transforms the
function such that we apply it to each element in the "bees" dimension individually.

Again, the above code is equivalent to ForAll("bees")(Apply(np.sum)(f)).

What is the "bees" dimension? Good question! transformed_f certainly doesn't know (yet).
We use Spec objects to define the dimensions of the input data. Let's create some 
example data along with a spec for our example:
>>> a, b, c = np.ones((2, 3)), np.ones((3,)), np.ones((4,))
>>> spec = Spec(a=("foo", "bees"), b=("bees",), c=("bar",))

Notice that the spec mirrors the dimensions of the data. Also notice that the "bees"
dimension is shared between the a and b arguments. This tells our function that when 
working with the "bees" dimension, it should use axis 1 of a and axis 0 of b.

We must give this information to the transformed_f function so that it knows how to
handle our input data. We do this by calling the build method with our spec:
>>> transformed_f = transformed_f.build(spec)

Summarized, we can build up more complex functions by composing them with Transformation
objects and building the TransformedFunction with a given Spec:
>>> my_fn = compose(
...     lambda a, b, c: np.sum(a) + b + np.sum(c),
...     Apply(np.sum),
...     ForAll("bees"),
...     Apply(np.prod),  # Multiply the elements, AFTER looping over "bees"
... ).build(spec)
>>> my_fn(a, b, c)  # Should be 7 * 7 * 7 = 343
343.0
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Union

from spekk import Spec
from spekk.trees import has_treedef, leaves

from vbeam.beamformers.transformations.axis import Axis, concretize_axes
from vbeam.fastmath import numpy as np
from vbeam.fastmath.util import specced_vmap


def compose(x, *wrapping_functions):
    """Apply each f in fs to x.

    Let's say we have some functions:
    >>> f = lambda x: x+1
    >>> g = lambda x: x*2
    >>> h = lambda x: x**2

    We can use compose to apply each function in order:
    >>> compose(1, f, g, h)
    16

    This would be the same as calling:
    >>> h(g(f(1)))
    16

    In situations with a lot of nested function calls, compose may be more readable.
    Also notice that when using compose, functions are evaluated in the order that they
    are passed in (left-to-right), while with the nested function calls, the functions
    are evaluated in the reverse order (right-to-left).

    Compose can also be used to build up a function from smaller function
    transformations:

    >>> wrap_double = lambda f: (lambda x: 2*f(x))
    >>> f = compose(
    ...   lambda x: x+1,
    ...   wrap_double,
    ...   wrap_double,
    ... )
    >>> f(1)
    8
    """
    for wrap in wrapping_functions:
        x = wrap(x)
    return x


def identity(x):
    "Return the input unchanged."
    return x


def get_fn_name(f) -> str:
    if hasattr(f, "__qualname__"):
        return f.__qualname__
    if hasattr(f, "__name__"):
        return f.__name__
    return repr(f)


@dataclass
class Specced:
    """A wrapper around f that has information about the dimensions (Spec) of its inputs
    and outputs."""

    f: callable
    transform_input_spec: Callable[[Spec], Spec] = identity
    transform_output_spec: Callable[[Spec], Spec] = identity

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __hash__(self):
        return hash((self.f, self.transform_input_spec, self.transform_output_spec))


def _transformed_function_repr_fn(
    tf: "TransformedFunction",
    step_repr_fn: Optional[Callable[["TransformedFunction", str], str]] = None,
):
    "Format a TransformedFunction in a nice way."
    s = f"TransformedFunction(\n  <compose(\n"
    for step in tf.traverse(depth_first=True):
        step_repr = (
            repr(step.transformation)
            if isinstance(step, TransformedFunction)
            else get_fn_name(step)
        )
        if step_repr_fn is not None:
            step_repr = step_repr_fn(step, step_repr)
        else:
            step_repr = f"    {step_repr},\n"
        s += step_repr
    s += "  )>\n)"
    return s


@dataclass
class TransformedFunctionError(Exception):
    original_exception: Exception
    transformed_function: "TransformedFunction"
    error_step: Union[callable, "TransformedFunction"]

    def __post_init__(self):
        if isinstance(self.transformed_function, TransformedFunction):
            s = _transformed_function_repr_fn(
                self.transformed_function,
                lambda step, default_repr: f"⚠   {default_repr},\n"
                + f"⚠     ↳ This step raised {repr(self.original_exception)}\n"
                if step is self.error_step
                else f"    {default_repr},\n",
            )
            s = f"{str(self.original_exception)}\n" + s
        else:
            s = str(self.original_exception)
        super().__init__(s)


@dataclass
class _WrappedWithErrorHandling:
    f: Union[callable, Specced]

    def __call__(self, *args, **kwargs):
        try:
            return self.f(*args, **kwargs)
        except Exception as e:
            raise TransformedFunctionError(e, self.f, self.f) from e


@dataclass
class TransformedFunction:
    wrapped_fn: Union[callable, "TransformedFunction"]
    transformation: "Transformation"

    input_spec: Optional[Spec] = None
    passed_spec: Optional[Spec] = None
    returned_spec: Optional[Spec] = None
    output_spec: Optional[Spec] = None

    def __call__(self, *args, **kwargs):
        try:
            wrapped_fn = self.wrapped_fn
            # Handle special case where the wrapped function is not a
            # TransformedFunction (e.g. it's the kernel function)
            if not isinstance(self.wrapped_fn, TransformedFunction):
                wrapped_fn = _WrappedWithErrorHandling(self.wrapped_fn)
            tf = self.transformation.transform(
                wrapped_fn, self.input_spec, self.returned_spec
            )
            return tf(*args, **kwargs)
        # Handle errors that can occur while running the transformed function
        except TransformedFunctionError as e:
            # A nested step raised an error. Let's add this step to the stack trace
            raise TransformedFunctionError(
                e.original_exception, self, e.error_step
            ) from e.original_exception
        except Exception as e:
            raise TransformedFunctionError(e, self, self) from e

    def build(self, input_spec: Spec) -> "TransformedFunction":
        try:
            # The spec that will be passed into the wrapped function
            passed_spec = self.transformation.transform_input_spec(input_spec)

            if isinstance(self.wrapped_fn, TransformedFunction):
                # Recursively build the wrapped function
                wrapped_fn = self.wrapped_fn.build(passed_spec)
                returned_spec = (
                    wrapped_fn.output_spec
                    if wrapped_fn.output_spec is not None
                    else Spec()
                )
            else:
                # Base case: the wrapped function is just a function.
                wrapped_fn = self.wrapped_fn
                returned_spec = (
                    # If the function is specced, we use its transformed output spec
                    wrapped_fn.transform_output_spec(passed_spec)
                    if isinstance(wrapped_fn, Specced)
                    # Else, we assume that it returns a scalar (no dimensions)
                    else Spec(())
                )

            # Lastly, transform the output spec according to the Transformation.
            output_spec = self.transformation.transform_output_spec(returned_spec)
            return TransformedFunction(
                wrapped_fn,
                self.transformation,
                input_spec=input_spec,
                passed_spec=passed_spec,
                returned_spec=returned_spec,
                output_spec=output_spec,
            )
        # Handle errors that can occur while building
        except TransformedFunctionError as e:
            # A nested step raised an error. Let's reraise the exception.
            raise TransformedFunctionError(
                e.original_exception, self, e.error_step
            ) from e.original_exception
        except Exception as e:
            raise TransformedFunctionError(e, self, self) from e

    def traverse(self, *, depth_first: bool = False):
        if not depth_first:
            yield self
        if isinstance(self.wrapped_fn, TransformedFunction):
            yield from self.wrapped_fn.traverse(depth_first=depth_first)
        else:
            yield self.wrapped_fn
        if depth_first:
            yield self

    def __repr__(self):
        return _transformed_function_repr_fn(self)

    def __hash__(self):
        return object.__hash__(self)


class Transformation(ABC):
    """A Transformation takes a function and transforms/modifies it. It may also keep
    track of changes to the dimensions of the input and output of the function, by
    updating the Spec.

    transform takes the function-to-be-transformed and the input spec, and returns a
      new TransformedFunction object.
    transform_input_spec takes the input spec and returns a new spec that represents the
      data _after_ the transformation. E.g., if the transformation removes a dimension
      from a function argument, it should return a new spec with one less dimension for
      that argument.
    transform_output_spec takes the output spec from the transformed function and
      returns a new spec representing the returned data. E.g. if the transformation sums
      over a dimension in the returned data, it should return a new spec with that
      dimension removed."""

    def __call__(self, wrapped_fn) -> TransformedFunction:
        if isinstance(wrapped_fn, Transformation):
            return PartialTransformation([wrapped_fn, self])
        return TransformedFunction(wrapped_fn, self)

    @abstractmethod
    def transform(
        self, to_be_transformed: callable, input_spec: Spec, output_spec: Spec
    ) -> callable:
        ...

    @abstractmethod
    def transform_input_spec(self, spec: Spec) -> Spec:
        ...

    @abstractmethod
    def transform_output_spec(self, spec: Spec) -> Spec:
        ...


@dataclass
class PartialTransformation(Transformation):
    """A sequence of partially applied Transformations.

    In the following example, f1 and f2 are identical.
    p_tf = PartialTransformation([t1, t2, t3])
    f1 = p_tf(wrapped_fn)
    f2 = t3(t2(t1(wrapped_fn)))
    """

    partial_transformations: Sequence[Transformation]

    def __call__(self, wrapped_fn) -> TransformedFunction:
        for t in self.partial_transformations:
            if isinstance(t, PartialTransformation):
                # Makes sure that nested partial transformations are flattened
                wrapped_fn = t(wrapped_fn)
            else:
                wrapped_fn = TransformedFunction(wrapped_fn, t)
        return wrapped_fn

    def transform(
        self, to_be_transformed: callable, input_spec: Spec, output_spec: Spec
    ) -> callable:
        for t in self.partial_transformations:
            to_be_transformed = t.transform(to_be_transformed, input_spec, output_spec)
        return to_be_transformed

    def transform_input_spec(self, spec: Spec) -> Spec:
        for t in self.partial_transformations:
            spec = t.transform_input_spec(spec)
        return spec

    def transform_output_spec(self, spec: Spec) -> Spec:
        for t in self.partial_transformations:
            spec = t.transform_output_spec(spec)
        return spec

    def __repr__(self):
        return f"PartialTransformation({self.partial_transformations})"


class Wrap(Transformation):
    def __init__(self, f: callable, *args, **kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def transform(
        self, to_be_transformed: callable, input_spec: Spec, output_spec: Spec
    ) -> callable:
        return self.f(to_be_transformed, *self.args, **self.kwargs)

    def transform_input_spec(self, spec: Spec) -> Spec:
        return spec

    def transform_output_spec(self, spec: Spec) -> Spec:
        return spec

    def __repr__(self) -> str:
        args_str = ", ".join([str(arg) for arg in self.args])
        kwargs_str = ", ".join([f"{k}={str(v)}" for k, v in self.kwargs.items()])
        repr_str = f"Wrap({get_fn_name(self.f)}"
        if self.args:
            repr_str += f", {args_str}"
        if self.kwargs:
            repr_str += f", {kwargs_str}"
        # Make sure the repr string is not too long
        if len(repr_str) > 140:
            repr_str = repr_str[: (140 - len("… <truncated>"))] + "… <truncated>"
        return repr_str + ")"


@dataclass
class ForAll(Transformation):
    dimension: str

    def __init__(self, dimension: str):
        self.dimension = dimension

    def transform(
        self, to_be_transformed: callable, input_spec: Spec, output_spec: Spec
    ) -> callable:
        return specced_vmap(to_be_transformed, input_spec, self.dimension)

    def transform_input_spec(self, spec: Spec) -> Spec:
        # The returned function works on 1 element of the dimension at a time, so it has
        # one less dimension.
        return spec.remove_dimension(self.dimension)

    def transform_output_spec(self, spec: Spec) -> Spec:
        # We re-add the dimension after the function has been applied to each element.
        return spec.add_dimension(self.dimension)

    def __repr__(self) -> str:
        return f'ForAll("{self.dimension}")'


class Apply(Transformation):
    def __init__(self, f: Callable, *args, **kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def transform(
        self, to_be_transformed: callable, input_spec: Spec, output_spec: Spec
    ) -> callable:
        def with_applied_f(*args, **kwargs):
            result = to_be_transformed(*args, **kwargs)
            args, kwargs = concretize_axes(output_spec, self.args, self.kwargs)
            return self.f(result, *args, **kwargs)

        return with_applied_f

    def transform_input_spec(self, spec: Spec) -> Spec:
        return spec

    def transform_output_spec(self, spec: Spec) -> Spec:
        tree = (self.args, self.kwargs)
        for leaf in leaves(tree, lambda x: isinstance(x, Axis) or not has_treedef(x)):
            if isinstance(leaf.value, Axis):
                spec = spec.update_leaves(leaf.value.new_dimensions)
        return spec

    def __repr__(self) -> str:
        args_str = ", ".join([str(arg) for arg in self.args])
        kwargs_str = ", ".join([f"{k}={str(v)}" for k, v in self.kwargs.items()])
        repr_str = f"Apply({get_fn_name(self.f)}"
        if self.args:
            repr_str += f", {args_str}"
        if self.kwargs:
            repr_str += f", {kwargs_str}"
        # Make sure the repr string is not too long
        if len(repr_str) > 140:
            repr_str = repr_str[: (140 - len("… <truncated>"))] + "… <truncated>"
        return repr_str + ")"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
