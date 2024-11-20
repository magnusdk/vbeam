import spekk.transformations
from fastmath import ops
from spekk.transformations import *


def do_nothing(f):
    return f


class ForAll(spekk.transformations.ForAll):
    def __post_init__(self):
        self.vmap_impl = ops.vmap
        super().__post_init__()


class Reduce(spekk.transformations.Reduce):
    def __post_init__(self):
        self.reduce_impl = ops.reduce
        super().__post_init__()


__all__ = [*spekk.transformations.__all__, "do_nothing"]
