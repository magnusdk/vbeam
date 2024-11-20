import spekk.transformations
from spekk.transformations import *

from vbeam.fastmath import numpy as api


def do_nothing(f):
    return f


class ForAll(spekk.transformations.ForAll):
    def __post_init__(self):
        self.vmap_impl = api.vmap
        super().__post_init__()


class Reduce(spekk.transformations.Reduce):
    def __post_init__(self):
        self.reduce_impl = api.reduce
        super().__post_init__()


__all__ = [*spekk.transformations.__all__, "do_nothing"]
