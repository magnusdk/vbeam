import numpy
from spekk import trees


def assert_equal_treedefs(a: trees.Tree, b: trees.Tree):
    if trees.has_treedef(a):
        assert trees.has_treedef(b)
        a = trees.treedef(a)
        b = trees.treedef(b)
        assert a.keys() == b.keys()
        for k in a.keys():
            assert_equal_treedefs(a.get(k), b.get(k))
    else:
        numpy.testing.assert_equal(a, b)
