from spekk import ops


def get_xyz(v: ops.array):
    x = ops.take(v, 0, axis="xyz")
    y = ops.take(v, 1, axis="xyz")
    z = ops.take(v, 2, axis="xyz")
    return x, y, z


def distance(point1: ops.array, point2: ops.array) -> float:
    return ops.linalg.vector_norm(point1 - point2, axis="xyz")
