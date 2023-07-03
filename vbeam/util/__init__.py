from typing import Union, Sequence


def ensure_positive_index(n: int, index: Union[int, Sequence[int]]) -> int:
    """Take an index that may be negative, following numpy's negative index semantics,
    and ensure that it is positive. That is, if the index=-1, then it will be the last
    index of an array."""
    if isinstance(index, int):
        return n + index if index < 0 else index
    else:
        return [ensure_positive_index(n, i) for i in index]
