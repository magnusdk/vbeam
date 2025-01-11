from hypothesis import strategies as st
from spekk import ops

from vbeam.geometry import Direction, Orientation


def angles():
    return st.floats(min_value=-ops.pi, max_value=ops.pi)


@st.composite
def directions(draw):
    return Direction(draw(angles()), draw(angles()))


@st.composite
def orientations(draw):
    return Orientation(draw(angles()), draw(angles()), draw(angles()))


@st.composite
def points(draw):
    x = draw(st.floats(min_value=-2, max_value=2))
    y = draw(st.floats(min_value=-2, max_value=2))
    z = draw(st.floats(min_value=-2, max_value=2))
    return ops.array([x, y, z], ["xyz"])
