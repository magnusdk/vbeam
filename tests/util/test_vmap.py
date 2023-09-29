from vbeam.fastmath import Backend
from vbeam.util.vmap import vmap_all_except


def test_summing_one_axis_only(np: Backend):
    f = vmap_all_except(np.sum, 1)

    x = np.ones((2, 3, 4))
    result = f(x)
    assert result.shape == (2, 4), "Axis 1 is summed over and is removed from shape"
    assert (result == 3).all()


def test_f_produces_nd_array(np: Backend):
    @vmap_all_except(1)
    def f(x):
        return np.ones((3, 5, 6)) * np.sum(x)

    x = np.ones((2, 3, 4))
    result = f(x)
    assert result.shape == (2, 3, 5, 6, 4)
    assert (result == 3).all()


def test_only_one_axis_item_is_processed(np: Backend):
    x = np.zeros((2, 3, 4))

    @vmap_all_except(1)
    def f(x):
        assert x.shape == (3,)  # Only axis 1 is processed, rest is vectorized over
        return x

    result = f(x)
    assert result.shape == (2, 3, 4), "Shape is unchanged in no-op"
    assert (result == 0).all()
