from typing import Callable, List, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import lists

from minitorch import MathTest
import minitorch
from minitorch.operators import (
    add,
    addLists,
    eq,
    id,
    inv,
    inv_back,
    log_back,
    lt,
    max,
    mul,
    neg,
    negList,
    prod,
    relu,
    relu_back,
    log,
    exp,
    sigmoid,
)

from .strategies import assert_close, small_floats, positive_floats

# ## Task 0.1 Basic hypothesis tests.


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    """Check that the main operators all return the same value of the python version"""
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if abs(x) > 1e-5:
        assert_close(inv(x), 1.0 / x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a: float) -> None:
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a: float, b: float) -> None:
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_id(a: float) -> None:
    assert id(a) == a


@pytest.mark.task0_1
@given(small_floats)
def test_lt(a: float) -> None:
    """Check that a - 1.0 is always less than a"""
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_max(a: float) -> None:
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.task0_1
@given(small_floats)
def test_eq(a: float) -> None:
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


# ## Task 0.2 - Property Testing

# Implement the following property checks
# that ensure that your operators obey basic
# mathematical rules.


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_sigmoid(a: float, b: float) -> None:
    """Check properties of the sigmoid function, specifically
    * It is always between 0.0 and 1.0.
    * one minus sigmoid is the same as sigmoid of the negative
    * It crosses 0 at 0.5
    * It is  strictly increasing.
    """
    # TODO: Implement for Task 0.2.
    assert sigmoid(a) >= 0.0
    assert sigmoid(a) <= 1.0
    assert_close(1 - sigmoid(a), sigmoid(-a))
    assert sigmoid(0.0) == 0.5
    assert (a - b) * (sigmoid(a) - sigmoid(b)) >= 0


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a: float, b: float, c: float) -> None:
    """Test the transitive property of less-than (a < b and b < c implies a < c)"""
    # TODO: Implement for Task 0.2.
    if lt(a, b) and lt(b, c):
        assert lt(a, c)
    if lt(b, a) and lt(c, b):
        assert lt(c, a)


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(a: float, b: float) -> None:
    """Write a test that ensures that :func:`minitorch.operators.mul` is symmetric, i.e.
    gives the same value regardless of the order of its input.
    """
    # TODO: Implement for Task 0.2.
    assert mul(a, b) == mul(b, a)


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(z: float, x: float, y: float) -> None:
    r"""Write a test that ensures that your operators distribute, i.e.
    :math:`z \times (x + y) = z \times x + z \times y`
    """
    # TODO: Implement for Task 0.2.
    assert_close(mul(z, add(x, y)), add(mul(z, x), mul(z, y)))


@pytest.mark.task0_2
@given(positive_floats, positive_floats)
def test_logarithm(x: float, y: float) -> None:
    """Test the property of the log function:
    * log(x * y) = log(x) + log(y).
    * It is strictly increasing
    """
    # TODO: Implement for Task 0.2.
    if x > 1e-5 and y > 1e-5:
        assert_close(log(mul(x, y)), add(log(x), log(y)))
    assert (x - y) * (log(x) - log(y)) >= 0


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_exp(x: float, y: float) -> None:
    """Test the property of the exp function:
    * exp(x) * exp(y) = exp(x + y).
    * It crosses 0 at 1.
    * it is strictly increasing.
    """
    # TODO: Implement for Task 0.2.
    assert_close((exp(x) * exp(y) / exp(x + y)), 1)
    assert exp(0.0) == 1.0
    assert (x - y) * (exp(x) - exp(y)) >= 0


@pytest.mark.task0_2
@given(positive_floats, small_floats, small_floats)
def test_log_back(x: float, a: float, b: float) -> None:
    """Check properties of the log_back function, specifically
    * f(x, -b) = -f(x, b)
    * f(x, a) + f(x, b) = f(x, a + b)
    """
    # TODO: Implement for Task 0.2.
    assert log_back(x, -b) == -log_back(x, b)
    if x > 1e-5:
        assert_close(log_back(x, a) + log_back(x, b), log_back(x, a + b))


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_inv_back(x: float, a: float, b: float) -> None:
    """Check properties of the log_back function, specifically
    * f(-x, a) = f(x, a)
    * f(x, a) + f(x, b) = f(x, a + b)
    """
    # TODO: Implement for Task 0.2.
    assert inv_back(-x, a) == inv_back(x, a)
    if x > 1e-5:
        assert_close(inv_back(x, a) + inv_back(x, b), inv_back(x, a + b))


# ## Task 0.3  - Higher-order functions

# These tests check that your higher-order functions obey basic
# properties.


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a: float, b: float, c: float, d: float) -> None:
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1: List[float], ls2: List[float]) -> None:
    """Write a test that ensures that the sum of `ls1` plus the sum of `ls2`
    is the same as the sum of each element of `ls1` plus each element of `ls2`.
    """
    # TODO: Implement for Task 0.3.
    assert_close(
        minitorch.operators.sum(ls1) + minitorch.operators.sum(ls2),
        minitorch.operators.sum(addLists(ls1, ls2)),
    )


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls: List[float]) -> None:
    assert_close(sum(ls), minitorch.operators.sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x: float, y: float, z: float) -> None:
    assert_close(prod([x, y, z]), x * y * z)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls: List[float]) -> None:
    check = negList(ls)
    for i, j in zip(ls, check):
        assert_close(i, -j)


# ## Generic mathematical tests

# For each unit this generic set of mathematical tests will run.


one_arg, two_arg, _ = MathTest._tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn: Tuple[str, Callable[[float], float]], t1: float) -> None:
    name, base_fn = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(
    fn: Tuple[str, Callable[[float, float], float]], t1: float, t2: float
) -> None:
    name, base_fn = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a: float, b: float) -> None:
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
