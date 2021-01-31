from math import cos, sin
from math import exp as real_exp
from builtins import round as python_round


def exp(value):
    if isinstance(value, complex):
        i = complex(0, 1)
        re = (value/i).real
        return cos(re) + i*sin(re)
    elif isinstance(value, int) or isinstance(value, float):
        return real_exp(value)
    else:
        raise ValueError


def round(number, n):
    if not isinstance(number, complex):
        return python_round(number, n)
    else:
        return round(number.real, n) + round(number.imag, n) * complex(0, 1)


def absargmax(array):
    """
    Finds the value and position of the element with the largest magnitude in a column vector.
    """
    canpos = 0
    canval = 0
    for index, value in enumerate(array):
        if abs(value[0]) > canval:
            canpos = index
            canval = abs(value[0])
    return canpos, canval


def rayleigh_quotient(A, v):
    """
    A and v are matrices, with v being a column vector.

    This implements (v.T * A * v) / (v.T * v).

    If v is an approximate eigenvector, we return the approximate corresponding eigenvalue.
    """
    top = v.transpose * (A*v)
    bottom = v.transpose * v
    return top[0][0]/bottom[0][0]
