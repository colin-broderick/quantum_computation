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
