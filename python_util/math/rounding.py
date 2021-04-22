import numpy as np


def round_to_nearest_integer(x):
    """ Round the value x to the nearest integer. This method is necessary since in Python 3 the builtin
    round() function is performing Bankers rounding, i.e. rounding to the nearest even integer value.

    :param x: value to be rounded
    :type x: Union[int, float]
    :return: the closest integer to x
    """
    if x % 1 >= 0.5:
        return int(x) + 1
    else:
        return int(x)


def round_by_precision_and_base(x, prec=2, base=1.0):
    """ Round the value `x` to the nearest multiple of `base` and return the value with a precision `prec`.

    :param x: value to be rounded (can be an array)
    :param prec: precision of final rounding
    :param base: only multiples of this value are considered as results
    :return: rounded value of `x` to nearest multiple of `base`
    """
    return (base * (np.array(x) / base).round()).round(prec)


if __name__ == '__main__':
    x = 1.132354235234234
    print(x)
    print(round_by_precision_and_base(x, base=0.0001, prec=2))
    print(round_by_precision_and_base(x, base=0.0001, prec=4))
    print(round_by_precision_and_base(x, base=1))
    print(round_by_precision_and_base(x, base=10))
    print(round_by_precision_and_base(x, base=100))
