import warnings


def deprecated(func):
    def func_wrapper(*args):
        warnings.warn("'{}' is deprecated and should not be used as is".format(func.__name__))
        func(*args)
    return func_wrapper
