from collections import deque
from functools import reduce, wraps


class ddict(dict):
    __getattr__ = (
        lambda *args: item
        if type(item := dict.__getitem__(*args)) is not dict
        else ddict(item)
    )
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def use_parameters(decorator):
    @wraps(decorator)
    def pass_parameters(*args, **kwargs):
        def with_parameters(function):
            return decorator(function, *args, **kwargs)

        return with_parameters

    return pass_parameters


@use_parameters
def performance_timer(function, mean_window=250):
    from time import perf_counter_ns
    from numpy import mean

    function.times = deque(maxlen=mean_window)

    @wraps(function)
    def wrapper(*args, **kwargs):
        start = perf_counter_ns() * 1e-6
        return_value = function(*args, **kwargs)
        end = perf_counter_ns() * 1e-6
        function.times += [end - start]
        print(
            "{} returned in {:.3f} ms. (mean: {:.3f})".format(
                function.__name__,
                function.times[-1],
                mean(function.times),
            )
        )
        return return_value

    return wrapper


@use_parameters
def stateful_decorator(
    function,
    *_,
    keep,
    append_empty,
):
    old_args = deque(maxlen=keep)

    @wraps(function)
    def call(*args, **kwargs):
        return_value = None
        if len(args) > 0 or append_empty:
            old_args.appendleft(list(args))
        if len(old_args) == keep:
            call_args = reduce(list.__add__, old_args)
            return_value = function(*call_args, **kwargs)
        return return_value

    return call
