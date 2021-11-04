from collections import deque
from functools import reduce, update_wrapper, wraps


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


class StatefulDecorator:
    def __init__(self, function, keep=2):
        update_wrapper(self, function)
        self.__function__ = function
        self.__old_args__ = deque(maxlen=keep)
        self.__keep__ = keep

    def __call__(self, *args, **kwargs):
        return_value = None
        self.__old_args__.appendleft(list(args))
        if len(self.__old_args__) == self.__keep__:
            call_args = reduce(list.__add__, self.__old_args__)
            return_value = self.__function__(*call_args, **kwargs)
        return return_value
