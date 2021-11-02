from functools import reduce, update_wrapper, wraps


def performance_timer(function):
    from time import perf_counter_ns
    from numpy import mean

    @wraps(function)
    def wrapper(*args, **kwargs):
        start = perf_counter_ns() * 1e-6
        return_value = function(*args, **kwargs)
        end = perf_counter_ns() * 1e-6
        print(f"{function.__name__} exited in {end -start:.3} ms.")
        return return_value

    return wrapper


class StatefulDecorator:
    def __init__(self, function, keep=1):
        update_wrapper(self, function)
        self.__function__ = function
        self.__old_args__ = []
        self.__keep__ = keep

    def __call__(self, *args, **kwargs):
        return_value = None
        if len(self.__old_args__) >= self.__keep__:
            call_args = reduce(list.__add__, self.__old_args__) + list(args)
            return_value = self.__function__(*call_args, **kwargs)
        self.__old_args__ += [list(args)]
        self.__old_args__ = self.__old_args__[-self.__keep__ :]
        return return_value
