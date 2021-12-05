from collections import deque
from functools import reduce, wraps


class ddict(dict):
    __getattr__ = (
        lambda *args: item
        if type(item := dict.get(*args)) is not dict
        else ddict(item)
    )
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __dir__ = dict.keys


def use_arguments(decorator):
    def pass_parameters(*args, **kwargs):
        def with_parameters(function):
            return decorator(function, *args, **kwargs)

        return with_parameters

    return pass_parameters


@use_arguments
def stateful_decorator(
    function,
    *_,
    needs,
    use_defaults=False,
    default_return=None,
):
    old_args = deque(maxlen=needs)

    @wraps(function)
    def call(*args, **kwargs):
        return_value = default_return
        if len(args) > 0:
            old_args.appendleft(list(args))
        if len(old_args) == needs or use_defaults:
            call_args = reduce(list.__add__, old_args)
            return_value = function(*call_args, **kwargs)
        return return_value

    return call
