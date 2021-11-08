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
    keep,
    append_empty,
    default_value=lambda: None,
):
    old_args = deque(maxlen=keep)

    @wraps(function)
    def call(*args, **kwargs):
        return_value = default_value()
        if len(args) > 0 or append_empty:
            old_args.appendleft(list(args))
        if len(old_args) == keep:
            call_args = reduce(list.__add__, old_args)
            return_value = function(*call_args, **kwargs)
        return return_value

    return call
