from collections import deque
from functools import reduce, wraps

def use_arguments(decorator):
    def pass_parameters(*args, **kwargs):
        def with_parameters(function):
            return decorator(function, *args, **kwargs)

        return with_parameters

    return pass_parameters


def handle_generator(
    generator_factory,
    *_,
    exception,
    handler,
):
    @wraps(generator_factory)
    def handle_exceptions(*args, **kwargs):
        generator = generator_factory(*args, **kwargs)
        while True:
            try:
                yield next(generator)
            except StopIteration:
                break
            except exception as e:
                handler(e)

        pass

    return handle_exceptions


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
