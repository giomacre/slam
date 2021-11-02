from functools import reduce, update_wrapper, wraps


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
