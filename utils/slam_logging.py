from collections import deque
from functools import partial, wraps
from numpy import mean
from utils.decorators import use_arguments


@use_arguments
def performance_timer(
    function,
    *_,
    mean_window=250,
    value_analysis=lambda _: "",
):
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
            "{} returned {} in {:.3f} ms. (mean: {:.3f})".format(
                function.__name__,
                value_analysis(return_value),
                function.times[-1],
                mean(function.times),
            )
        )
        return return_value

    return wrapper


log_pose_estimation = performance_timer(
    value_analysis=lambda v: f", and {v} points passed the cheirality check"
)
log_feature_extraction = performance_timer(
    value_analysis=lambda v: f"{len(v.key_pts)} features"
)
log_feature_match = performance_timer(value_analysis=lambda v: f"{len(v[0])} matches")
log_triangulation = performance_timer(value_analysis=lambda v: f"{len(v)} points")
