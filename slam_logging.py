from collections import deque
from functools import partial

from numpy import mean

from decorators import ddict


def create_logger(frame_id):
    previous_counts = deque(maxlen=100)

    def log_matches(frame_id, previous_counts, matches):
        last_count = len(matches) if matches is not None else 0
        previous_counts += [last_count]

        print(
            "frame {} returned {} valid matches (mean {:.0f})\n".format(
                frame_id(),
                last_count,
                mean(previous_counts),
            )
        )

    return ddict(
        log_matches=partial(log_matches, frame_id, previous_counts),
        log_pose=lambda pose: print(pose),
    )
