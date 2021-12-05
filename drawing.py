from functools import partial
from time import sleep
import cv2 as cv
import numpy as np
from decorators import stateful_decorator

from worker import create_worker


def create_drawer_thread(thread_context):
    decorator = stateful_decorator(needs=1)
    draw_loop = decorator(
        partial(
            draw_matches,
            thread_context.close,
        )
    )
    worker = create_worker(
        draw_loop,
        thread_context,
    )

    def draw_task(frame):
        n_segments = 5
        observations = []
        for obs in frame.observations:
            n_points = n_segments + 1 if (n := len(obs.frames)) > n_segments else n
            observations += [(obs.frames[-n_points:], obs.idxs[-n_points:])]
        return worker((frame.image, observations))

    return draw_task


def draw_matches(on_quit, image, observations):
    to_int = lambda x: np.round(x).astype(np.int32)
    image_with_matches = image
    for frames, idxs in observations:
        pts = np.array(
            [to_int(f.key_pts[idx]) for f, idx in zip(frames, idxs)],
            dtype=np.int32,
        )

        if len(pts) > 1:
            cv.polylines(
                image_with_matches,
                [pts],
                isClosed=False,
                color=(128, 0, 0),
                thickness=1,
            )
        cv.circle(
            image_with_matches,
            pts[-1],
            radius=2,
            color=(0, 0, 255),
        )
    cv.imshow("", image_with_matches)
    if cv.waitKey(delay=1) == ord("q"):
        on_quit()
