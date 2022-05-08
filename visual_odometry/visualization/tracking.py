from functools import partial
from itertools import islice
import cv2 as cv
import numpy as np
from ..utils.decorators import stateful_decorator
from ..utils.worker import create_worker


def create_drawer_thread(
    thread_context,
    n_segments=5,
):
    decorator = stateful_decorator(needs=1)
    draw_loop = decorator(
        partial(
            draw_matches,
            thread_context.close_context,
        )
    )
    worker = create_worker(
        draw_loop,
        thread_context,
        empty_queue_handler=lambda _: draw_loop(),
        name="ImageViewer",
    )

    def draw_task(frames):
        observations = []
        for pt in frames[-1].landmarks.values():
            last_n = reversed(
                [
                    *islice(
                        reversed(pt.observations.items()),
                        n_segments + 1,
                    )
                ]
            )
            observations += [last_n]
        return worker(
            frames[-1].image,
            [
                np.array(
                    [frames[f_id].key_pts[p_id] for f_id, p_id in obs],
                    dtype=np.int32,
                )
                for obs in observations
            ],
        )

    return draw_task


def draw_matches(on_quit, image, observations):
    image_with_matches = image
    for pts in observations:
        if len(pts) > 1:
            cv.polylines(
                image_with_matches,
                [pts],
                isClosed=False,
                color=(0, 255, 0),
                thickness=1,
            )
        cv.circle(
            image_with_matches,
            pts[-1],
            radius=3,
            color=(0, 0, 255),
        )
    cv.imshow("", image_with_matches)
    if cv.waitKey(delay=1) == ord("q"):
        on_quit()
