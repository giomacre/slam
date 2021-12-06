from concurrent.futures import thread
from functools import partial
from time import sleep
import cv2 as cv
import numpy as np
from decorators import stateful_decorator

from worker import create_worker


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
        name="ImageViewer",
    )

    def draw_task(frames):
        observations = []
        for obs in frames[-1].observations:
            n_points = n_segments + 1 if (n := len(obs.frames)) > n_segments else n
            observations += [(obs.frames[-n_points:], obs.idxs[-n_points:])]
        return worker(
            (
                frames[-1].image,
                [
                    np.array(
                        [
                            frames[f_id].key_pts[p_id]
                            for f_id, p_id in zip(f_ids, p_ids)
                        ],
                        dtype=np.int32,
                    )
                    for f_ids, p_ids in observations
                ],
            )
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
