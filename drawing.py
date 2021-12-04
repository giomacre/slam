from functools import partial
import cv2 as cv
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
    return create_worker(
        draw_loop,
        thread_context,
    )


def draw_matches(
    on_quit,
    image,
    matches,
):
    to_int = lambda x: tuple(int(round(c)) for c in x)
    image_with_matches = image.copy()
    for m in matches:
        current_pos = to_int(m[..., 0])
        last_pos = to_int(m[..., 1])
        cv.circle(
            image_with_matches,
            current_pos,
            radius=2,
            color=(0, 0, 255),
        )
        cv.line(
            image_with_matches,
            current_pos,
            last_pos,
            color=(255, 0, 0),
        )
    cv.imshow("", image_with_matches)
    if cv.waitKey(delay=1) == ord("q"):
        on_quit()
