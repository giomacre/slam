from functools import partial
from multiprocessing import Process, Queue
import cv2 as cv
from numpy import block


def create_drawer_thread():
    queue = Queue()

    def drawer_thread(queue):
        while True:
            draw_matches(*queue.get(timeout=0.5))

    thread = Process(
        target=drawer_thread,
        args=(queue,),
        daemon=True,
    )
    thread.start()
    return partial(queue.put, block=False)


def draw_matches(image, matches):
    to_int = lambda x: tuple(int(round(c)) for c in x)
    image_with_matches = image.copy()
    for m in matches:
        current_pos = to_int(m[..., 0])
        last_pos = to_int(m[..., 1])
        cv.circle(
            image_with_matches,
            current_pos,
            radius=4,
            color=(0, 0, 255),
        )
        cv.circle(
            image_with_matches,
            last_pos,
            radius=2,
            color=(255, 0, 0),
        )
        cv.line(
            image_with_matches,
            current_pos,
            last_pos,
            color=(0, 255, 0),
        )
    cv.imshow("", image_with_matches)
    if cv.waitKey(delay=1) == ord("q"):
        exit(code=0)
