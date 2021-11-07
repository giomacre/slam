import cv2 as cv

from worker import create_worker_process


def create_drawer_process():
    return create_worker_process(draw_matches)


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
    cv.waitKey(delay=1)
