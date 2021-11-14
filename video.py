from functools import wraps
import cv2 as cv
from decorators import ddict


def skip_items(
    iterator,
    take_every=2,
    default_behaviour=lambda _: None,
):
    def __skip_items__():
        count = 0
        while True:
            count += 1
            try:
                current_value = next(iterator)
            except StopIteration:
                break
            if count % take_every == 0:
                yield current_value
            else:
                default_behaviour(current_value)

    return __skip_items__()


class Video:
    def __init__(
        self,
        path,
        downscale_factor=1,
    ):
        self.__filter__ = filter
        self.__count__ = 0
        self.__downscale_factor__ = downscale_factor
        self.__video__ = cv.VideoCapture(path)
        self.height, self.width, *_ = (
            int(self.__video__.get(p)) // downscale_factor
            for p in [
                cv.CAP_PROP_FRAME_HEIGHT,
                cv.CAP_PROP_FRAME_WIDTH,
            ]
        )

    @property
    def frames_read(self):
        return self.__count__

    def get_video_stream(self):
        while self.__video__.isOpened():
            ret, frame = self.__video__.read()
            if not ret:
                break
            self.__count__ += 1
            if self.__downscale_factor__ > 1:
                frame = cv.resize(
                    frame,
                    (self.width, self.height),
                    interpolation=cv.INTER_AREA,
                )
            yield ddict(image=frame)
        self.release()

    def release(self):
        self.__video__.release()
