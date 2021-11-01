import cv2 as cv


def create_frame_skip_filter(take_every=2):
    def generator():
        count = 0
        while True:
            count += 1
            yield count % take_every == 0

    generator = generator()
    return lambda _: next(generator)


class Video:
    def __init__(
        self,
        path,
        filter=lambda _: True,
        downscale_factor=1,
    ):
        self.__path__ = path
        self.__count__ = 0
        self.__filter__ = filter
        self.__downscale_factor__ = downscale_factor
        video = cv.VideoCapture(self.__path__)
        ret, frame = video.read()
        if not ret:
            exit(code=1)
        self.height, self.width, *_ = (x // downscale_factor for x in frame.shape)

    def get_video_stream(self):
        video = cv.VideoCapture(self.__path__)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            self.__count__ += 1
            if not self.__filter__(frame):
                continue
            if self.__downscale_factor__ > 1:
                frame = cv.resize(
                    frame,
                    (self.width, self.height),
                    interpolation=cv.INTER_AREA,
                )
            yield {
                "frame_id": self.__count__,
                "image": frame,
            }
        self.release()

        def release():
            video.release()
