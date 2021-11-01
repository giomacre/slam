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
    def __init__(self, path):
        self.__path__ = path
        self.__count__ = 0
        video = cv.VideoCapture(self.__path__)
        ret, frame = video.read()
        if not ret:
            exit(code=1)
        self.height, self.width, *_ = frame.shape

    def get_video_stream(self, filter=lambda _: True, downscale_factor=1):
        stream_size = (self.width // downscale_factor, self.height // downscale_factor)
        video = cv.VideoCapture(self.__path__)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            self.__count__ += 1
            if not filter(frame):
                continue
            if downscale_factor > 1:
                frame = cv.resize(
                    frame,
                    stream_size,
                    interpolation=cv.INTER_AREA,
                )
            yield {
                "frame_id": self.__count__,
                "image": frame,
            }
        video.release()
