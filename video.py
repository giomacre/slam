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
        self.video_path = path
        video = cv.VideoCapture(self.video_path)
        ret, frame = video.read()
        if not ret:
            exit(code=1)
        self.height, self.width, *_ = frame.shape

    def get_video_stream(self, filter=lambda _: True, downscale_factor=1):
        stream_size = (self.width // downscale_factor, self.height // downscale_factor)
        video = cv.VideoCapture(self.video_path)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if not filter(frame):
                continue
            if downscale_factor > 1:
                yield cv.resize(
                    frame,
                    stream_size,
                    interpolation=cv.INTER_AREA,
                )
            else:
                yield frame
        video.release()
