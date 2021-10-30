import cv2 as cv

SCALE = 2


class Video:
    def __init__(self, path) -> None:
        self.video_path = path
        video = cv.VideoCapture(self.video_path)
        ret, frame = video.read()
        if not ret:
            exit(code=1)
        self.height, self.width, *_ = (x // SCALE for x in frame.shape)

    def get_video_stream(self):
        video = cv.VideoCapture(self.video_path)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            yield cv.resize(
                frame,
                (self.width, self.height),
                interpolation=cv.INTER_AREA,
            ) if SCALE > 1 else frame
        video.release()
