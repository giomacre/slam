import cv2 as cv

SCALE = 2

def video_stream(video_path):
    video = cv.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        height, width, *_ = frame.shape
        frame = cv.resize(
            frame,
            (width // SCALE, height // SCALE),
            interpolation=cv.INTER_AREA,
        )
        yield frame
    video.release()
