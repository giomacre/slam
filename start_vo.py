#!/usr/bin/env python3
from visual_odometry import start

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentTypeError

    def check_file(path):
        from cv2 import VideoCapture

        video = VideoCapture(path)
        if video.isOpened():
            return path
        raise ArgumentTypeError(f"{path} is not a valid file.")

    parser = ArgumentParser()
    parser.add_argument(
        "video_path",
        type=check_file,
    )
    args = parser.parse_args()
    start(args.video_path)
