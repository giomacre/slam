#!/usr/bin/env python3

import os
import numpy as np
import cv2 as cv
from video import Video
from features import (
    create_orb_flann_matcher,
)
from geometry import (
    create_orb_pose_estimator,
    unnormalize,
)

VIDEO_PATH = "./videos/drone.webm"
FX = 1
FY = FX

video = Video(VIDEO_PATH)
frames = video.get_video_stream()
matcher = create_orb_flann_matcher()
w, h = video.width, video.height
K = np.array(
    [
        [FX, 0, w / 2],
        [0, FY, h / 2],
        [0, 0, 1],
    ]
)
pose_estimator = create_orb_pose_estimator(
    K,
    matcher.match_keypoints,
)


for frame in frames:
    R, t, matches = pose_estimator.compute_pose(frame)
    if matches is None:
        continue
    os.system("clear")
    print(f"rotation:\n{R}\n")
    print(f"translation:\n{t}\n")
    print(f"good_points:\n{len(matches)}\n")
    matcher.draw_matches(frame, unnormalize(K, matches))
    if cv.waitKey(delay=0) == ord("q"):
        break
