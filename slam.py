#!/usr/bin/env python3

from operator import pos
import os
import numpy as np
import cv2 as cv
from video import Video
from features import (
    create_orb_flann_matcher,
    create_stateful_matcher,
    create_bruteforce_matcher,
)
from geometry import (
    create_orb_pose_estimator,
    unnormalize,
)

VIDEO_PATH = "./videos/new_york.raw"
FX = 50
FY = FX

video = Video(VIDEO_PATH)
frames = video.get_video_stream()
matcher = create_stateful_matcher(create_orb_flann_matcher)
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

camera_R = np.eye(3)
position = np.zeros((3, 1))
for frame in frames:
    frame_id, R, t, matches = pose_estimator.compute_pose(frame)
    os.system("clear")
    print(
        f"frame: {frame_id} returned {len(matches) if matches is not None else 0} matches \n"
    )
    if matches is None:
        continue
    camera_R = R @ camera_R
    position = position + t
    print(f"rotation:\n{camera_R}\n")
    print(f"translation:\n{position}\n")
    matcher.draw_matches(frame, unnormalize(K, matches))
    if cv.waitKey(delay=0) == ord("q"):
        break
