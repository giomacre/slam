#!/usr/bin/env python3

import os
import sys
import numpy as np
import cv2 as cv
from video import (
    Video,
    create_frame_skip_filter,
)
from features import (
    create_stateful_matcher,
    create_bruteforce_matcher,
)
from geometry import (
    create_orb_pose_estimator,
)

video_path = sys.argv[1]
FX = 1
FY = FX

video = Video(
    video_path,
    create_frame_skip_filter(take_every=5),
    downscale_factor=2,
)
frames = video.get_video_stream()
matcher = create_stateful_matcher(create_bruteforce_matcher)
w, h = video.width, video.height
K = np.array(
    [
        [FX, 0, w / 2],
        [0, FY, h / 2],
        [0, 0, 1],
    ]
)
print(K)
pose_estimator = create_orb_pose_estimator(
    K,
    matcher.match_keypoints,
)

camera_R = np.eye(3)
position = np.zeros((3, 1))
for frame in frames:
    R, t, matches = pose_estimator.compute_pose(frame).values()
    matches_returned = len(matches) if matches is not None else 0
    os.system("clear")
    print(f"frame {frame['frame_id']} returned {matches_returned} matches \n")
    if matches is None:
        continue
    camera_R = R @ camera_R
    position = position + t
    print(f"rotation:\n{camera_R}\n")
    print(f"translation:\n{position}\n")
    matcher.draw_matches(frame["image"], matches)
    if cv.waitKey(delay=0) == ord("q"):
        break
