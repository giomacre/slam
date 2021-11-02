#!/usr/bin/env python3

import os
import sys
import numpy as np
import cv2 as cv
import drawing
from video import (
    Video,
    create_frame_skip_filter,
)
from features import (
    create_bruteforce_matcher,
)
from geometry import (
    create_orb_pose_estimator,
)

np.set_printoptions(precision=3, suppress=True)

video_path = sys.argv[1]
FX = 1
FY = FX

video = Video(
    video_path,
    create_frame_skip_filter(take_every=1),
    downscale_factor=2,
)
frames = video.get_video_stream()
matcher = create_bruteforce_matcher()
print(matcher.__name__)
w, h = video.width, video.height
K = np.array(
    [
        [FX, 0, w / 2, 0],
        [0, FY, h / 2, 0],
        [0, 0, 1, 0],
    ]
)
pose_estimator = create_orb_pose_estimator(
    K,
    matcher,
)

current_pose = np.eye(4)
matches_counts = []
for frame in frames:
    T, matches = pose_estimator.compute_pose(frame).values()
    matches_returned = len(matches) if matches is not None else 0
    matches_counts += [matches_returned]
    os.system("clear")
    print(
        f"frame {frame['frame_id']} returned {matches_returned} matches (mean {np.mean(matches_counts)})\n"
    )
    if matches_returned == 0:
        continue
    current_pose = T @ current_pose
    print(f"current_pose:\n{current_pose}\n")
    drawing.draw_matches(frame["image"], matches)
    if cv.waitKey(delay=1) == ord("q"):
        break
