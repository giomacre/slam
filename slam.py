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
    create_orb_detector,
    create_orb_flann_matcher,
)
from geometry import (
    create_pose_estimator,
)

np.set_printoptions(precision=3, suppress=True)

video_path = sys.argv[1]
DOWNSCALE = 2
FX = 550
FY = FX

video = Video(
    video_path,
    create_frame_skip_filter(take_every=1),
    downscale_factor=DOWNSCALE,
)
frames = video.get_video_stream()
w, h = video.width, video.height
K = np.array(
    [
        [FX / DOWNSCALE, 0, w / 2, 0],
        [0, FY / DOWNSCALE, h / 2, 0],
        [0, 0, 1, 0],
    ]
)
pose_estimator = create_pose_estimator(
    K,
    create_orb_detector(),
    #create_orb_flann_matcher(),
    create_bruteforce_matcher(),
)

current_pose = np.eye(4)
matches_counts = []
for frame in frames:
    T, matches = pose_estimator.compute_pose(frame).values()
    matches = matches if matches is not None else []
    matches_returned = len(matches)
    drawing.draw_matches(
        frame["image"],
        matches,
    )
    if matches_returned == 0:
        continue
    matches_counts += [matches_returned]
    current_pose = T @ current_pose
    os.system("clear")
    print(
        f"frame {frame['frame_id']} returned {matches_returned} matches (mean {np.mean(matches_counts)})\n"
    )
    print(f"current_pose:\n{current_pose}\n")
