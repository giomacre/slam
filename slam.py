#!/usr/bin/env python3

from collections import deque
import os
import sys
import numpy as np
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
    # create_orb_flann_matcher(),
    create_bruteforce_matcher(),
)

current_pose = np.eye(4)
match_counts = deque(maxlen=100)
for frame in frames:
    T, matches = pose_estimator(frame).values()
    matches = matches if matches is not None else []
    matches_returned = len(matches)
    drawing.draw_matches(
        frame["image"],
        matches,
    )
    if matches_returned == 0:
        continue
    match_counts += [matches_returned]
    current_pose = T @ current_pose
    print(
        "frame {} returned {} matches (mean {:.0f})\n".format(
            frame["frame_id"],
            matches_returned,
            np.mean(match_counts),
        )
    )
    print(f"current_pose:\n{current_pose}\n")
