#!/usr/bin/env python3

import os
import numpy as np
import cv2 as cv
from video import Video
from features import (
    compute_orb_features,
    create_stateful_orb_flann_matcher,
)

VIDEO_PATH = "./videos/drone.webm"
FX = 350
FY = FX

video = Video(VIDEO_PATH)
frames = video.get_video_stream()
matcher = create_stateful_orb_flann_matcher()

w, h = video.width, video.height
K = np.array(
    [
        [FX, 0, w / 2],
        [0, FY, h / 2],
        [0, 0, 1],
    ]
)
K_inv = np.linalg.inv(K)
to_homogeneous = lambda x: np.concatenate(
    (
        x,
        np.ones(shape=(x.shape[0], 1, x.shape[2])),
    ),
    axis=1,
)

for frame in compute_orb_features(frames):
    matches = matcher.match_keypoints(frame)
    if matches is None:
        continue
    matches = to_homogeneous(matches)
    normalized = K_inv @ matches
    E, mask = cv.findEssentialMat(
        normalized[:, :2, 0],
        normalized[:, :2, 1],
        K,
    )
    _, R, t, mask = cv.recoverPose(
        E,
        normalized[:, :2, 0],
        normalized[:, :2, 1],
        K,
        mask=mask,
    )
    os.system("clear")
    print(f"rotation:\n{R}\n")
    print(f"translation:\n{t}\n")
    print(f"good_points:\n{np.count_nonzero(mask)}\n")
    matcher.draw_matches()
    if cv.waitKey(delay=0) == ord("q"):
        break
