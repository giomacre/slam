#!/usr/bin/env python3

import numpy as np
import cv2 as cv
from video import video_stream
from features import (
    compute_orb_features,
    create_stateful_orb_flann_matcher,
)

VIDEO_PATH = "./drone.webm"

frames = video_stream(VIDEO_PATH)
matcher = create_stateful_orb_flann_matcher()

for frame in compute_orb_features(frames):
    matches = matcher.match_keypoints(frame)
    if matches is not None:
        matcher.draw_matches(matches)
    if cv.waitKey(delay=1) == ord("q"):
        break
