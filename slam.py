#!/usr/bin/env python3

from collections import deque
from itertools import islice
import sys
import numpy as np
from drawing import create_drawer_thread
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
from slam_logging import create_logger

np.set_printoptions(precision=3, suppress=True)

DOWNSCALE = 1
FX = 525
FY = FX

if __name__ == "__main__":
    video_path = sys.argv[1]
    video = Video(
        video_path,
        downscale_factor=DOWNSCALE,
    )
    logger = create_logger(lambda: video.frames_read)
    send_draw_task = create_drawer_thread()
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

    tracked_frames = []
    current_pose = np.eye(4)
    for frame in frames:
        T, matches = pose_estimator(frame)
        matches = matches if matches is not None else []
        send_draw_task((frame.image, matches))
        context = (frame.desc, T, len(tracked_frames))
        match context:
            case (None, *_): continue
            case (*_, 0): frame.pose = np.eye(4)
            case (_, None, n) if n > 0: frame.pose = tracked_frames[-1].pose
            case _: frame.pose = T @ tracked_frames[-1].pose
        tracked_frames += [frame]
        # os.system("cls 2>/dev/null || clear")
        logger.log_matches(matches)
        logger.log_pose(frame.pose)
