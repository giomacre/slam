#!/usr/bin/env python3

from collections import deque
from functools import partial
from itertools import chain, islice
import os
import sys
import cv2 as cv
import numpy as np
from decorators import ddict
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
from visualization import create_map_thread
from worker import create_thread_context

np.set_printoptions(precision=3, suppress=True)

DOWNSCALE = 1
FX = 525
FY = FX

if __name__ == "__main__":
    video_path = sys.argv[1]
    video = Video(
        video_path,
        downscale_factor=DOWNSCALE,
        filter=create_frame_skip_filter(take_every=10),
    )
    logger = create_logger(lambda: video.frames_read)
    frames = video.get_video_stream()
    width, height = video.width, video.height
    K = np.array(
        [
            [FX / DOWNSCALE, 0, width / 2, 0],
            [0, FY / DOWNSCALE, height / 2, 0],
            [0, 0, 1, 0],
        ]
    )
    Kinv = np.linalg.inv(K[:3, :3])
    pose_estimator = create_pose_estimator(
        K,
        create_orb_detector(),
        # create_orb_flann_matcher(),
        create_bruteforce_matcher(),
    )
    thread_context = create_thread_context()
    send_draw_task = create_drawer_thread(thread_context)
    send_map_task = create_map_thread(
        1280,
        720,
        thread_context,
    )
    for worker in thread_context.threads:
        worker.start()

    tracked_frames = []
    points = np.empty((1, 4))
    for frame in frames:
        T, matches = pose_estimator(frame)
        wait_draw = send_draw_task((frame.image, matches))
        context = (
            frame.desc,
            len(matches),
            len(tracked_frames),
        )
        match context:
            case (None, *_): continue
            case (*_, 0): frame.pose = np.eye(4)
            case (_, 0, _): frame.pose = tracked_frames[-1].pose
            case _: frame.pose = T @ tracked_frames[-1].pose
        tracked_frames += [frame]

        # Triangulation (does not work), or does it?
        if len(matches) > 0:
            points_4d = np.array(
                cv.triangulatePoints(
                    (K @ tracked_frames[-1].pose)[:3],
                    (K @ tracked_frames[-2].pose)[:3],
                    matches[..., 1].T,
                    matches[..., 1].T,
                )
            ).T
            points_4d /= points_4d[:, -1:]
            points = np.vstack((points, points_4d))

        wait_map = send_map_task(
            (
                Kinv,
                [frame.pose for frame in tracked_frames],
                points,
                #np.empty((1,4)),
            )
        )
        if thread_context.terminated():
            break
        wait_draw()
        wait_map()

for thread in thread_context.threads:
    thread.join()
