#!/usr/bin/env python3

from functools import partial, reduce
import sys
from time import sleep
import cv2
import numpy as np
from decorators import ddict
from drawing import create_drawer_thread
from optical_flow import create_lk_orb_detector, create_lk_tracker
from video import (
    Video,
    skip_items,
)
from features import (
    create_bruteforce_matcher,
    create_orb_detector,
)
from geometry import (
    create_point_triangulator,
    create_pose_estimator,
)
from visualization import create_map_thread
from worker import create_thread_context
from threading import current_thread

np.set_printoptions(precision=3, suppress=True)


DOWNSCALE = 1
FX = 525
FY = FX

N_FEATURES = 350
KEYFRAME_THRESHOLD = 0.75

if __name__ == "__main__":
    video_path = sys.argv[1]
    video = Video(
        video_path,
        downscale_factor=DOWNSCALE,
    )
    width, height = video.width, video.height

    thread_context = create_thread_context()
    send_draw_task = create_drawer_thread(thread_context, n_segments=10)

    video_stream = video.get_video_stream()
    initialize_pose = partial(
        skip_items,
        video_stream,
        take_every=6,
        default_behaviour=lambda f: send_draw_task((f.image, [])),
    )

    K = np.array(
        [
            [FX / DOWNSCALE, 0, width / 2, 0],
            [0, FY / DOWNSCALE, height / 2, 0],
            [0, 0, 1, 0],
        ]
    )
    Kinv = np.linalg.inv(K[:3, :3])
    detector = create_lk_orb_detector(
        scoreType=cv2.ORB_FAST_SCORE,
    )
    pose_estimator = create_pose_estimator(
        K,
        create_lk_tracker(),
    )
    triangulation = create_point_triangulator(K)
    send_map_task = create_map_thread(
        (1280, 720),
        Kinv,
        (width, height),
        thread_context,
    )
    thread_context.start()

    frames = video_stream
    tracked_frames = []
    last_keyframe = None
    for frame in frames:
        frame.id = len(tracked_frames)
        if frame.id == 0:
            frame = detector(frame, N_FEATURES)
            if len(frame.key_pts) == 0:
                continue
            frame.pose = np.eye(4)
            frame.is_keyframe = True
            last_keyframe = frame
        else:
            query_idxs, train_idxs = pose_estimator(
                frame,
                tracked_frames[-1],
            )
            num_tracked = len(query_idxs)
            if num_tracked == 0:
                continue
            frame.observations = [None] * num_tracked
            for i in range(len(query_idxs)):
                landmark = tracked_frames[-1].observations[train_idxs[i]]
                landmark.frames += [frame.id]
                landmark.idxs += [i]
                frame.observations[i] = landmark
            current_pts = frame.key_pts[query_idxs]
            current_obs = frame.observations
            if (
                len(
                    [
                        x
                        for x in frame.observations
                        if np.isin(
                            last_keyframe.id,
                            x.frames,
                        )
                    ]
                )
                / len(last_keyframe.observations)
                < KEYFRAME_THRESHOLD
            ):
                frame.is_keyframe = True
                last_keyframe = frame
                frame = detector(
                    frame,
                    N_FEATURES - num_tracked,
                    current_pts,
                )
                if len(frame.key_pts) > 0:
                    current_pts = np.vstack(
                        [
                            current_pts,
                            frame.key_pts,
                        ]
                    )
                    current_obs = [
                        *current_obs,
                        *[
                            ddict(frames=[frame.id], idxs=[i])
                            for i in range(
                                num_tracked,
                                num_tracked + len(frame.key_pts),
                            )
                        ],
                    ]
            frame.key_pts = current_pts
            frame.observations = current_obs
        tracked_frames += [frame]
        wait_draw = send_draw_task(tracked_frames)
        wait_map = send_map_task(
            (
                [frame.pose for frame in tracked_frames],
                [],
            )
        )
        wait_draw()
        wait_map()
        if thread_context.is_closed:
            break
    thread_context.wait_close()
    thread_context.cleanup()
    thread_context.join_all()
    print(f"{current_thread()} exiting.")
