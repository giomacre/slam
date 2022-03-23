#!/usr/bin/env python3

from functools import partial, reduce
import sys
from time import sleep
import cv2
import numpy as np
from decorators import ddict
from drawing import create_drawer_thread
from frame import create_frame
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
    tracker = create_lk_tracker()
    pose_estimator = create_pose_estimator(K)
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
    for image in frames:
        frame = create_frame(len(tracked_frames), image)
        if frame.id == 0:
            frame = detector(frame, N_FEATURES)
            if len(frame.key_pts) == 0:
                continue
            frame.pose = np.eye(4)
            frame.is_keyframe = True
            last_keyframe = frame
        else:

            matches, query_idxs, train_idxs = tracker(
                frame,
                tracked_frames[-1],
            )
            if len(matches) == 0:
                continue
            # Filter outliers with RANSAC
            S, inliers = pose_estimator(matches)
            if S is None:
                continue
            train_idxs = train_idxs[inliers]
            frame.key_pts = matches[inliers, ..., 0]
            # Compute the transform with respect to the last keyframe
            kf_idxs = np.array(
                [
                    tracked_frames[-1].observations[i].idxs[last_keyframe.id]
                    for i in train_idxs
                ]
            )
            S, inliers = pose_estimator(
                np.dstack(
                    (
                        frame.key_pts,
                        last_keyframe.key_pts[
                            kf_idxs,
                        ],
                    )
                )
            )
            if S is None:
                continue
            num_tracked = sum(inliers)
            train_idxs = train_idxs[inliers]
            frame.observations = [None] * num_tracked
            for i in range(num_tracked):
                landmark = tracked_frames[-1].observations[train_idxs[i]]
                landmark.idxs |= {frame.id: i}
                frame.observations[i] = landmark
            frame.pose = S @ last_keyframe.pose
            current_obs = frame.observations
            current_pts = frame.key_pts[inliers]
            if num_tracked / len(last_keyframe.key_pts) < KEYFRAME_THRESHOLD:
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
                            ddict(idxs={frame.id: i})
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
