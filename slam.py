#!/usr/bin/env python3

from functools import partial, reduce
import sys
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

np.set_printoptions(precision=3, suppress=True)


DOWNSCALE = 1
FX = 525
FY = FX

count = []

if __name__ == "__main__":
    video_path = sys.argv[1]
    video = Video(
        video_path,
        downscale_factor=DOWNSCALE,
    )
    width, height = video.width, video.height

    thread_context = create_thread_context()
    send_draw_task = create_drawer_thread(thread_context)

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
    pose_estimator = create_pose_estimator(
        K,
        # create_bruteforce_matcher(
        #     create_orb_detector(nfeatures=1000),
        #     normType=cv2.NORM_HAMMING,
        # ),
        create_lk_tracker(
            create_lk_orb_detector(
                scoreType=cv2.ORB_FAST_SCORE,
            ),
            min_points=1500,
            max_points=2000,
        ),
    )
    triangulation = create_point_triangulator(K)
    send_map_task = create_map_thread(
        (1280, 720),
        Kinv,
        (width, height),
        thread_context,
    )
    for worker in thread_context.threads:
        worker.start()

    frames = video_stream
    tracked_frames = []
    for frame in frames:
        frame.id = len(tracked_frames)
        if len(tracked_frames) == 0:
            query_idxs, _ = pose_estimator(frame)
        else:
            query_idxs, train_idxs = pose_estimator(
                frame,
                tracked_frames[-1],
            )
        context = (
            len(frame.key_pts),
            len(query_idxs),
            len(tracked_frames),
        )
        match context:
            case (0, *_):
                continue
            case (*_, 0):
                frames = initialize_pose()
                frame.pose = np.eye(4)
                frame.observations = [
                    ddict(
                        frames=[frame],
                        idxs=[i],
                    )
                    for i in range(len(frame.key_pts))
                ]
            case (_, 0, _):
                frames = initialize_pose()
                continue
            case _:
                frame.observations = [
                    ddict(
                        frames=[frame],
                        idxs=[i],
                    )
                    for i in range(len(frame.key_pts))
                ]
                for i in range(len(query_idxs)):
                    landmark = tracked_frames[-1].observations[train_idxs[i]]
                    landmark.frames += [frame]
                    landmark.idxs += [query_idxs[i]]
                    frame.observations[query_idxs[i]] = landmark
                frames = video_stream
        tracked_frames += [frame]
        wait_draw = send_draw_task(frame)
        wait_map = send_map_task(
            (
                [frame.pose for frame in tracked_frames],
                [],
            )
        )
        if thread_context.terminated():
            break
        # wait_draw()
        # wait_map()

for thread in thread_context.threads:
    thread.join()
