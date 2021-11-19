#!/usr/bin/env python3

from functools import partial
import sys
import cv2
import numpy as np
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


def clean_frame(frame):
    del (
        frame.desc,
        frame.image,
        frame.points,
        frame.key_pts,
        frame.origin_frames,
        frame.origin_pts,
        frame.tracked_idxs,
    )


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
                nfeatures=500,
                scoreType=cv2.ORB_FAST_SCORE,
            ),
            min_points = 1000,
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
            good_points = pose_estimator(frame)
        else:
            good_points = pose_estimator(frame, tracked_frames[-1])
        pts = np.dstack(
            [
                frame.key_pts[frame.tracked_idxs],
                frame.origin_pts[frame.tracked_idxs],
            ]
        )
        wait_draw = send_draw_task((frame.image, pts))
        context = (
            len(frame.key_pts),
            good_points,
            len(tracked_frames),
        )
        match context:
            case (0, *_):
                continue
            case (*_, 0):
                frames = initialize_pose()
                frame.pose = np.eye(4)
                frame.points = []
            case (_, 0, _):
                frames = initialize_pose()
                continue
            case _:
                candidates = frame.id - frame.origin_frames > 1
                idx_sort = np.argsort(frame.origin_frames[candidates], kind="stable")
                sorted = frame.origin_frames[candidates][idx_sort]
                frame_ids, idx_first_occurences = np.unique(sorted, return_index=True)
                position = lambda f: f.pose[:3, 3]
                selected = np.fromiter(
                    (
                        (
                            (
                                d := np.linalg.norm(
                                    position(tracked_frames[i]) - position(frame)
                                )
                            )
                            > 1.25
                            and d < 12.5
                            for i in frame_ids
                        )
                    ),
                    np.bool8,
                )
                good_refs = np.isin(
                    frame.origin_frames[candidates],
                    frame_ids[selected],
                )
                frame.points = []
                for origin_frame, (current_pts, origin_pts) in zip(
                    map(
                        lambda id: tracked_frames[id],
                        frame_ids[selected],
                    ),
                    map(
                        lambda i: (
                            frame.key_pts[candidates][i],
                            frame.origin_pts[candidates][i],
                        ),
                        np.array(
                            np.split(
                                idx_sort,
                                idx_first_occurences[1:],
                            )
                        )[selected],
                    ),
                ):
                    triangulated = triangulation(
                        frame.pose,
                        origin_frame.pose,
                        current_pts,
                        origin_pts,
                    )[:, :3]
                    frame.points = (
                        triangulated
                        if len(frame.points) == 0
                        else np.vstack((frame.points, triangulated))
                    )
                clean_frame(tracked_frames[-1])
                frames = video_stream
        tracked_frames += [frame]
        wait_map = send_map_task(
            (
                [frame.pose for frame in tracked_frames],
                frame.points,
            )
        )
        if thread_context.terminated():
            break
        wait_draw()
        wait_map()

for thread in thread_context.threads:
    thread.join()
