#!/usr/bin/env python3

import sys
import numpy as np
from drawing import create_drawer_thread
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

if __name__ == "__main__":
    video_path = sys.argv[1]
    video = Video(
        video_path,
        downscale_factor=DOWNSCALE,
    )
    video_stream = video.get_video_stream()
    frames = video_stream
    frame_skip = skip_items(frames, take_every=3)
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
        create_bruteforce_matcher(),
    )
    triangulation = create_point_triangulator(K)
    thread_context = create_thread_context()
    send_draw_task = create_drawer_thread(thread_context)
    send_map_task = create_map_thread(
        (1280, 720),
        (width, height),
        thread_context,
    )
    for worker in thread_context.threads:
        worker.start()
    tracked_frames = []
    for frame in frames:
        frame.id = len(tracked_frames)
        if len(tracked_frames) == 0:
            matches = pose_estimator(frame)
        else:
            matches = pose_estimator(frame, tracked_frames[-1])
        wait_draw = send_draw_task((frame.image, matches))
        context = (
            frame.desc,
            len(matches),
            len(tracked_frames),
        )
        match context:
            case (None, *_):
                continue
            case (*_, 0):
                frames = skip_items(video_stream, take_every=2)
                frame.pose = np.eye(4)
                frame.points = np.empty((1, 4))
            case (_, 0, _):
                frames = skip_items(video_stream, take_every=2)
                frame.pose = tracked_frames[-1].pose
                frame.points = tracked_frames[-1].points
            case _:
                # idx_sort = np.argsort(frame.origin_frames, kind="stable")
                # sorted = frame.origin_frames[idx_sort]
                # frame_ids, idx_first_occurences = np.unique(sorted, return_index=True)
                # print(
                #     [
                #         *zip(
                #             # map(
                #             #     lambda id: tracked_frames[id]
                #             #     if id < len(tracked_frames)
                #             #     else frame,
                #             #     frame_ids,
                #             # ),
                #             frame_ids,
                #             np.split(
                #                 idx_sort,
                #                 idx_first_occurences[1:],
                #             ),
                #         )
                #     ]
                # )
                frames = video_stream
                frame.points = triangulation(
                    frame.pose,
                    tracked_frames[-1].pose,
                    matches,
                )
        tracked_frames += [frame]
        wait_map = send_map_task(
            (
                Kinv,
                [frame.pose for frame in tracked_frames],
                frame.points,
                # np.vstack([frame.points for frame in tracked_frames]),
            )
        )
        if thread_context.terminated():
            break
        wait_draw()
        wait_map()

for thread in thread_context.threads:
    thread.join()
