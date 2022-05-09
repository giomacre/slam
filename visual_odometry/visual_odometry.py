from functools import partial
from cv2 import undistortPoints
import numpy as np
from threading import current_thread

from .mapping.landmarks import initialize_tracked_landmarks

from .frontend.camera_calibration import (
    get_calibration_params,
    compute_parallax,
)
from .frontend import create_frontend
from .frontend.frame import create_frame
from .frontend.tracking import (
    create_lk_feature_detector,
    track_to_new_frame,
)
from .frontend.video import Video
from .multiview_geometry import (
    epipolar_pose,
    epipolar_ransac,
    triangulation,
)
from .frontend.pnp import pnp_ransac
from .utils.worker import create_thread_context
from .visualization.mapping import create_map_thread
from .visualization.tracking import create_drawer_thread

from .mapping.landmarks import initialize_tracked_landmarks

np.set_printoptions(precision=3, suppress=True)


def start(video_path, ground_truth_path):
    video = Video(video_path)
    video_stream = video.get_video_stream()
    tracked_frames = []

    K, Kinv, d = get_calibration_params()
    undistort = (
        lambda kp: undistortPoints(
            kp,
            K,
            d,
            R=np.eye(3),
            P=K,
        ).squeeze()
        if len(kp)
        else np.array([])
    )

    get_parallax = partial(compute_parallax, K, Kinv)
    frontend = create_frontend(
        create_lk_feature_detector(undistort),
        track_to_new_frame,
        partial(epipolar_ransac, K),
        partial(epipolar_pose, K),
        undistort,
        lambda *a: np.mean(get_parallax(*a)),
        partial(pnp_ransac, K),
    )

    thread_context = create_thread_context()
    send_draw_task = create_drawer_thread(thread_context, n_segments=5)
    send_map_task = create_map_thread(
        (800, 600),
        Kinv,
        thread_context,
    )

    def process_frame(triangulate_new_points, poses, tracked_frames, image):
        frame = create_frame(image, len(tracked_frames))
        frame = frontend(frame)
        tracked_frames += [frame]
        new_points = []
        if frame.is_keyframe:
            new_points = triangulate_new_points(frame)
        await_draw = send_draw_task(tracked_frames)
        await_map = send_map_task(
            tracked_frames,
            new_points,
            poses,
        )
        return lambda: [f() for f in [await_draw, await_map]]

    poses = []
    if ground_truth_path is not None:
        D = np.loadtxt(ground_truth_path).reshape(-1, 3, 4)
        for i in range(len(D)):
            T = np.vstack(
                [
                    D[i],
                    np.array([0, 0, 0, 1]),
                ]
            )
            poses += [T]

    process_frame = partial(
        process_frame,
        partial(
            initialize_tracked_landmarks,
            get_parallax,
            partial(triangulation, K),
            tracked_frames,
        ),
        poses,
        tracked_frames,
    )
    thread_context.start()
    for image in video_stream:
        wait_visualization = process_frame(image)
        # wait_visualization()
        if thread_context.is_closed:
            break
    thread_context.wait_close()
    thread_context.cleanup()
    thread_context.join_all()
    print(f"{current_thread()} exiting.")
