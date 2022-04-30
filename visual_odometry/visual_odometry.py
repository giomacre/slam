from functools import lru_cache, partial
from cv2 import undistortPoints
import numpy as np
from threading import current_thread

from .mapping.landmarks import initialize_tracked_landmarks

from .frontend.camera_calibration import (
    get_calibration_params,
    compute_parallax,
)
from .utils.params import frontend_params
from .frontend.method import create_frontend
from .frontend.frame import create_frame
from .frontend.klt import (
    create_lk_orb_detector,
    track_to_new_frame,
)
from .frontend.video import Video
from .utils.multiview_geometry import (
    epipolar_ransac,
    pnp_ransac,
    triangulation,
)
from .utils.worker import create_thread_context
from .visualization.mapping import create_map_thread
from .visualization.tracking import create_drawer_thread

from .mapping.landmarks import initialize_tracked_landmarks

np.set_printoptions(precision=3, suppress=True)


def start(video_path):
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
        tracked_frames,
        create_lk_orb_detector(undistort),
        track_to_new_frame,
        partial(epipolar_ransac, K),
        undistort,
        lambda *a: np.mean(get_parallax(*a)),
        partial(pnp_ransac, K),
    )

    thread_context = create_thread_context()
    send_draw_task = create_drawer_thread(thread_context)
    send_map_task = create_map_thread(
        (800, 600),
        Kinv,
        thread_context,
    )

    def process_frame(triangulate_new_points, tracked_frames, image):
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
        )
        return lambda: [f() for f in [await_draw, await_map]]

    process_frame = partial(
        process_frame,
        partial(
            initialize_tracked_landmarks,
            partial(
                triangulation,
                K,
                # lru_cache(maxsize=1)(
                #     lambda: np.hstack(
                #         (
                #             K,
                #             np.array([0, 0, 0]).reshape(3, 1),
                #         )
                #     )
                # ),
            ),
            tracked_frames,
        ),
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
