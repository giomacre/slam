from collections import deque
from functools import partial
from typing import DefaultDict
from cv2 import undistortPoints
import numpy as np
from threading import current_thread
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
from .utils.geometry import (
    create_point_triangulator,
    epipolar_ransac,
    pnp_ransac,
)
from .utils.worker import create_thread_context
from .visualization.mapping import create_map_thread
from .visualization.tracking import create_drawer_thread

np.set_printoptions(precision=3, suppress=True)


def main():
    from argparse import ArgumentParser, ArgumentTypeError

    def check_file(path):
        from cv2 import VideoCapture

        video = VideoCapture(path)
        if video.isOpened():
            return path
        raise ArgumentTypeError(f"{path} is not a valid file.")

    parser = ArgumentParser()
    parser.add_argument(
        "video_path",
        type=check_file,
    )
    args = parser.parse_args()
    video = Video(args.video_path)
    video_stream = video.get_video_stream()

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

    detector = create_lk_orb_detector(undistort)
    tracker = track_to_new_frame
    epipolar_localizer = partial(epipolar_ransac, K)
    get_parallax = partial(compute_parallax, K, Kinv)
    frontend = create_frontend(
        detector,
        tracker,
        epipolar_localizer,
        undistort,
        lambda *a: np.mean(get_parallax(*a)),
        partial(pnp_ransac, K),
    )

    triangulation = create_point_triangulator(K)

    thread_context = create_thread_context()
    send_draw_task = create_drawer_thread(thread_context)
    send_map_task = create_map_thread(
        (800, 600),
        Kinv,
        thread_context,
    )

    thread_context.start()
    frames = video_stream
    tracked_frames = []
    map_points = []
    for image in frames:
        frame = create_frame(len(tracked_frames), image)
        frame = frontend(frame)
        if frame is None:
            continue
        tracked_frames += [frame]
        if frame.is_keyframe:
            candidate_pts = [
                (id, lm)
                for id, lm in frame.observations.items()
                if not lm.is_initialized and len(lm.idxs) > 1
            ]
            matches = DefaultDict(lambda: [[], []])
            for curr_idx, lm in candidate_pts:
                id, idx = next(x for x in lm.idxs.items())
                ref_idxs, curr_idxs = matches[id]
                ref_idxs += [idx]
                curr_idxs += [curr_idx]
            for kf_id, (ref_idxs, curr_idxs) in matches.items():
                ref_kf = tracked_frames[kf_id]
                ref_idxs = np.array(ref_idxs)
                curr_idxs = np.array(curr_idxs)
                parallax = get_parallax(
                    frame.pose,
                    ref_kf.pose,
                    frame.undist[curr_idxs],
                    ref_kf.undist[ref_idxs],
                )
                pts_3d, good_pts = triangulation(
                    frame.pose,
                    ref_kf.pose,
                    frame.undist[curr_idxs],
                    ref_kf.undist[ref_idxs],
                )
                old_kps = parallax > frontend_params["kf_parallax_threshold"]
                for i in sorted(
                    ref_idxs[old_kps & ~good_pts],
                    reverse=True,
                ):
                    del ref_kf.observations[i].idxs[kf_id]
                    del ref_kf.observations[i]
                map_points = []
                for i, pt in zip(curr_idxs[good_pts], pts_3d[good_pts]):
                    to_idx = lambda kp: tuple(np.rint(kp).astype(int)[::-1])
                    landmark = frame.observations[i]
                    landmark.coords = pt
                    img_idx = to_idx(frame.key_pts[i])
                    landmark.color = frame.image[img_idx] / 255.0
                    landmark.is_initialized = True
                    map_points += [landmark]

        await_draw = send_draw_task(tracked_frames)
        await_map = send_map_task(
            tracked_frames,
            [lm for lm in frame.observations.values() if lm.is_initialized],
        )
        if thread_context.is_closed:
            break
        # await_draw()
        # await_map()
    thread_context.wait_close()
    thread_context.cleanup()
    thread_context.join_all()
    print(f"{current_thread()} exiting.")


if __name__ == "__main__":
    from sys import argv

    main(argv)
