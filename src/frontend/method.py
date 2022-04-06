from functools import partial
from operator import itemgetter
from ..utils.params import frontend_params
import numpy as np


def create_frontend(
    detector,
    tracker,
    epipolar_ransac,
    undistort,
    average_parallax,
    pnp_ransac,
):
    context = dict(
        last_frame=None,
        current_keyframe=None,
    )
    current_context = partial(
        itemgetter("last_frame", "current_keyframe"),
        context,
    )

    def match_features(frame):
        last_frame, current_keyframe = current_context()
        tracked, train_idxs = tracker(
            frame,
            last_frame,
        )
        if len(tracked) == 0:
            return [None] * 3
        # Filter outliers with RANSAC
        frame.key_pts = tracked
        frame.undist = undistort(frame.key_pts)
        _, inliers = epipolar_ransac(
            frame.undist,
            last_frame.undist[train_idxs],
        )
        if inliers is None:
            return [None] * 3
        train_idxs = train_idxs[inliers]
        frame.key_pts = frame.key_pts[inliers]
        frame.undist = frame.undist[inliers]
        kf_idxs = np.array(
            [
                *(
                    last_frame.observations[i].idxs[current_keyframe.id]
                    for i in train_idxs
                )
            ]
        )
        observations = [current_keyframe.observations[i] for i in kf_idxs]
        return frame, kf_idxs, observations

    def localization(frame, kf_idxs, observations):
        _, current_keyframe = current_context()
        pts_3d, idxs_3d = [[]] * 2
        values = tuple(
            np.array(a)
            for a in zip(
                *(
                    (lm.coords, idx)
                    for idx, lm in enumerate(observations)
                    if lm.is_initialized
                )
            )
        )
        if len(values) > 0:
            pts_3d, idxs_3d = values
        if len(pts_3d) < 4:
            T, inliers = epipolar_ransac(
                frame.undist,
                current_keyframe.undist[kf_idxs],
            )
            if T is None:
                return [None] * 2
            frame.pose = T @ current_keyframe.pose
        else:
            T, mask = pnp_ransac(pts_3d, frame.undist[idxs_3d])
            if T is None:
                return [None] * 2
            outliers = idxs_3d[~mask]
            inliers = np.full(len(frame.key_pts), True)
            inliers[outliers] = False
            frame.pose = T
        frame.key_pts = frame.key_pts[inliers]
        frame.undist = frame.undist[inliers]
        kf_idxs = kf_idxs[inliers]
        frame.observations = [None] * len(kf_idxs)
        for i, kf_idx in enumerate(kf_idxs):
            landmark = current_keyframe.observations[kf_idx]
            landmark.idxs |= {frame.id: i}
            frame.observations[i] = landmark

        return frame, kf_idxs

    def keyframe_recognition(frame, kf_idxs):
        _, current_keyframe = current_context()
        avg_parallax = average_parallax(
            frame.pose,
            current_keyframe.pose,
            frame.undist,
            current_keyframe.undist[kf_idxs],
        )
        print(avg_parallax)
        current_landmarks = [
            *(lm for lm in frame.observations if lm.is_initialized),
        ]
        kf_landmarks = [
            *(lm for lm in current_keyframe.observations if lm.is_initialized),
        ]
        num_tracked = len(frame.key_pts)
        if (
            avg_parallax > frontend_params["kf_avg_parallax"]
            or current_keyframe.id > 0
            and avg_parallax > frontend_params["kf_avg_parallax"] / 2.0
            and num_tracked < frontend_params["n_features"] / 2.0
            and len(current_landmarks) / len(kf_landmarks)
            < frontend_params["kf_point_ratio"]
        ):
            frame = detector(
                frame,
                frontend_params["n_features"] - num_tracked,
            )
            if len(frame.key_pts) == num_tracked:
                return None
            frame.is_keyframe = True
            context["current_keyframe"] = frame
        context["last_frame"] = frame
        return frame

    def frontend(frame):
        if frame.id == 0:
            frame = detector(frame, frontend_params["n_features"])
            if len(frame.key_pts) == 0:
                return None
            frame.pose = np.eye(4)
            frame.is_keyframe = True
            context["current_keyframe"] = frame
            context["last_frame"] = frame
            return frame
        frame, kf_idxs, observations = match_features(frame)
        if frame is None:
            return None
        frame, kf_idxs = localization(frame, kf_idxs, observations)
        if frame is None:
            return None
        return keyframe_recognition(frame, kf_idxs)

    return frontend
