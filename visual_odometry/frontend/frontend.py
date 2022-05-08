from collections import deque
from functools import partial
from operator import itemgetter

from ..utils.params import frontend_params
import numpy as np


def create_frontend(
    detector,
    tracker,
    epipolar_ransac,
    epipolar_pose,
    undistort,
    average_parallax,
    pnp_pose,
):

    (
        kf_parallax_threshold,
        max_features,
        min_features,
        kf_landmark_ratio,
    ) = itemgetter(
        "kf_parallax_threshold",
        "max_features",
        "min_features",
        "kf_landmark_ratio",
    )(
        frontend_params
    )

    context = dict(
        tracked_frames=deque([None, None], maxlen=2),
        current_keyframe=None,
    )
    current_context = partial(
        itemgetter("tracked_frames", "current_keyframe"),
        context,
    )

    def match_features(frame):
        (frame_t1, _), current_keyframe = current_context()
        tracked, train_idxs = tracker(
            frame,
            frame_t1,
        )
        # Filter outliers with RANSAC
        frame.key_pts = tracked
        frame.undist = undistort(frame.key_pts)
        kf_idxs = np.array(
            [
                frame_t1.landmarks[i].observations[current_keyframe.id]
                for i in train_idxs
            ]
        )
        E_kc, inliers = epipolar_ransac(
            frame.undist,
            current_keyframe.undist[kf_idxs],
        )
        if inliers is None:
            print("RANSAC filtering failed")
        if inliers is not None:
            kf_idxs = kf_idxs[inliers]
            frame.key_pts = frame.key_pts[inliers]
            frame.undist = frame.undist[inliers]
        return frame, kf_idxs, E_kc

    def localization(frame, kf_idxs, E_kc):
        (frame_t1, frame_t2), current_keyframe = current_context()

        def adjust_scale(T_kc):
            T_kl = np.linalg.inv(frame_t1.pose) @ current_keyframe.pose
            kf_scale = np.linalg.norm(T_kl[:3, 3])
            last_scale = (
                np.linalg.norm((np.linalg.inv(frame_t2.pose) @ frame_t1.pose)[:3, 3])
                if frame.id > 1
                else frontend_params["epipolar_scale"]
            )
            if T_kc is None:
                T_kc = T_kl
                T_kc[:3, 3] /= kf_scale
            T_kc[:3, 3] *= kf_scale + last_scale
            return T_kc

        values = tuple(
            np.array(a)
            for a in zip(
                *(
                    (current_keyframe.landmarks[kf_idx].coords, idx)
                    for idx, kf_idx in enumerate(kf_idxs)
                    if current_keyframe.landmarks[kf_idx].is_initialized
                )
            )
        )
        pts_3d, idxs_3d = values if len(values) > 0 else [[]] * 2
        if len(pts_3d) < 4:
            print("Not enough landmarks for PnP")
            T_kc = epipolar_pose(
                E_kc,
                frame.undist,
                current_keyframe.undist[kf_idxs],
            )
            T_kc = adjust_scale(T_kc)
            frame.pose = current_keyframe.pose @ T_kc
            return frame, kf_idxs
        T, mask = pnp_pose(pts_3d, frame.undist[idxs_3d])
        if T is None:
            print("PnP tracking failed")
            T_kc = epipolar_pose(
                E_kc,
                frame.undist,
                current_keyframe.undist[kf_idxs],
            )
            T_kc = adjust_scale(T_kc)
            frame.pose = current_keyframe.pose @ T_kc
            (
                frame.key_pts,
                frame.undist,
                frame.desc,
            ) = [np.array([])] * 3
            kf_idxs = np.array([])
            return frame, kf_idxs
        outliers = idxs_3d[~mask]
        inliers = np.full(len(frame.key_pts), True)
        inliers[outliers] = False
        frame.pose = T
        frame.key_pts = frame.key_pts[inliers]
        frame.undist = frame.undist[inliers]
        kf_idxs = kf_idxs[inliers]
        return frame, kf_idxs

    def transfer_observations(frame, kf_idxs):
        _, current_keyframe = current_context()
        frame.landmarks = {}
        for i, kf_idx in enumerate(kf_idxs):
            landmark = current_keyframe.landmarks[kf_idx]
            landmark.observations |= {frame.id: i}
            frame.landmarks[i] = landmark
        return frame

    def keyframe_recognition(frame, kf_idxs):
        _, current_keyframe = current_context()
        avg_parallax = (
            average_parallax(
                frame.pose,
                current_keyframe.pose,
                frame.undist,
                current_keyframe.undist[kf_idxs],
            )
            if len(frame.undist) > 0
            else 0
        )
        current_landmarks = [lm for lm in frame.landmarks.values() if lm.is_initialized]
        kf_landmarks = [
            lm for lm in current_keyframe.landmarks.values() if lm.is_initialized
        ]
        tracked_lm_ratio = (
            len(current_landmarks) / len(kf_landmarks) if len(kf_landmarks) else 0
        )
        num_tracked = len(frame.key_pts)
        if (
            avg_parallax > kf_parallax_threshold
            or num_tracked < min_features
            or current_keyframe.id > 0
            and (
                len(current_landmarks) < min_features * 0.66
                or avg_parallax > kf_parallax_threshold / 2.0
                and tracked_lm_ratio < kf_landmark_ratio
            )
        ):
            n_ret, frame = detector(
                frame,
                max_features - num_tracked,
            )
            if n_ret == 0:
                context["tracked_frames"].appendleft(frame)
                return frame
            frame.is_keyframe = True
            context["current_keyframe"] = frame
        context["tracked_frames"].appendleft(frame)
        return frame

    def frontend(frame):
        if frame.id == 0:
            n_ret, frame = detector(
                frame,
                frontend_params["max_features"],
            )
            if n_ret == 0:
                return frame
            frame.pose = np.eye(4)
            frame.is_keyframe = True
            context["current_keyframe"] = frame
            context["tracked_frames"].appendleft(frame)
            return frame
        frame, kf_idxs, E_kc = match_features(frame)
        frame, kf_idxs = localization(frame, kf_idxs, E_kc)
        frame = transfer_observations(frame, kf_idxs)
        return keyframe_recognition(frame, kf_idxs)

    return frontend
