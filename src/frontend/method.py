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
        # Filter outliers with RANSAC
        frame.key_pts = tracked
        frame.undist = undistort(frame.key_pts)
        T, inliers = epipolar_ransac(
            frame.undist,
            last_frame.undist[train_idxs],
        )
        if T is None:
            print("Ess. Matrix failed")
        if inliers is not None:
            train_idxs = train_idxs[inliers]
            frame.key_pts = frame.key_pts[inliers]
            frame.undist = frame.undist[inliers]
            frame.pose = T @ last_frame.pose
        kf_idxs = np.array(
            [
                *(
                    last_frame.observations[i].idxs[current_keyframe.id]
                    for i in train_idxs
                )
            ]
        )
        return frame, kf_idxs

    def localization(frame, kf_idxs):
        # print(f"LOCALIZATION {frame.id=}")
        _, current_keyframe = current_context()
        pts_3d, idxs_3d = [[]] * 2
        values = tuple(
            np.array(a)
            for a in zip(
                *(
                    (current_keyframe.observations[kf_idx].coords, idx)
                    for idx, kf_idx in enumerate(kf_idxs)
                    if current_keyframe.observations[kf_idx].is_initialized
                )
            )
        )
        if len(values) > 0:
            pts_3d, idxs_3d = values
        if len(pts_3d) >= 4:
            # print("PnP RANSAC")
            T, mask = pnp_ransac(pts_3d, frame.undist[idxs_3d])
            # print(f"{T=}\n{np.count_nonzero(mask)=}\n{current_keyframe.pose=}")
            frame.key_pts, frame.undist, frame.desc
            if T is None:
                print("PnP tracking failed")
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
        else:
            print("Not enough landmarks for PnP")
        frame.observations = {}
        for i, kf_idx in enumerate(kf_idxs):
            landmark = current_keyframe.observations[kf_idx]
            landmark.idxs |= {frame.id: i}
            frame.observations[i] = landmark

        return frame, kf_idxs

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
        current_landmarks = [
            lm for lm in frame.observations.values() if lm.is_initialized
        ]
        kf_landmarks = [
            lm for lm in current_keyframe.observations.values() if lm.is_initialized
        ]
        tracked_lm_ratio = (
            len(current_landmarks) / len(kf_landmarks) if len(kf_landmarks) else 0
        )
        num_tracked = len(frame.key_pts)
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
        if (
            avg_parallax > kf_parallax_threshold
            or current_keyframe.id > 0
            and (
                num_tracked < min_features
                or avg_parallax > kf_parallax_threshold / 2.0
                and (
                    num_tracked < max_features / 2.0
                    and tracked_lm_ratio < kf_landmark_ratio + 0.1
                    or tracked_lm_ratio < kf_landmark_ratio
                )
            )
        ):
            n_ret, frame = detector(
                frame,
                max_features - num_tracked,
            )
            if n_ret == 0:
                context["last_frame"] = frame
                return frame
            frame.is_keyframe = True
            context["current_keyframe"] = frame
        context["last_frame"] = frame
        return frame

    def frontend(frame):
        if frame.id == 0:
            n_ret, frame = detector(frame, frontend_params["max_features"])
            if n_ret == 0:
                return frame
            frame.pose = np.eye(4)
            frame.is_keyframe = True
            context["current_keyframe"] = frame
            context["last_frame"] = frame
            return frame
        last_frame, _ = current_context()
        frame.pose = last_frame.pose
        frame, kf_idxs = match_features(frame)
        frame, kf_idxs = localization(frame, kf_idxs)
        return keyframe_recognition(frame, kf_idxs)

    return frontend
