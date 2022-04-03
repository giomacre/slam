import enum
from params import frontend_params
import numpy as np
from utils.decorators import ddict


def create_frontend(
    detector,
    tracker,
    epipolar_ransac,
    undistort,
):
    context = ddict()
    context.last_frame = None
    context.current_keyframe = None

    def match_features(frame):
        tracked, train_idxs = tracker(
            frame,
            context.last_frame,
        )
        if len(tracked) == 0:
            return [None] * 2
        # Filter outliers with RANSAC
        frame.key_pts = tracked
        frame.undist = undistort(frame.key_pts)
        _, inliers = epipolar_ransac(
            frame.undist,
            context.last_frame.undist[train_idxs],
        )
        if inliers is None:
            return [None] * 2
        train_idxs = train_idxs[inliers]
        frame.key_pts = frame.key_pts[inliers]
        frame.undist = frame.undist[inliers]
        return frame, train_idxs

    def localization(frame, train_idxs):
        last_frame, current_keyframe = (
            context.last_frame,
            context.current_keyframe,
        )
        kf_idxs = np.array(
            [last_frame.observations[i].idxs[current_keyframe.id] for i in train_idxs]
        )
        frame.observations = [None] * len(kf_idxs)
        for i, kf_idx in enumerate(kf_idxs):
            landmark = current_keyframe.observations[kf_idx]
            landmark.idxs |= {frame.id: i}
            frame.observations[i] = landmark
        S, _ = epipolar_ransac(
            frame.undist,
            current_keyframe.undist[kf_idxs],
        )
        if S is None:
            return None
        frame.pose = S @ context.current_keyframe.pose
        return frame

    def keyframe_recognition(frame):
        current_landmarks = [lm for lm in frame.observations if lm.is_initialized]
        kf_landmarks = [
            lm for lm in context.current_keyframe.observations if lm.is_initialized
        ]
        num_tracked = len(frame.key_pts)
        if (
            context.current_keyframe.id == 0
            and (
                num_tracked / len(context.current_keyframe.key_pts)
                < frontend_params.kf_threshold
            )
            or (
                context.current_keyframe.id > 0
                and len(current_landmarks) / len(kf_landmarks)
                < frontend_params.kf_threshold
            )
        ):
            frame = detector(
                frame,
                frontend_params.n_features - num_tracked,
            )
            if len(frame.key_pts) == num_tracked:
                return None
            frame.is_keyframe = True
            context.last_frame = frame
            context.current_keyframe = frame
        context.last_frame = frame
        return frame

    def frontend(frame):
        if frame.id == 0:
            frame = detector(frame, frontend_params.n_features)
            if len(frame.key_pts) == 0:
                return None
            frame.pose = np.eye(4)
            frame.is_keyframe = True
            context.current_keyframe = frame
            context.last_frame = frame
            return frame
        frame, train_idxs = match_features(frame)
        if frame is None:
            return None
        frame = localization(frame, train_idxs)
        if frame is None:
            return None
        return keyframe_recognition(frame)

    return frontend
