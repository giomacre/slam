import cv2
from frontend.optical_flow import create_lk_orb_detector, create_lk_tracker
from geometry import create_pose_estimator
from params import frontend_params
import numpy as np

from utils.decorators import ddict


def create_localizer(detector, tracker, pose_estimator):
    context = ddict
    context.last_frame = None
    context.current_keyframe=None

    def localization(frame):
        if frame.id == 0:
            frame = detector(frame, frontend_params.n_features)
            if len(frame.key_pts) == 0:
                return None
            frame.pose = np.eye(4)
            frame.is_keyframe = True
            context.current_keyframe = frame
            context.last_frame = frame
            return frame
        matches, query_idxs, train_idxs = tracker(
            frame,
            context.last_frame,
        )
        if len(matches) == 0:
            return None
        # Filter outliers with RANSAC
        S, inliers = pose_estimator(matches)
        if S is None:
            return None
        train_idxs = train_idxs[inliers]
        frame.key_pts = matches[inliers, ..., 0]
        # Compute the transform with respect to the last keyframe
        kf_idxs = np.array(
            [
                context.last_frame.observations[i].idxs[context.current_keyframe.id]
                for i in train_idxs
            ]
        )
        S, inliers = pose_estimator(
            np.dstack(
                (
                    frame.key_pts,
                    context.current_keyframe.key_pts[kf_idxs],
                )
            )
        )
        if S is None:
            return None
        num_tracked = sum(inliers)
        kf_idxs = kf_idxs[inliers]
        frame.observations = [None] * num_tracked
        for i in range(num_tracked):
            landmark = context.current_keyframe.observations[kf_idxs[i]]
            landmark.idxs |= {frame.id: i}
            frame.observations[i] = landmark
        frame.pose = S @ context.current_keyframe.pose
        frame.key_pts = frame.key_pts[inliers]
        if num_tracked / len(context.current_keyframe.key_pts) < frontend_params.kf_threshold:
            frame.is_keyframe = True
            context.current_keyframe = frame
            frame = detector(
                frame,
                frontend_params.n_features - num_tracked,
            )
            if len(frame.key_pts) == num_tracked:
                return None
        context.last_frame = frame
        return frame

    return localization
