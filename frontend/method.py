from params import frontend_params
import numpy as np
from utils.decorators import ddict


def create_frontend(
    detector,
    tracker,
    epipolar_ransac,
):
    context = ddict()
    context.last_frame = None
    context.current_keyframe = None

    def match_features(frame):
        matches, train_idxs = tracker(
            frame,
            context.last_frame,
        )
        if len(matches) == 0:
            return [None] * 2
        # Filter outliers with RANSAC
        _, inliers = epipolar_ransac(matches)
        if inliers is None:
            return [None] * 2
        train_idxs = train_idxs[inliers]
        frame.key_pts = matches[inliers, ..., 0]
        return frame, train_idxs

    def localization(frame, train_idxs):
        last_frame, current_keyframe = (
            context.last_frame,
            context.current_keyframe,
        )
        kf_idxs = np.array(
            [last_frame.observations[i].idxs[current_keyframe.id] for i in train_idxs]
        )
        S, inliers = epipolar_ransac(
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
        return frame

    def keyframe_recognition(frame):
        num_tracked = len(frame.key_pts)
        if (
            num_tracked / len(context.current_keyframe.key_pts)
            < frontend_params.kf_threshold
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
