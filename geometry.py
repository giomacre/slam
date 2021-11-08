import numpy as np
import cv2 as cv
from decorators import ddict
from slam_logging import log_pose_estimation


def create_pose_estimator(K, detector, matcher):
    K = K[:3, :3]

    @log_pose_estimation
    def compute_pose(frame):
        no_pose = None, []
        key_pts, desc = detector(frame.image)
        frame |= ddict(
            key_pts=key_pts,
            desc=desc,
        )
        matches = matcher(frame)
        if len(matches) == 0:
            return no_pose
        R, t, mask = get_pose_from_image_points(K, matches)
        if R is None:
            return no_pose
        return (
            np.vstack(
                (
                    np.hstack((R, t)),
                    [0, 0, 0, 1],
                )
            ),
            matches[mask.astype(np.bool).ravel()],
        )

    return compute_pose


def get_pose_from_image_points(K, points):
    E, mask = cv.findEssentialMat(
        points[..., 0],
        points[..., 1],
        K,
    )
    if E is None:
        return [None] * 3
    _, R, t, mask = cv.recoverPose(
        E,
        points[..., 0],
        points[..., 1],
        K,
        mask=mask,
    )
    return R, t, mask
