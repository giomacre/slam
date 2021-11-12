import numpy as np
import cv2 as cv
from decorators import ddict
from slam_logging import log_pose_estimation, log_triangulation


def create_pose_estimator(K, detector, matcher):
    K = K[:3, :3]
    no_reference = ddict(
        image=None,
        key_pts=None,
        desc=None,
    )

    @log_pose_estimation
    def compute_pose(query_frame, train_frame=no_reference):
        no_value = []
        query_frame = detector(query_frame)
        matches, query_idxs, train_idxs = matcher(query_frame, train_frame)
        if len(matches) == 0:
            return no_value
        R, t, mask = get_pose_from_image_points(K, matches)
        mask = mask.astype(np.bool).ravel()
        if R is None:
            return no_value
        T = np.vstack(
            (np.hstack((R, t)), [0, 0, 0, 1]),
        )
        query_idxs = query_idxs[mask]
        train_idxs = train_idxs[mask]
        query_frame.pose = T @ train_frame.pose
        query_frame.origin_frames[query_idxs] = train_frame.origin_frames[train_idxs]
        query_frame.origin_pts[query_idxs] = train_frame.origin_pts[train_idxs]
        query_frame.tracked_idxs = query_idxs
        return matches[mask]

    return compute_pose


def get_pose_from_image_points(K, points):
    E, mask = cv.findEssentialMat(
        points[..., 0],
        points[..., 1],
        K,
        threshold=1.0,
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

def create_point_triangulator(K):
    @log_triangulation
    def triangulation(
        current_pose,
        reference_pose,
        matches,
    ):
        points_4d = np.array(
            cv.triangulatePoints(
                (K @ current_pose)[:3],
                (K @ reference_pose)[:3],
                matches[..., 0].T,
                matches[..., 1].T,
            )
        ).T
        points_4d /= points_4d[:, -1:]
        return points_4d

    return triangulation
