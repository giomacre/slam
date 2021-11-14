import numpy as np
import cv2 as cv
from decorators import ddict
from slam_logging import log_pose_estimation, log_triangulation


def create_pose_estimator(K, detector, matcher):
    K = K[:3, :3]
    no_reference = ddict(
        image=None,
        key_pts=np.array([]),
        desc=None,
    )
    print(no_reference)

    @log_pose_estimation
    def compute_pose(query_frame, train_frame=no_reference):
        query_frame = detector(query_frame)
        matches, query_idxs, train_idxs = matcher(query_frame, train_frame)
        if len(matches) == 0:
            return 0
        R, t, *masks = get_pose_from_image_points(K, matches)
        mask, mask_pose = (m.astype(np.bool).ravel() for m in masks)

        if R is None:
            return 0
        T = np.vstack(
            (np.hstack((R, t)), [0, 0, 0, 1]),
        )
        query_idxs = query_idxs[mask]
        train_idxs = train_idxs[mask]
        query_frame.pose = T @ train_frame.pose
        query_frame.origin_frames[query_idxs] = train_frame.origin_frames[train_idxs]
        query_frame.origin_pts[query_idxs] = train_frame.origin_pts[train_idxs]
        query_frame.tracked_idxs = query_idxs
        return np.sum(mask_pose)

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
    _, R, t, mask_pose = cv.recoverPose(
        E,
        points[..., 0],
        points[..., 1],
        K,
        mask=mask.copy(),
    )
    return R, t, mask, mask_pose


def create_point_triangulator(K):
    @log_triangulation
    def triangulation(
        current_pose,
        reference_pose,
        current_points,
        reference_points,
    ):
        points_4d = np.array(
            cv.triangulatePoints(
                (K @ current_pose)[:3],
                (K @ reference_pose)[:3],
                current_points.T,
                reference_points.T,
            )
        ).T
        points_4d = points_4d[points_4d[:, -1] != 0]
        points_4d /= points_4d[:, -1:]
        camera_coordinates = current_pose @ points_4d.T
        in_front = camera_coordinates[2, :] > 0.0
        return points_4d[in_front]

    return triangulation
