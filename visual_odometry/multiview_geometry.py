from functools import reduce
import numpy as np
import cv2 as cv

from .frontend.camera_calibration import to_image_coords
from .utils.slam_logging import log_pose_estimation, log_triangulation
from .utils.params import ransac_params


# @log_pose_estimation
def epipolar_ransac(K, query_pts, train_pts):
    if len(train_pts) < 5:
        return [None] * 2
    E, mask = cv.findEssentialMat(
        train_pts,
        query_pts,
        K,
        prob=ransac_params["em_confidence"],
        threshold=ransac_params["em_threshold"],
    )
    n_inliers = np.count_nonzero(mask)
    if n_inliers < 0.5 * len(query_pts):
        return [None] * 2
    retval, R, t, _ = cv.recoverPose(
        E,
        train_pts,
        query_pts,
        K,
        mask=mask.copy(),
    )
    if retval:
        T = construct_pose(R.T, -R.T @ t)
    else:
        T = None
    mask = mask.astype(np.bool).ravel()
    return T, mask


# @log_triangulation
def triangulation(
    K,
    current_pose,
    reference_pose,
    current_points,
    reference_points,
):
    current_extrinsics = np.linalg.inv(current_pose)[:3]
    reference_extrinsics = np.linalg.inv(reference_pose)[:3]
    points_4d = np.array(
        cv.triangulatePoints(
            (K @ reference_extrinsics),
            (K @ current_extrinsics),
            reference_points.T,
            current_points.T,
        )
    ).T
    points_4d /= points_4d[:, -1:]
    camera_coords = np.dstack(
        [
            (extrinsics @ points_4d.T).T
            for extrinsics in [
                current_extrinsics,
                reference_extrinsics,
            ]
        ]
    ).T
    projected = to_image_coords(K, camera_coords)
    low_err = reduce(
        np.bitwise_and,
        (
            np.linalg.norm(a - b.T, axis=0) < ransac_params["p3p_threshold"]
            for a, b in zip(
                projected,
                [current_points, reference_points],
            )
        ),
    )
    in_front = reduce(
        np.bitwise_and,
        (pts[2, :] > 0 for pts in camera_coords),
    ).T
    good_pts = in_front & low_err
    return points_4d[..., :3], good_pts


def construct_pose(R, t):
    return np.vstack(
        (np.hstack((R, t)), [0, 0, 0, 1]),
    )
