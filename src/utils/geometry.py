from functools import reduce
import numpy as np
import cv2 as cv
from ..frontend.camera_calibration import to_image_coords
from .slam_logging import log_pose_estimation, log_triangulation


# @log_pose_estimation
def epipolar_ransac(K, query_pts, train_pts):
    E, mask = cv.findEssentialMat(
        train_pts,
        query_pts,
        K,
        prob=0.999,
    )
    if E is None:
        return [None] * 2
    # train_pts, query_pts = cv.correctMatches(
    #     E,
    #     train_pts[None, ...],
    #     query_pts[None, ...],
    # )
    _, R, t, mask_pose = cv.recoverPose(
        E,
        train_pts,
        query_pts,
        K,
        mask=mask.copy(),
    )
    if np.sum(mask_pose) == 0:
        return [None] * 2
    T = construct_pose(R.T, -R.T @ t * 0.25)
    mask = mask.astype(np.bool).ravel()
    return T, mask


def pnp_ransac(K, lm_coords, image_coords):
    retval, retvec, tvec, inliers = cv.solvePnPRansac(
        lm_coords,
        image_coords,
        K,
        distCoeffs=None,
    )
    if not retval:
        return [None] * 2
    R = cv.Rodrigues(retvec)[0].T
    T = construct_pose(R, -R @ tvec)
    mask = np.full(len(lm_coords), False)
    mask[inliers.flatten()] = True
    return T, mask


def create_point_triangulator(K):
    # @log_triangulation
    K = np.hstack((K, np.array([0, 0, 0]).reshape(3, 1)))

    def triangulation(
        current_pose,
        reference_pose,
        current_points,
        reference_points,
    ):
        current_extrinsics = np.linalg.inv(current_pose)
        reference_extrinsics = np.linalg.inv(reference_pose)
        points_4d = np.array(
            cv.triangulatePoints(
                (K @ reference_extrinsics),
                (K @ current_extrinsics),
                reference_points.T,
                current_points.T,
            )
        ).T
        points_4d = points_4d[points_4d[:, -1] != 0]
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
                np.linalg.norm(a - b.T, axis=0) < 5.0
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

    return triangulation


def construct_pose(R, t):
    return np.vstack(
        (np.hstack((R, t)), [0, 0, 0, 1]),
    )
