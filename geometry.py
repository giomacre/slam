from functools import reduce
import numpy as np
import cv2 as cv
from camera_calibration import to_image_coords
from utils.decorators import ddict
from utils.slam_logging import log_pose_estimation, log_triangulation


def create_pose_estimator(K):
    no_pose = [None] * 2

    # @log_pose_estimation
    def compute_pose(matches):
        R, t, mask = get_relative_transform(K, matches)
        if R is None:
            return no_pose
        mask = mask.astype(np.bool).ravel()
        S = np.vstack(
            (np.hstack((R, t)), [0, 0, 0, 1]),
        )
        return S, mask

    return compute_pose


def get_relative_transform(K, points):
    E, mask = cv.findEssentialMat(
        points[..., 1],
        points[..., 0],
        K,
        threshold=1.0,
    )
    if E is None:
        return [None] * 3
    _, R, t, mask_pose = cv.recoverPose(
        E,
        points[..., 1],
        points[..., 0],
        K,
        mask=mask.copy(),
    )
    if np.sum(mask_pose) == 0:
        return [None] * 3
    R = R.T
    t = -R @ t
    return R, t, mask


def create_point_triangulator(K):
    # @log_triangulation
    K = np.hstack((K, np.array([0, 0, 0]).reshape(3, 1)))

    def triangulation(
        current_pose,
        reference_pose,
        current_points,
        reference_points,
    ):
        current_extrinsics = to_extrinsics(current_pose)
        reference_extrinsics = to_extrinsics(reference_pose)
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


def to_extrinsics(camera_pose):
    world_to_camera = np.vstack(
        (
            np.hstack(
                (
                    camera_pose[:3, :3].T,
                    -camera_pose[:3, :3].T @ camera_pose[:3, 3:],
                )
            ),
            [0, 0, 0, 1],
        ),
    )
    return world_to_camera
