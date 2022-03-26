import numpy as np
import cv2 as cv
from utils.decorators import ddict
from utils.slam_logging import log_pose_estimation, log_triangulation


def create_pose_estimator(K):
    K = K[:3, :3]
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
                (K @ reference_extrinsics)[:3],
                (K @ current_extrinsics)[:3],
                reference_points.T,
                current_points.T,
            )
        ).T
        points_4d = points_4d[points_4d[:, -1] != 0]
        points_4d /= points_4d[:, -1:]
        camera_coordinates = current_extrinsics @ points_4d.T
        in_front = camera_coordinates[2, :] > 0.0
        return points_4d[in_front]

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
