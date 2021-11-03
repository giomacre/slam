import numpy as np
import cv2 as cv
from decorators import performance_timer


def create_pose_estimator(K, detector, matcher):
    K = K[:3, :3]
    K_inv = np.linalg.inv(K)

    @performance_timer
    def compute_pose(frame):
        key_pts, desc = detector(frame["image"])
        no_pose = {
            "T": None,
            "matches": None,
        }
        frame |= {
            "key_pts": key_pts,
            "desc": desc,
        }
        matches = matcher(frame)
        if matches is None:
            return no_pose
        camera_coords = to_camera_coords(K_inv, matches)
        R, t, mask = get_pose_from_camera_points(K, camera_coords)
        if R is None:
            return no_pose
        return {
            "T": np.vstack(
                (
                    np.hstack((R, t)),
                    [0, 0, 0, 1],
                )
            ),
            "matches": matches[mask.astype(np.bool).ravel()],
        }

    return compute_pose


@performance_timer
def get_pose_from_camera_points(K, points):
    cv_view = points[:, :2]
    E, mask = cv.findEssentialMat(cv_view[..., 0], cv_view[..., 1], K)
    if E is None:
        return [None] * 3
    _, R, t, mask = cv.recoverPose(
        E,
        cv_view[..., 0],
        cv_view[..., 1],
        K,
        mask=mask,
    )
    return R, t, mask


def to_homogeneous(x):
    return np.concatenate(
        (
            x,
            np.ones(shape=(x.shape[0], 1, x.shape[2])),
        ),
        axis=1,
    )


def to_camera_coords(K_inv, x):
    return K_inv @ to_homogeneous(x)


def to_image_coords(K, x):
    return (K @ x)[:, :2, :]
