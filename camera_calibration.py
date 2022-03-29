from params import camera_params
import numpy as np
import cv2


def get_calibration_matrix(width, height):

    fx, fy, cx, cy = [camera_params[p] for p in ["fx", "fy", "cx", "cy"]]
    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]
    )
    d = np.array([camera_params[p] for p in ["k1", "k2", "p1", "p2", "k3"]])
    Knew, _ = cv2.getOptimalNewCameraMatrix(
        K,
        d,
        (width, height),
        camera_params.alpha,
        (width, height),
    )
    K_inv = np.linalg.inv(Knew)
    return Knew, K_inv


def to_homogeneous(x):
    return np.concatenate(
        (
            x,
            np.ones(shape=(x.shape[0], 1, *x.shape[2:])),
        ),
        axis=1,
    )


def to_camera_coords(K_inv, x):
    return (K_inv @ to_homogeneous(x).T).T


def to_image_coords(K, x):
    projected = (K @ x.T).T
    scaled = projected / projected[:, -1, ..., None]
    return scaled[:, :2, ...]
