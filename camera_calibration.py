from params import camera_params
import numpy as np


def get_calibration_params():

    fx, fy, cx, cy = [camera_params[p] for p in ["fx", "fy", "cx", "cy"]]
    d = np.array([camera_params[p] for p in ["k1", "k2", "p1", "p2", "k3"]])
    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]
    )
    K_inv = np.linalg.inv(K)
    return K, K_inv, d


def to_homogeneous(x):
    return np.concatenate(
        (
            x,
            np.ones(shape=(*x.shape[:-2], 1, x.shape[-1])),
        ),
        axis=np.arange(len(x.shape))[-2],
    )


def to_camera_coords(K_inv, x):
    return K_inv @ to_homogeneous(x)


def to_image_coords(K, x):
    projected = K @ x
    scaled = projected / (
        projected.take(indices=-1, axis=np.arange(len(x.shape))[-2],).reshape(
            (
                *x.shape[:-2],
                1,
                x.shape[-1],
            )
        )
    )
    return scaled[..., :2, :]
