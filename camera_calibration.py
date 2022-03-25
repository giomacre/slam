from params import camera_params
import numpy as np
import cv2


def get_calibration_matrix(width, height):

    fx, fy, cx, cy = [
        camera_params[p]
        for p in [
            "fx",
            "fy",
            "cx",
            "cy",
        ]
    ]
    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]
    )
    d = np.array(
        [
            camera_params[p]
            for p in [
                "k1",
                "k2",
                "p1",
                "p2",
                "k3",
            ]
        ]
    )
    Knew, _ = cv2.getOptimalNewCameraMatrix(
        K,
        d,
        (width, height),
        camera_params.alpha,
        (width, height),
    )
    K_inv = np.linalg.inv(Knew)
    return Knew, K_inv
