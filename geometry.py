import numpy as np
import cv2 as cv
from decorators import performance_timer


def create_pose_estimator(K, detector, matcher):
    return PoseEstimator(
        K,
        detector,
        matcher,
    )


class PoseEstimator:
    def __init__(self, K, feature_extractor, feature_matcher):
        self.__K__ = K[:3, :3]
        self.__Kinv__ = np.linalg.inv(self.__K__)
        self.__extractor__ = feature_extractor
        self.__matcher__ = feature_matcher

    def compute_pose(self, frame):
        key_pts, desc = self.__extractor__(frame["image"])
        no_pose = {
            "T": None,
            "matches": None,
        }
        frame |= {
            "key_pts": key_pts,
            "desc": desc,
        }
        matches = self.__matcher__(frame)
        if matches is None:
            return no_pose
        homogeneous = to_homogeneous(matches)
        normalized = self.__Kinv__ @ homogeneous
        cv_view = normalized[:, :2]
        E, mask = cv.findEssentialMat(
            cv_view[..., 0],
            cv_view[..., 1],
            self.__K__,
        )
        if E is None:
            return no_pose
        _, R, t, mask = cv.recoverPose(
            E,
            cv_view[..., 0],
            cv_view[..., 1],
            self.__K__,
            mask=mask,
        )
        return {
            "T": np.vstack(
                (
                    np.hstack((R, t)),
                    [0, 0, 0, 1],
                )
            ),
            "matches": matches[mask.astype(np.bool).ravel()],
        }


def to_homogeneous(x):
    return np.concatenate(
        (
            x,
            np.ones(shape=(x.shape[0], 1, x.shape[2])),
        ),
        axis=1,
    )


def unnormalize(K, x):
    return (K @ x)[:, :2, :]
