import numpy as np
import cv2 as cv


def create_orb_pose_estimator(K, matcher):
    orb = cv.ORB_create(
        nfeatures=1500,
        WTA_K=4,
    )
    return PoseEstimator(
        K,
        lambda f: orb.detectAndCompute(f, None),
        matcher,
    )


class PoseEstimator:
    def __init__(self, K, feature_extractor, feature_matcher):
        self.__K__ = K
        self.__Kinv__ = np.linalg.inv(K)
        self.__extractor__ = feature_extractor
        self.__matcher__ = feature_matcher
        self.__frame_count__ = 0

    def compute_pose(self, frame):
        self.__frame_count__ += 1
        key_pts, desc = self.__extractor__(frame)
        if desc is None:
            return [self.__frame_count__] + [None] * 3
        frame = {
            "frame_id": self.__frame_count__,
            "image": frame,
            "key_pts": key_pts,
            "desc": desc,
        }
        matches = self.__matcher__(frame)
        if matches is None:
            return [self.__frame_count__] + [None] * 3
        matches = to_homogeneous(matches)
        normalized = self.__Kinv__ @ matches
        cv_view = normalized[:, :2]
        E, mask = cv.findEssentialMat(
            cv_view[..., 0],
            cv_view[..., 1],
            self.__K__,
        )
        if E is None:
            return [self.__frame_count__] + [None] * 3
        _, R, t, mask = cv.recoverPose(
            E,
            cv_view[..., 0],
            cv_view[..., 1],
            self.__K__,
            mask=mask,
        )
        return self.__frame_count__, R, t, normalized[mask.astype(np.bool).ravel()]


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
