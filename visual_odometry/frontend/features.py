import os
import sys
import cv2 as cv
import numpy as np
from ..mapping.landmarks import create_landmark
from ..utils.slam_logging import log_feature_match, log_feature_extraction
from ..utils.params import frontend_params

sys.path.append(
    os.path.join(
        os.path.dirname(sys.argv[0]),
        "external",
        "ANMS-Codes",
        "Python",
    )
)
from ssc import ssc

# Detectors


def create_feature_detector(undistort):
    detector = cv.FastFeatureDetector_create(
        threshold=frontend_params["fast_threshold"],
    ).detect

    @log_feature_extraction
    def feature_detector(frame, max_features, mask=None):
        n_extracted = 0
        key_pts = []
        if max_features > 0:
            gray_image = cv.cvtColor(frame.image, cv.COLOR_BGR2GRAY)
            key_pts = detector(gray_image, mask)
            key_pts = sorted(
                key_pts,
                key=lambda x: x.response,
                reverse=True,
            )
        n_extracted = len(key_pts)
        if n_extracted == 0 and max_features > 0:
            return 0, frame
        if n_extracted > 0:
            key_pts = ssc(
                key_pts,
                min(max_features, len(key_pts)) + 1,
                0.15,
                frame.image.shape[1],
                frame.image.shape[0],
            )
            key_pts = cv.KeyPoint_convert(key_pts)
            n_extracted = len(key_pts)
            key_pts = cv.cornerSubPix(
                gray_image,
                key_pts,
                winSize=(3, 3),
                zeroZone=(-1, -1),
                criteria=(
                    cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,
                    frontend_params["klt_max_iter"],
                    frontend_params["klt_convergence_threshold"],
                ),
            )
            undist = undistort(key_pts)
            start = max(frame.landmarks.keys()) + 1 if len(frame.landmarks) > 0 else 0
            observations = {
                i: create_landmark(frame, i)
                for i in range(
                    start,
                    start + len(key_pts),
                )
            }
            if len(frame.key_pts > 0):
                key_pts = np.vstack(
                    [
                        frame.key_pts,
                        key_pts,
                    ]
                )
                undist = np.vstack(
                    [
                        frame.undist,
                        undist,
                    ]
                )
                observations |= frame.landmarks
            frame.key_pts = key_pts
            frame.undist = undist
            frame.landmarks = observations
        return n_extracted, frame

    return feature_detector
