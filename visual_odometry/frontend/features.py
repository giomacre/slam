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


def create_orb_detector(undistort, **orb_args):
    orb = cv.ORB_create(**orb_args)
    detector = cv.FastFeatureDetector_create()

    @log_feature_extraction
    def orb_detector(frame, max_features, mask=None):
        n_extracted = 0
        key_pts = []
        if max_features > 0:
            key_pts = detector.detect(frame.image, mask=mask)
            key_pts = sorted(
                key_pts,
                key=lambda x: x.response,
                reverse=True,
            )
        if len(key_pts) == 0 and max_features > 0:
            return 0, frame
        if len(key_pts) > 0:
            key_pts = ssc(
                key_pts,
                (min(max_features, len(key_pts)) + 1),
                0.15,
                frame.image.shape[1],
                frame.image.shape[0],
            )
            n_extracted = len(key_pts)
            key_pts = cv.KeyPoint_convert(key_pts)
            key_pts = cv.cornerSubPix(
                cv.cvtColor(frame.image, cv.COLOR_BGR2GRAY),
                key_pts,
                (3, 3),
                (-1, -1),
                (
                    cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,
                    300,
                    0.001,
                ),
            )
            undist = undistort(key_pts)
            start = (
                max(frame.observations.keys()) + 1 if len(frame.observations) > 0 else 0
            )
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
                observations |= frame.observations
            frame.key_pts = key_pts
            frame.undist = undist
            frame.observations = observations
        return n_extracted, frame

    return orb_detector
