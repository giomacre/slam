import os
import sys
import cv2 as cv
import numpy as np
from ..mapping.point import create_point
from ..utils.slam_logging import log_feature_match, log_feature_extraction

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
    def orb_detector(frame, max_features=None, mask=None):
        if max_features is not None:
            orb.setMaxFeatures(max_features)
        key_pts = []
        if orb.getMaxFeatures() > 0:
            key_pts = detector.detect(frame.image, mask=mask)
            key_pts = sorted(
                key_pts,
                key=lambda x: x.response,
                reverse=True,
            )

        if len(key_pts) == 0 and orb.getMaxFeatures() > 0:
            return frame
        if len(key_pts) > 0:
            key_pts = ssc(
                key_pts,
                min(orb.getMaxFeatures(), len(key_pts)) + 1,
                0.1,
                frame.image.shape[1],
                frame.image.shape[0],
            )
            key_pts = cv.KeyPoint_convert(key_pts)
            undist = undistort(key_pts)
            observations = [
                create_point(frame, i)
                for i in range(
                    len(frame.observations),
                    len(frame.observations) + len(key_pts),
                )
            ]
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
                observations = [
                    *frame.observations,
                    *observations,
                ]
            frame.key_pts = key_pts
            frame.undist = undist
            frame.observations = observations
        frame.desc = orb.compute(
            frame.image,
            cv.KeyPoint_convert(frame.key_pts),
        )[1]
        return frame

    return orb_detector
