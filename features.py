import cv2 as cv
import numpy as np
from decorators import stateful_decorator
from slam_logging import log_feature_match, log_feature_extraction

# Detectors


def create_orb_detector():
    orb = cv.ORB_create(
        nfeatures=3000,
        WTA_K=2,
        scoreType=cv.ORB_HARRIS_SCORE,
    )

    @log_feature_extraction
    def orb_detector(frame):
        return orb.detectAndCompute(frame, None)

    return orb_detector


# Matchers


def create_orb_flann_matcher():
    FLANN_INDEX_LSH = 6
    INDEX_PARAMS = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=8,
        key_size=20,
        multi_probe_level=2,
    )

    cv_matcher = cv.FlannBasedMatcher(INDEX_PARAMS)
    return create_feature_matcher(
        lambda d1, d2: cv_matcher.knnMatch(d1, d2, k=2),
        ratio_test_filter(),
    )


def create_bruteforce_matcher():
    cv_matcher = cv.BFMatcher_create(cv.NORM_HAMMING)
    return create_feature_matcher(
        lambda d1, d2: cv_matcher.knnMatch(d1, d2, k=2),
        ratio_test_filter(),
    )


def ratio_test_filter(thresh_value=0.7):
    def filter(matches):
        if len(matches) < 2:
            return False
        m1, m2 = matches
        return m1.distance < thresh_value * m2.distance

    return filter


def create_feature_matcher(matcher, match_filter):
    @log_feature_match
    @stateful_decorator(
        keep=2,
        append_empty=True,
        default_value=lambda: [],
    )
    def match_keypoints(query_frame, training_frame):
        if any(f.desc is None for f in [query_frame, training_frame]):
            return []
        matches = matcher(query_frame.desc, training_frame.desc)
        if len(matches) == 0:
            return []
        matches_1 = []
        matches_2 = []
        for match in matches:
            if match_filter(match):
                match = match[0]
                matches_1 += [query_frame.key_pts[match.queryIdx].pt]
                matches_2 += [training_frame.key_pts[match.trainIdx].pt]
        if len(matches_1) == 0:
            return []
        return np.dstack((matches_1, matches_2))

    return match_keypoints
