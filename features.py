import cv2 as cv
import numpy as np
from decorators import StatefulDecorator


def create_orb_flann_matcher():
    FLANN_INDEX_LSH = 6
    INDEX_PARAMS = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,
        key_size=12,
        multi_probe_level=2,
    )

    cv_matcher = cv.FlannBasedMatcher(INDEX_PARAMS)
    return create_feature_matcher(
        lambda d1, d2: cv_matcher.knnMatch(d1, d2, k=2),
        ratio_test_filter,
    )


def create_bruteforce_matcher():
    cv_matcher = cv.BFMatcher_create(cv.NORM_HAMMING2)
    return create_feature_matcher(
        lambda d1, d2: cv_matcher.knnMatch(d1, d2, k=2),
        ratio_test_filter,
    )


def ratio_test_filter(matches, thresh_value=0.7):
    m1, m2 = matches
    return m1.distance < thresh_value * m2.distance


def create_feature_matcher(matcher, match_filter):
    @StatefulDecorator
    def match_keypoints(training_frame, query_frame):
        matches = matcher(query_frame["desc"], training_frame["desc"])
        if len(matches) == 0:
            return None
        matches_1 = []
        matches_2 = []
        for match in matches:
            if match_filter(match):
                match = match[0]
                matches_1 += [query_frame["key_pts"][match.queryIdx].pt]
                matches_2 += [training_frame["key_pts"][match.trainIdx].pt]
        if len(matches_1) == 0:
            return None
        return np.dstack((matches_1, matches_2))

    return match_keypoints
