import cv2 as cv
import numpy as np


def create_orb_flann_matcher():
    FLANN_INDEX_LSH = 6
    INDEX_PARAMS = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,
        key_size=12,
        multi_probe_level=2,
    )

    cv_matcher = cv.FlannBasedMatcher(INDEX_PARAMS)
    return KeyPointMatcher(
        lambda d1, d2: cv_matcher.knnMatch(d1, d2, k=2),
        ratio_test_filter,
        draw_match,
    )


def create_stateful_matcher(matcher):
    matcher = matcher()
    return StatefulMatcher(
        matcher.match_keypoints,
        matcher.draw_matches,
    )


def create_bruteforce_matcher():
    cv_matcher = cv.BFMatcher_create(cv.NORM_HAMMING2, crossCheck=False)
    return KeyPointMatcher(
        lambda d1, d2: cv_matcher.knnMatch(d1, d2, k=2),
        # lambda m: len(m) > 0,
        ratio_test_filter,
        draw_match,
    )


def ratio_test_filter(matches, thresh_value=0.7):
    m1, m2 = matches
    return m1.distance < thresh_value * m2.distance


class KeyPointMatcher:
    def __init__(self, matcher, match_filter, drawer):
        self.__matcher__ = matcher
        self.__drawer__ = drawer
        self.__filter__ = match_filter

    def match_keypoints(self, frame1, frame2):
        matches = self.__matcher__(
            frame1["desc"],
            frame2["desc"],
        )
        if len(matches) == 0:
            return None
        matches_1 = []
        matches_2 = []
        for match in matches:
            if self.__filter__(match):
                match = match[0]
                matches_1 += [frame1["key_pts"][match.queryIdx].pt]
                matches_2 += [frame2["key_pts"][match.trainIdx].pt]
        return np.dstack((matches_1, matches_2))

    def draw_matches(self, image, matches):
        to_int = lambda x: tuple(int(round(c)) for c in x)
        frame_with_matches = image.copy()
        for m in matches:
            current_pos = to_int(m[..., 0])
            last_pos = to_int(m[..., 1])
            self.__drawer__(
                frame_with_matches,
                current_pos,
                last_pos,
            )
        cv.imshow("", frame_with_matches)


class StatefulMatcher:
    def __init__(self, matcher, drawer):
        self.draw_matches = drawer
        self.__matcher__ = matcher
        self.__old_frame__ = None

    def match_keypoints(self, new_frame):
        matches = None
        if self.__old_frame__ is not None:
            matches = self.__matcher__(
                new_frame,
                self.__old_frame__,
            )
        self.__old_frame__ = new_frame
        return matches


def draw_match(image, match1, match2):
    cv.circle(
        image,
        match1,
        radius=4,
        color=(0, 0, 255),
    )
    cv.circle(
        image,
        match2,
        radius=2,
        color=(255, 0, 0),
    )
    cv.line(
        image,
        match1,
        match2,
        color=(0, 255, 0),
    )
