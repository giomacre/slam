import cv2 as cv
import numpy as np


def create_orb_flann_matcher():
    FLANN_INDEX_LSH = 6
    INDEX_PARAMS = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,  # 12
        key_size=12,  # 20
        multi_probe_level=2,  # 2
    )

    cv_matcher = cv.FlannBasedMatcher(INDEX_PARAMS)
    matcher = KeyPointMatcher(cv_matcher.knnMatch, draw_match)
    return StatefulMatcher(
        matcher.match_keypoints,
        matcher.draw_matches,
    )


class KeyPointMatcher:
    def __init__(self, matcher, drawer):
        self.__matcher__ = matcher
        self.__drawer__ = drawer

    def match_keypoints(self, frame1, frame2):
        matches = self.__matcher__(
            frame1["desc"],
            frame2["desc"],
            k=2,
        )
        matches_1 = []
        matches_2 = []
        for m1, m2 in matches:
            thresh_value = 0.7
            if m1.distance < thresh_value * m2.distance:
                matches_1 += [frame1["key_pts"][m1.queryIdx].pt]
                matches_2 += [frame2["key_pts"][m1.trainIdx].pt]
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
