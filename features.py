import cv2 as cv

# Detection

orb = cv.ORB_create()

def compute_orb_features(frames):
    for frame_id, frame in enumerate(frames):
        key_pts = orb.detect(frame, None)
        key_pts, desc = orb.compute(frame, key_pts)
        yield {
            "frame_id": frame_id,
            "image": frame,
            "key_pts": key_pts,
            "desc": desc,
        }

# Matching

def create_orb_flann_matcher():
    FLANN_INDEX_LSH = 6
    INDEX_PARAMS = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,  # 12
        key_size=12,  # 20
        multi_probe_level=2,  # 2
    )

    cv_matcher = cv.FlannBasedMatcher(INDEX_PARAMS)
    return KeyPointMatcher(cv_matcher.knnMatch)


def create_stateful_orb_flann_matcher():
    matcher = create_orb_flann_matcher()
    return StatefulMatcher(
        matcher.match_keypoints,
        matcher.draw_matches,
    )


class KeyPointMatcher:
    def __init__(self, matcher):
        self.__matcher__ = matcher

    def match_keypoints(self, frame1, frame2):
        good_matches = []
        matches = self.__matcher__(
            frame1["desc"],
            frame2["desc"],
            k=2,
        )
        for x, y in matches:
            thresh_value = 0.7
            if x.distance < thresh_value * y.distance:
                good_matches += [x]
        return good_matches

    def draw_matches(self, frame1, frame2, matches):
        to_int = lambda x: tuple(int(round(c)) for c in x)
        frame_with_matches = frame1["image"].copy()
        for m in matches:
            current_pos = to_int(frame1["key_pts"][m.queryIdx].pt)
            last_pos = to_int(frame2["key_pts"][m.trainIdx].pt)
            cv.circle(
                frame_with_matches,
                current_pos,
                radius=4,
                color=(0, 0, 255),
            )
            cv.circle(
                frame_with_matches,
                last_pos,
                radius=2,
                color=(255, 0, 0),
            )
            cv.line(
                frame_with_matches,
                current_pos,
                last_pos,
                color=(0, 255, 0),
            )
        cv.imshow("", frame_with_matches)


class StatefulMatcher:
    def __init__(self, matcher, drawer):
        self.__matcher__ = matcher
        self.__drawer__ = drawer
        self.__old_frame__ = None
        self.draw_matches = lambda m: None

    def match_keypoints(self, new_frame):
        good_matches = None
        if self.__old_frame__ is not None:
            good_matches = self.__matcher__(
                new_frame,
                self.__old_frame__,
            )
            old_frame = self.__old_frame__
            self.draw_matches = lambda m: self.__drawer__(
                new_frame,
                old_frame,
                m,
            )
        self.__old_frame__ = new_frame
        return good_matches
