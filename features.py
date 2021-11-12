import cv2 as cv
import numpy as np
from slam_logging import log_feature_match, log_feature_extraction

# Detectors


def create_orb_detector():
    orb = cv.ORB_create(
        nfeatures=1500,
        WTA_K=2,
        scoreType=cv.ORB_HARRIS_SCORE,
    )

    @log_feature_extraction
    def orb_detector(frame):
        key_pts = orb.detect(frame.image)
        key_pts, frame.desc = orb.compute(frame.image, key_pts)
        if frame.desc is None:
            return frame
        frame.key_pts = np.array([k.pt for k in key_pts])
        frame.origin_pts = frame.key_pts.copy()
        frame.origin_frames = np.full(
            (len(frame.key_pts),),
            fill_value=frame.id,
        )
        frame.tracked_idxs = np.arange(len(frame.key_pts))
        return frame

    return orb_detector


# Matchers


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
    def match_keypoints(query_frame, train_frame):
        no_match = [[]] * 3
        if any(f.desc is None for f in [query_frame, train_frame]):
            return no_match
        matches = matcher(query_frame.desc, train_frame.desc)
        if len(matches) == 0:
            return no_match
        indices = np.vstack(
            [(m[0].queryIdx, m[0].trainIdx) for m in matches if match_filter(m)]
        )
        if len(indices) == 0:
            return no_match
        query_idxs, train_idxs = indices[:, 0], indices[:, 1]
        query_pts, train_pts = query_frame.key_pts, train_frame.key_pts
        query_matches = query_pts[query_idxs]
        train_matches = train_pts[train_idxs]
        return (
            np.dstack((query_matches, train_matches)),
            query_idxs,
            train_idxs,
        )

    return match_keypoints
