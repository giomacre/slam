import cv2 as cv
import numpy as np
from decorators import ddict
from slam_logging import log_feature_match, log_feature_extraction

# Detectors


def create_orb_detector(compute_descriptors=True, **orb_args):
    orb = cv.ORB_create(**orb_args)

    @log_feature_extraction
    def orb_detector(frame, max_features=None, mask=None):
        if max_features is not None:
            orb.setMaxFeatures(max_features)
        key_pts = orb.detect(frame.image, mask=mask)
        desc = None
        if compute_descriptors:
            key_pts, desc = orb.compute(frame.image, key_pts)
        key_pts = np.array([k.pt for k in key_pts])
        frame |= ddict(
            key_pts=key_pts,
            desc=desc,
        )
        return frame

    return orb_detector


# Matchers


def create_bruteforce_matcher(detector, k=2, **bf_matcher_args):
    cv_matcher = cv.BFMatcher_create(**bf_matcher_args)
    return create_feature_matcher(
        detector,
        lambda d1, d2: cv_matcher.knnMatch(d1, d2, k=k),
        ratio_test_filter(),
    )


def ratio_test_filter(thresh_value=0.7):
    def filter(matches):
        if len(matches) < 2:
            return False
        m1, m2 = matches
        return m1.distance < thresh_value * m2.distance

    return filter


def create_feature_matcher(detector, matcher, match_filter):
    no_match = [[]] * 3

    @log_feature_match
    def match_keypoints(query_frame, train_frame):
        query_frame = detector(query_frame)
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
        query_idxs = indices[:, 0]
        train_idxs = indices[:, 1]
        return (
            np.dstack(
                [
                    query_frame.key_pts[query_idxs],
                    train_frame.key_pts[train_idxs],
                ]
            ),
            query_idxs,
            train_idxs,
        )

    return match_keypoints
