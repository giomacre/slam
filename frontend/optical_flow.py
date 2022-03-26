import cv2 as cv
import numpy as np
from utils.decorators import ddict
from frontend.features import create_orb_detector
from utils.slam_logging import log_feature_match


def create_lk_orb_detector(**orb_args):
    orb = create_orb_detector(**orb_args)

    def detector(query_frame, max_features=None):
        mask_trackings = None
        if len(query_frame.key_pts) > 0:
            mask_trackings = np.full_like(
                query_frame.image[..., 0],
                fill_value=255,
                dtype=np.uint8,
            )
            for point in query_frame.key_pts:
                cv.circle(
                    mask_trackings,
                    np.int32(point),
                    5,
                    0,
                    thickness=cv.FILLED,
                )
        return orb(
            query_frame,
            max_features,
            mask_trackings,
        )

    return detector


def track_to_new_frame(query_frame, train_frame):
    train_pts = train_frame.key_pts.reshape(-1, 1, 2)
    query_gray, train_gray = (
        cv.cvtColor(f.image, cv.COLOR_BGR2GRAY)
        for f in [
            query_frame,
            train_frame,
        ]
    )
    tracked_points, *_ = cv.calcOpticalFlowPyrLK(
        train_gray,
        query_gray,
        train_pts.astype(np.float32),
        None,
    )
    tracked_reverse, *_ = cv.calcOpticalFlowPyrLK(
        query_gray,
        train_gray,
        tracked_points,
        None,
    )
    good_matches = np.abs(tracked_reverse - train_pts).max(axis=2) < 1
    good_matches = good_matches.ravel()
    return (
        np.dstack(
            (
                tracked_points.reshape(-1, 2),
                train_pts.reshape(-1, 2),
            )
        ),
        good_matches,
    )


def create_lk_tracker():
    @log_feature_match
    def tracker(query_frame, train_frame):
        tracked, good = track_to_new_frame(
            query_frame,
            train_frame,
        )
        num_tracked = np.count_nonzero(good)
        query_idxs = np.arange(num_tracked)
        train_idxs = np.flatnonzero(good)
        return (
            tracked[good],
            query_idxs,
            train_idxs,
        )

    return tracker
