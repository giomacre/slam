import cv2 as cv
from functools import reduce
import numpy as np
from .features import create_orb_detector
from ..utils.slam_logging import log_feature_match


def create_lk_orb_detector(undistort, **orb_args):
    base_detector = create_orb_detector(undistort, **orb_args)

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
        return base_detector(
            query_frame,
            max_features,
            mask_trackings,
        )

    return detector


@log_feature_match
def track_to_new_frame(query_frame, train_frame):
    train_pts = train_frame.key_pts.reshape(-1, 1, 2).copy()
    query_gray, train_gray = (
        cv.cvtColor(f.image, cv.COLOR_BGR2GRAY)
        for f in [
            query_frame,
            train_frame,
        ]
    )
    tracked_points, status, _ = cv.calcOpticalFlowPyrLK(
        train_gray,
        query_gray,
        train_pts,
        None,
        flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS,
    )
    status = status.ravel().astype(np.bool)
    inside_limits = reduce(
        np.bitwise_and,
        [
            tracked_points[..., 0] >= 0.5,
            tracked_points[..., 0] < query_gray.shape[1] - 0.5,
            tracked_points[..., 1] >= 0.5,
            tracked_points[..., 1] < query_gray.shape[0] - 0.5,
        ],
    ).ravel()
    tracked_reverse, status_reverse, _ = cv.calcOpticalFlowPyrLK(
        query_gray,
        train_gray,
        tracked_points,
        train_pts,
        flags=sum(
            [
                cv.OPTFLOW_USE_INITIAL_FLOW,
                cv.OPTFLOW_LK_GET_MIN_EIGENVALS,
            ]
        ),
    )
    status_reverse = status_reverse.ravel().astype(np.bool)
    good_matches = (
        np.linalg.norm(
            tracked_reverse - train_pts,
            axis=2,
        )
        < 0.5
    )
    good_matches = good_matches.ravel()
    good_idxs = np.flatnonzero(
        reduce(
            np.bitwise_and,
            [
                status,
                inside_limits,
                status_reverse,
                good_matches,
            ],
        )
    )
    return (
        tracked_points[good_idxs].reshape(-1, 2),
        good_idxs,
    )
