import cv2 as cv
from functools import reduce
import numpy as np
from .features import create_feature_detector
from ..utils.slam_logging import log_feature_match
from ..utils.params import frontend_params


def create_lk_feature_detector(undistort, **orb_args):
    base_detector = create_feature_detector(undistort, **orb_args)

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
                    frontend_params["keypoint_radius"],
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
    window = [frontend_params["klt_window_size"]] * 2
    criteria = (
        cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,
        frontend_params["klt_max_iter"],
        frontend_params["klt_convergence_threshold"],
    )
    tracked_points, status, errors = cv.calcOpticalFlowPyrLK(
        train_gray,
        query_gray,
        train_pts,
        None,
        flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS,
        winSize=window,
        criteria=criteria,
    )
    low_err = errors.ravel() < frontend_params["klt_inlier_threshold"]
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
        winSize=window,
        criteria=criteria,
    )
    status_reverse = status_reverse.ravel().astype(np.bool)
    good_matches = np.abs(tracked_reverse - train_pts).max(axis=2) < 0.5
    good_matches = good_matches.ravel()
    good_idxs = np.flatnonzero(
        reduce(
            np.bitwise_and,
            [
                status,
                low_err,
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
