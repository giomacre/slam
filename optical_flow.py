import cv2 as cv
import numpy as np
from features import create_orb_detector
from slam_logging import log_feature_match


def create_lk_orb_detector(**orb_args):
    orb = create_orb_detector(compute_descriptors=False, **orb_args)

    def detector(query_frame, tracked_points=None):
        mask_trackings = None
        if tracked_points is not None:
            mask_trackings = np.full_like(
                query_frame.image[..., 0],
                fill_value=255,
                dtype=np.uint8,
            )
            for point in tracked_points:
                cv.circle(
                    mask_trackings,
                    np.int32(point),
                    5,
                    0,
                    thickness=cv.FILLED,
                )
        return orb(query_frame, mask_trackings)

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
    tracked_reversed, *_ = cv.calcOpticalFlowPyrLK(
        query_gray,
        train_gray,
        tracked_points,
        None,
    )
    good_tracks = np.abs(tracked_reversed - train_pts).max(axis=2) < 1
    good_tracks = good_tracks.ravel()
    return (
        np.dstack(
            (
                tracked_points.reshape(-1, 2),
                train_pts.reshape(-1, 2),
            )
        ),
        good_tracks,
    )


def create_lk_tracker(detector, min_points=600):
    @log_feature_match
    def tracker(query_frame, train_frame):
        if len(train_frame.key_pts) == 0:
            detector(query_frame)
            return [[]] * 3

        tracks, good = track_to_new_frame(
            query_frame,
            train_frame,
        )
        num_tracked = np.count_nonzero(good)
        if num_tracked <  min_points:
            query_frame = detector(
                query_frame,
                tracks[good, ..., 0],
            )
            query_frame.key_pts = np.vstack(
                (
                    tracks[good, ..., 0],
                    query_frame.key_pts,
                )
            )
        else:
            query_frame.key_pts = tracks[good, ..., 0]
        query_frame.desc = None
        query_frame.origin_frames = np.full(
            len(query_frame.key_pts),
            query_frame.id,
        )
        query_frame.origin_pts = query_frame.key_pts.copy()
        query_idxs = np.arange(num_tracked)
        train_idxs = np.flatnonzero(good)
        return (
            tracks[good],
            query_idxs,
            train_idxs,
        )

    return tracker