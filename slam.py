#!/usr/bin/env python3

import numpy as np
import cv2 as cv

VIDEO_PATH = "./drone.webm"
SCALE = 2
# FLANN parameters
FLANN_INDEX_LSH = 6
index_params = dict(
    algorithm=FLANN_INDEX_LSH,
    table_number=6,  # 12
    key_size=12,  # 20
    multi_probe_level=2,  # 2
)
orb = cv.ORB_create()
matcher = cv.FlannBasedMatcher(
    index_params,
)


def video_stream(video_path):
    video = cv.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        height, width, *_ = frame.shape
        frame = cv.resize(
            frame,
            (width // SCALE, height // SCALE),
            interpolation=cv.INTER_AREA,
        )
        yield frame
    video.release()


def compute_key_points(video_path):
    for frame_id, frame in enumerate(video_stream(video_path)):
        key_pts = orb.detect(frame, None)
        key_pts, desc = orb.compute(frame, key_pts)
        yield {
            "frame_id": frame_id,
            "image": frame,
            "key_pts": key_pts,
            "desc": desc,
        }


def draw_matches(matches):
    match_img = image.copy()
    for m in matches:
        to_int = lambda x: tuple(int(round(c)) for c in x)
        current_pos = to_int(key_pts[m.queryIdx].pt)
        last_pos = to_int(match_against["key_pts"][m.trainIdx].pt)
        cv.circle(
            match_img,
            current_pos,
            radius=4,
            color=(0, 0, 255),
        )
        cv.circle(
            match_img,
            last_pos,
            radius=2,
            color=(255, 0, 0),
        )
        cv.line(
            match_img,
            current_pos,
            last_pos,
            color=(0, 255, 0),
        )
    cv.imshow("keypoints", match_img)


match_against = None
for frame in compute_key_points(VIDEO_PATH):
    id, image, key_pts, desc = frame.values()
    good_matches = []
    if match_against is not None:
        matches = matcher.knnMatch(desc, match_against["desc"], k=2)
        good_matches = []
        for x, y in matches:
            thresh_value = 0.7
            if x.distance < thresh_value * y.distance:
                good_matches += [x]
        draw_matches(good_matches)
    match_against = {
        "key_pts": key_pts,
        "desc": desc,
        "image": image,
    }
    cv.waitKey(1)
