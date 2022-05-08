from dataclasses import dataclass
import numpy as np
from collections import OrderedDict, defaultdict
from ..utils.params import frontend_params


def initialize_tracked_landmarks(
    get_parallax,
    triangulation,
    tracked_frames,
    frame,
):
    new_points = []
    candidate_pts = [
        (id, lm)
        for id, lm in frame.landmarks.items()
        if not lm.is_initialized and len(lm.observations) > 1
    ]
    matches = defaultdict(lambda: [[], []])
    for curr_idx, lm in candidate_pts:
        id, idx = next(x for x in lm.observations.items())
        ref_idxs, curr_idxs = matches[id]
        ref_idxs += [idx]
        curr_idxs += [curr_idx]
    for kf_id, (ref_idxs, curr_idxs) in matches.items():
        ref_kf = tracked_frames[kf_id]
        ref_idxs = np.array(ref_idxs)
        curr_idxs = np.array(curr_idxs)
        parallax = get_parallax(
            frame.pose,
            ref_kf.pose,
            frame.undist[curr_idxs],
            ref_kf.undist[ref_idxs],
        )
        pts_3d, good_pts = triangulation(
            frame.pose,
            ref_kf.pose,
            frame.undist[curr_idxs],
            ref_kf.undist[ref_idxs],
        )
        old_keyframes = parallax > 0.5 * frontend_params["kf_parallax_threshold"]
        for i in ref_idxs[~good_pts & old_keyframes]:
            del ref_kf.landmarks[i].observations[kf_id]
            del ref_kf.landmarks[i]
        for i, pt in zip(curr_idxs[good_pts], pts_3d[good_pts]):
            to_idx = lambda kp: tuple(np.rint(kp).astype(int)[::-1])
            landmark = frame.landmarks[i]
            landmark.coords = pt
            img_idx = to_idx(frame.key_pts[i])
            landmark.color = frame.image[img_idx] / 255.0
            landmark.is_initialized = True
            new_points += [landmark]
    return new_points


@dataclass
class LandMark:
    observations: OrderedDict
    coords: np.ndarray = np.empty(shape=())
    color: np.ndarray = np.empty(shape=())
    is_initialized: bool = False


def create_landmark(frame, key_pt_idx):
    return LandMark({frame.id: key_pt_idx})
