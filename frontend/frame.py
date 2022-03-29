import numpy as np
from utils.decorators import ddict


def create_frame(id, image):
    return ddict(
        id=id,
        image=image,
        key_pts=np.array([]),
        desc=np.array([]),
        pose=None,
        observations=[],
        is_keyframe=False,
    )
