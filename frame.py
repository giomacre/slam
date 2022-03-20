import numpy as np
from decorators import ddict


def create_frame(image):
    return ddict(
        id=0,
        image=image,
        key_pts=np.array([]),
        desc=np.array([]),
        pose=None,
        observations=[],
        is_keyframe=False,
    )
