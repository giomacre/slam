from utils.decorators import ddict
from collections import OrderedDict


def create_point(frame, key_pt_idx):
    point = ddict(
        idxs=OrderedDict({frame.id: key_pt_idx}),
        coords=None,
    )
    return point
