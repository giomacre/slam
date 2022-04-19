from dataclasses import dataclass, field
from typing import List, OrderedDict
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    id: int
    key_pts: np.ndarray = np.array([])
    undist: np.ndarray = np.empty(shape=())
    pose: np.ndarray = np.empty(shape=())
    observations: OrderedDict = field(default_factory=lambda: OrderedDict([]))
    is_keyframe: bool = False


def create_frame(image, id):
    return Frame(image, id)
