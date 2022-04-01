from dataclasses import dataclass, field
from typing import List
import numpy as np
from utils.decorators import ddict


@dataclass
class Frame:
    id: int
    image: np.ndarray
    key_pts: np.ndarray = np.array([])
    desc: np.ndarray = np.empty(shape=())
    pose: np.ndarray = np.empty(shape=())
    observations: List = field(default_factory=lambda: [])
    is_keyframe: bool = False


def create_frame(id, image):
    return Frame(id, image)
