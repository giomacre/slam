from dataclasses import dataclass
import numpy
from collections import OrderedDict


@dataclass
class Point:
    idxs: OrderedDict
    coords: numpy.ndarray = numpy.empty(shape=())
    color: numpy.ndarray = numpy.empty(shape=())
    is_initialized: bool = False


def create_point(frame, key_pt_idx):
    return Point({frame.id: key_pt_idx})
