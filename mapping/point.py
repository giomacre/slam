from dataclasses import dataclass
from types import NoneType
from typing import Union

import numpy
from sympy import Order
from utils.decorators import ddict
from collections import OrderedDict


@dataclass
class Point:
    idxs: OrderedDict
    coords: numpy.ndarray = numpy.empty(shape=())
    color: numpy.ndarray = numpy.empty(shape=())
    is_initialized: bool = False


def create_point(frame, key_pt_idx):
    return Point({frame.id: key_pt_idx})
