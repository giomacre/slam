#!/usr/bin/env python3
import sys, os
import numpy

sys.path.append(
    os.path.join(
        os.path.dirname(sys.argv[0]),
        "lib",
    )
)
from geometry import SO3, SE3

print(SE3)
print("\n".join(filter(lambda x: not str.startswith(x, "_"), dir(SE3))))
print("")

print(SO3)
print("\n".join(filter(lambda x: not str.startswith(x, "_"), dir(SO3))))
print("")

I = numpy.eye(4)
T = SE3.fromH(I)

print(T.q().R(), T.t())
