#! /usr/bin/env python3
import sys

import sympy

sys.path.append("lib")

import PyCeres as ceres
import numpy as np

I = np.zeros((7, 1), dtype=np.float64)
I[3] = 1.0
x = I.copy()
x[-3:] = 0.001
problem = ceres.Problem()
problem.AddParameterBlock(
    I,
    7,
    ceres.create_se3_parameterization(),
)
problem.AddResidualBlock(
    ceres.create_se3_functor(I),
    None,
    x,
)

options = ceres.SolverOptions()
options.max_num_iterations = 5
options.linear_solver_type = ceres.LinearSolverType.DENSE_QR
options.minimizer_progress_to_stdout = True

summary = ceres.Summary()
ceres.Solve(options, problem, summary)
print(summary.BriefReport())
