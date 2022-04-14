#!/usr/bin/env python3
import sys, os
import numpy as np

sys.path.append(
    os.path.join(
        os.path.dirname(sys.argv[0]),
        "lib",
    )
)
from geometry import SO3, SE3
import PyCeresFactors as factors
import PyCeres as ceres

# print(SE3)
print("")

# print(SO3)
# print("\n".join(filter(lambda x: not str.startswith(x, "_"), dir(SO3))))
# print("")

with open("numpy_dumps/K.npb", "rb") as file:
    K = np.load(file)
for a in ["epnp", "p3p", "itpnp"]:
    with open(f"numpy_dumps/T{a}.npb", "rb") as file:
        T = np.load(file)

    with open(f"numpy_dumps/3dc{a}.npb", "rb") as file:
        lm_coords = np.load(file)

    with open(f"numpy_dumps/2dc{a}.npb", "rb") as file:
        image_coords = np.load(file)
    d = np.random.normal(scale=0.25, size=6)
    d = SE3.Log(SE3.identity())
    T0 = SE3.fromH(T) * SE3.Exp(d)
    T0_c = T0.array()

    problem = ceres.Problem()
    problem.AddParameterBlock(T0_c, 7, factors.SE3Parameterization())

    K = K.astype(np.float32)

    for px, pt_3d in zip(image_coords, lm_coords):
        factor = factors.SE3ReprojectionFactor(K, px.T, pt_3d.T)
        problem.AddResidualBlock(factor, None, T0_c)

    options = ceres.SolverOptions()
    options.max_num_iterations = 25
    # options.linear_solver_type = ceres.LinearSolverType.DENSE_QR
    options.linear_solver_type = ceres.LinearSolverType.DENSE_SCHUR
    # options.linear_solver_type = ceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = (
        ceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
    )
    options.function_tolerance = 1e-3
    options.minimizer_progress_to_stdout = True

    summary = ceres.Summary()
    ceres.Solve(options, problem, summary)

print(
    "\n".join(
        dir(factor),
    )
)
