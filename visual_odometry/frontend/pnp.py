import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(sys.argv[0]),
        "lib",
    )
)

from geometry import SE3
import PyCeresFactors as factors
import PyCeres as ceres
from ..utils.params import ransac_params
import numpy as np
import cv2 as cv
from .camera_calibration import to_image_coords, to_homogeneous
from ..multiview_geometry import construct_pose


def ceres_pnp_solver(
    K,
    T,
    world_pts,
    image_pts,
):
    T0 = SE3.fromH(T)
    T_curr = T0.array()
    Kf = K.astype(np.float32)
    problem = ceres.Problem()
    chi_err = ransac_params["pnp_ceres_huber_threshold"]
    robust_loss = (
        ceres.HuberLoss(np.sqrt(chi_err))
        if ransac_params["pnp_ceres_use_huber_loss"]
        else None
    )
    problem.AddParameterBlock(
        T_curr,
        7,
        factors.SE3Parameterization(),
    )
    for i in range(len(world_pts)):
        factor = factors.SE3ReprojectionFactor(
            Kf,
            image_pts[i].T,
            world_pts[i].T,
        )
        problem.AddResidualBlock(factor, robust_loss, T_curr)

    options = ceres.SolverOptions()
    options.linear_solver_type = ceres.LinearSolverType.DENSE_QR
    options.trust_region_strategy_type = (
        ceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
    )
    options.num_threads = 1
    options.function_tolerance = 1e-3
    options.minimizer_progress_to_stdout = False
    summary = ceres.Summary()
    ceres.Solve(options, problem, summary)
    T = SE3(T_curr).H()
    world_h = to_homogeneous(world_pts.squeeze().T)
    camera_coords = (np.linalg.inv(T) @ world_h)[:3, ...]
    in_front = camera_coords[2, ...] > 0.0
    image_pts = image_pts.squeeze().T
    reproj_err = np.linalg.norm(
        image_pts
        - to_image_coords(
            K,
            camera_coords,
        ),
        axis=0,
    )
    inliers = (reproj_err < (ransac_params["p3p_threshold"])) & in_front
    if np.count_nonzero(inliers) < 4:
        return False, None, None
    return (
        summary.IsSolutionUsable(),
        T,
        inliers,
    )


def pnp_ransac(K, lm_coords, image_coords):
    def initial_estimate():
        return cv.solvePnPRansac(
            lm_coords,
            image_coords,
            K,
            distCoeffs=None,
            reprojectionError=ransac_params["p3p_threshold"],
            confidence=ransac_params["p3p_confidence"],
            flags=cv.SOLVEPNP_P3P,
        )

    def iterative_refinement(rotvec, tvec, inliers):
        retval, rot, t, inliers_r = cv.solvePnPRansac(
            lm_coords[inliers],
            image_coords[inliers],
            K,
            None,
            rotvec,
            tvec,
            reprojectionError=ransac_params["p3p_threshold"],
            confidence=ransac_params["p3p_confidence"],
            iterationsCount=ransac_params["p3p_iterations"],
            useExtrinsicGuess=True,
            flags=cv.SOLVEPNP_ITERATIVE,
        )
        if not retval or len(inliers_r) < 4:
            return [None] * 2
        R = cv.Rodrigues(rot)[0]
        T = construct_pose(R.T, -R.T @ t)
        return T, inliers_r

    def ceres_refinement(rot, t, inliers):
        R = cv.Rodrigues(rot)[0]
        T = construct_pose(R.T, -R.T @ t)
        retval_rl, T_rl, inliers_rl = ceres_pnp_solver(
            K,
            T,
            lm_coords[inliers],
            image_coords[inliers],
        )
        if not retval_rl:
            return None, None
        return T_rl, inliers_rl

    retval, rot, t, inliers = initial_estimate()
    if not retval or len(inliers) < 4:
        return [None] * 2
    if ransac_params["pnp_ceres_refinement"]:
        T_r, inliers_r = ceres_refinement(rot, t, inliers)
    else:
        T_r, inliers_r = iterative_refinement(rot, t, inliers)
    if T_r is None:
        return [None] * 2
    inliers = inliers[inliers_r.flatten()]
    mask = np.full(len(lm_coords), False)
    inliers = inliers.flatten()
    mask[inliers] = True
    if np.count_nonzero(mask) < 0.5 * len(lm_coords):
        return [None] * 2
    return T_r, mask
