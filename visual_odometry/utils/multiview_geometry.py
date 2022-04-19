from functools import reduce
import sys
import os
from click import option
import numpy as np
import cv2 as cv

from ..frontend.camera_calibration import to_homogeneous, to_image_coords
from .slam_logging import log_pose_estimation, log_triangulation, performance_timer
from .params import frontend_params, ransac_params

sys.path.append(
    os.path.join(
        os.path.dirname(sys.argv[0]),
        "lib",
    )
)

from geometry import SE3
import PyCeresFactors as factors
import PyCeres as ceres

# @log_pose_estimation
def epipolar_ransac(K, query_pts, train_pts):
    if len(train_pts) < 5:
        return [None] * 2
    E, mask = cv.findEssentialMat(
        train_pts,
        query_pts,
        K,
        prob=ransac_params["em_confidence"],
        threshold=ransac_params["em_threshold"],
    )
    n_inliers = np.count_nonzero(mask)
    if n_inliers < 5:
        return [None] * 2

    train_pts, query_pts = cv.correctMatches(
        E,
        train_pts[None, ...].copy(),
        query_pts[None, ...].copy(),
    )
    _, R, t, _ = cv.recoverPose(
        E,
        train_pts,
        query_pts,
        K,
        mask=mask.copy(),
    )
    T = construct_pose(R.T, -R.T @ t * frontend_params["epipolar_scale"])
    mask = mask.astype(np.bool).ravel()
    return T, mask


def ceres_pnp_solver(
    K,
    T,
    world_pts,
    image_pts,
    use_robust_loss=False,
):
    chi_err = frontend_params["pnp_ceres_chi"]
    T0 = SE3.fromH(T)
    T_curr = T0.array()
    Kf = K.astype(np.float32)
    problem = ceres.Problem()
    robust_loss = ceres.HuberLoss(np.sqrt(chi_err)) if use_robust_loss else None
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


def pnp_ransac(K, lm_coords, image_coords, T=None):
    @performance_timer()
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
        _, rot, t, inliers_r = cv.solvePnPRansac(
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
        R = cv.Rodrigues(rot)[0]
        T = construct_pose(R.T, -R.T @ t)
        if len(inliers_r) < 4:
            return [None] * 2
        return T, inliers_r

    @performance_timer()
    def ceres_refinement(rot, t, inliers):
        R = cv.Rodrigues(rot)[0]
        T = construct_pose(R.T, -R.T @ t)
        retval_r, T_r, inliers_r = ceres_pnp_solver(
            K,
            T,
            lm_coords[inliers],
            image_coords[inliers],
            use_robust_loss=True,
        )
        if not retval_r:
            return None, None
        return T_r, inliers_r
        retval_l2, T_l2, inliers_l2 = ceres_pnp_solver(
            K,
            T_r,
            lm_coords[inliers][inliers_r],
            image_coords[inliers][inliers_r],
        )
        if not retval_l2:
            return None, None
        inliers_r[inliers_r] = inliers_l2
        return T_l2, inliers_r

    retval, rot, t, inliers = initial_estimate()
    if not retval or len(inliers) < 4:
        return [None] * 2
    if ransac_params["p3p_ceres_refinement"]:
        # inliers = np.arange(len(lm_coords))
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


def create_point_triangulator(K):
    # @log_triangulation
    K = np.hstack((K, np.array([0, 0, 0]).reshape(3, 1)))

    def triangulation(
        current_pose,
        reference_pose,
        current_points,
        reference_points,
    ):
        current_extrinsics = np.linalg.inv(current_pose)
        reference_extrinsics = np.linalg.inv(reference_pose)
        points_4d = np.array(
            cv.triangulatePoints(
                (K @ reference_extrinsics),
                (K @ current_extrinsics),
                reference_points.T,
                current_points.T,
            )
        ).T
        points_4d = points_4d[points_4d[:, -1] != 0]
        points_4d /= points_4d[:, -1:]
        camera_coords = np.dstack(
            [
                (extrinsics @ points_4d.T).T
                for extrinsics in [
                    current_extrinsics,
                    reference_extrinsics,
                ]
            ]
        ).T
        projected = to_image_coords(K, camera_coords)
        low_err = reduce(
            np.bitwise_and,
            (
                np.linalg.norm(a - b.T, axis=0) < ransac_params["p3p_threshold"]
                for a, b in zip(
                    projected,
                    [current_points, reference_points],
                )
            ),
        )
        in_front = reduce(
            np.bitwise_and,
            (pts[2, :] > 0 for pts in camera_coords),
        ).T
        good_pts = in_front & low_err
        return points_4d[..., :3], good_pts

    return triangulation


def construct_pose(R, t):
    return np.vstack(
        (np.hstack((R, t)), [0, 0, 0, 1]),
    )
