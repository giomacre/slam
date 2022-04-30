#!/usr/bin/env python3
import json
from operator import itemgetter
from typing import DefaultDict
from visual_odometry import start
from visual_odometry.utils.params import import_parameters
import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(sys.argv[0]),
        "lib",
    )
)

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentTypeError

    def check_video(path):
        from cv2 import VideoCapture

        video = VideoCapture(path)
        if video.isOpened():
            return path
        raise ArgumentTypeError(f"{path} is not a valid video file.")

    def check_params(path):
        params = DefaultDict(lambda: None)
        try:
            file = open(path, "r")
            params_json = json.load(file)
            params |= params_json
            camera_properties = [
                "fx",
                "fy",
                "cx",
                "cy",
                "k1",
                "k2",
                "p1",
                "p2",
                "k3",
            ]
            frontend_properties = [
                "max_features",
                "min_features",
                "keypoint_radius",
                "kf_landmark_ratio",
                "kf_parallax_threshold",
                "epipolar_scale",
                "pnp_ceres_chi",
                "fast_threshold",
                "klt_window_size",
                "klt_max_iter",
                "klt_convergence_threshold",
                "klt_inlier_threshold",
            ]
            ransac_properties = [
                "em_threshold",
                "em_confidence",
                "p3p_threshold",
                "p3p_confidence",
                "p3p_iterations",
                "p3p_ceres_refinement",
            ]
            visualization_properties = [
                "follow_camera",
            ]
            properties = [
                *camera_properties,
                *frontend_properties,
                *ransac_properties,
                *visualization_properties,
            ]
            values = itemgetter(*properties)(params)
            unspecified = [*map(lambda x: x is None, values)]
            if any(unspecified):
                unspecified_values = [v for v, u in zip(properties, unspecified) if u]
                raise ArgumentTypeError(
                    "{} does not specify the following required properties: {}".format(
                        path,
                        unspecified_values,
                    )
                )

            camera_params = {k: params[k] for k in camera_properties}
            frontend_params = {k: params[k] for k in frontend_properties}
            ransac_params = {k: params[k] for k in ransac_properties}
            visualization_params = {k: params[k] for k in visualization_properties}
            import_parameters(
                camera_params,
                frontend_params,
                ransac_params,
                visualization_params,
            )
        except IOError:
            raise ArgumentTypeError(f"{path} is not a valid path")
        except ValueError:
            raise ArgumentTypeError(f"{path} is not a valid json file.")

    parser = ArgumentParser()
    parser.add_argument(
        "-v",
        "--video_path",
        type=check_video,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--camera_params",
        type=check_params,
        required=True,
    )
    args = parser.parse_args()
    start(args.video_path)
