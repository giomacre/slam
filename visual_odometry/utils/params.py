from typing import DefaultDict

camera_params = {}
frontend_params = {}
ransac_params = {}


def import_parameters(camera, frontend, ransac):
    camera_params.update(camera)
    frontend_params.update(frontend)
    ransac_params.update(ransac)
    print(camera_params)
    print(frontend_params)
    print(ransac_params)
