camera_params = {}
frontend_params = {}
ransac_params = {}
visualization_params = {}


def import_parameters(camera, frontend, ransac, visualization):
    camera_params.update(camera)
    frontend_params.update(frontend)
    ransac_params.update(ransac)
    visualization_params.update(visualization)
    print(camera_params)
    print(frontend_params)
    print(ransac_params)
