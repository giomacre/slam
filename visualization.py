from functools import partial
import os
import sys
from worker import create_worker
from decorators import ddict, stateful_decorator
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
import pypangolin as pango
import OpenGL.GL as gl


def setup_pangolin(
    width,
    height,
    context,
    focal_length=500,
    z_near=0.1,
    z_far=1000,
    camera_pos=[0.0, 0.5, -3.0],
    target=[0.0] * 3,
    up_direction=[0.0, -1.0, 0.0],
):
    pango.CreateWindowAndBind("", width, height)
    gl.glEnable(gl.GL_DEPTH_TEST)
    projection_matrix = pango.ProjectionMatrixRDF_TopLeft(
        width,
        height,
        focal_length,
        focal_length,
        width // 2,
        height // 2,
        z_near,
        z_far,
    )
    model_view = pango.ModelViewLookAtRDF(
        *camera_pos,
        *target,
        *up_direction,
    )
    render_state = pango.OpenGlRenderState(
        projection_matrix,
        model_view,
    )
    handler = pango.Handler3D(render_state)
    display = pango.CreateDisplay().SetAspect(width / height).SetHandler(handler)
    context.render_state = render_state
    context.display = display


def create_map_thread(width, height, thread_context):
    render_context = ddict(
        render_state=None,
        display=None,
    )
    setup_window = partial(setup_pangolin, width, height, render_context)

    def draw_cube(
        context,
        Kinv,
        poses,
        points,
    ):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        context.display.Activate(context.render_state)
        gl.glLineWidth(2)
        gl.glColor(1.0, 1.0, 0.0)
        for pose in poses:
            pango.glDrawFrustum(
                Kinv[:3, :3],
                width,
                height,
                np.linalg.inv(pose),
                0.5,
            )
        gl.glPointSize(2)
        gl.glColor(1.0, 0.0, 0.0)
        pango.glDrawPoints(points[:, :3])
        pango.FinishFrame()

    decorator = stateful_decorator(
        keep=1,
        append_empty=False,
    )
    visualization_loop = decorator(
        partial(
            draw_cube,
            render_context,
        )
    )
    return create_worker(
        visualization_loop,
        thread_context,
        one_shot=setup_window,
    )
