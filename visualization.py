from functools import partial
import os
import sys
from worker import create_worker_process
from decorators import ddict

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
import pypangolin as pango
from OpenGL.GL import *


def setup_pangolin(
    width,
    height,
    context,
    focal_length=500,
    z_near=0.1,
    z_far=1000,
    camera_pos=[0.0, 0.5, -3.0],
    target=[0.0] * 3,
    up_direction=pango.AxisY,
):
    pango.CreateWindowAndBind("", width, height)
    glEnable(GL_DEPTH_TEST)
    projection_matrix = pango.ProjectionMatrix(
        width,
        height,
        focal_length,
        focal_length,
        width // 2,
        height // 2,
        z_near,
        z_far,
    )
    model_view = pango.ModelViewLookAt(
        *camera_pos,
        *target,
        up_direction,
    )
    render_state = pango.OpenGlRenderState(
        projection_matrix,
        model_view,
    )
    handler = pango.Handler3D(render_state)
    display = pango.CreateDisplay().SetAspect(width / height).SetHandler(handler)
    context.render_state = render_state
    context.display = display


def create_3d_visualization_process(width, height):
    context = ddict(
        {
            "render_state": None,
            "display": None,
        }
    )
    setup = partial(setup_pangolin, width, height, context)

    def draw_cube(context, _):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        context.display.Activate(context.render_state)
        pango.glDrawColouredCube()
        pango.FinishFrame()

    visualization_loop = partial(draw_cube, context)
    return create_worker_process(
        visualization_loop,
        one_shot=setup,
        terminate=pango.ShouldQuit,
    )
