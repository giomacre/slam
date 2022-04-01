from functools import partial
import os
import sys
from utils.worker import create_worker
from utils.decorators import ddict, stateful_decorator
import numpy as np

sys.path.append(
    os.path.join(
        os.path.dirname(sys.argv[0]),
        "lib",
    )
)
import pypangolin as pango
import OpenGL.GL as gl


def create_map_thread(windows_size, Kinv, video_size, thread_context):
    render_context = ddict(
        render_state=None,
        display=None,
    )
    setup_window = partial(setup_pangolin, *windows_size, render_context)

    decorator = stateful_decorator(needs=1)
    visualization_loop = decorator(
        partial(
            draw_map,
            render_context,
            Kinv,
            windows_size,
        )
    )
    worker = create_worker(
        visualization_loop,
        thread_context,
        one_shot=setup_window,
        timeout=16e-3,
        empty_queue_handler=lambda _: visualization_loop(),
        name="PyPangolinViewer",
    )

    def prepare_task(frames, points):
        worker(
            [f.pose for f in frames],
            [(p.coords, p.color) for p in points],
        )

    return prepare_task


def draw_map(
    context,
    Kinv,
    video_size,
    poses,
    points,
):
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    context.display.Activate(context.render_state)
    gl.glColor(1.0, 0.85, 0.3)
    for pose in poses[:-1]:
        pango.glDrawFrustum(
            Kinv,
            *video_size,
            pose,
            0.2,
        )
    pango.glDrawLineStrip(
        [
            *map(
                lambda p: p[:3, 3:],
                poses,
            )
        ]
    )
    gl.glLineWidth(2)
    gl.glColor(0.4, 0.0, 1.0)
    pango.glDrawFrustum(
        Kinv,
        *video_size,
        poses[-1],
        1,
    )
    gl.glPointSize(2)
    for pt, color in points:
        gl.glColor(*color)
        pango.glDrawPoints([pt])

    pango.FinishFrame()


def setup_pangolin(
    width,
    height,
    context,
    focal_length=500,
    z_near=0.1,
    z_far=1000,
    camera_pos=[15.0, -20.0, -25.0],
    target=[0.0, 0.5, 0.5],
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
