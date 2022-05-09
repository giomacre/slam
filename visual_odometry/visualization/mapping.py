from functools import partial
from itertools import chain, islice
from operator import itemgetter
from ..utils.worker import create_worker
from ..utils.params import frontend_params, visualization_params
import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(sys.argv[0]),
        "lib",
    )
)
import pypangolin as pango
import OpenGL.GL as gl


def create_map_thread(windows_size, Kinv, thread_context):
    context = dict(
        render_state=None,
        display=None,
        poses=[],
        true_positions=[],
        positions=[],
        map_points={"coords": [], "colors": []},
    )
    render_context = partial(
        itemgetter(*iter(context.keys())),
        context,
    )
    setup_window = partial(
        setup_pangolin,
        *windows_size,
        context,
    )
    visualization_loop = partial(
        draw_map,
        render_context,
        Kinv,
        windows_size,
    )
    worker = create_worker(
        visualization_loop,
        thread_context,
        one_shot=setup_window,
        timeout=16e-3,
        empty_queue_handler=lambda _: visualization_loop(),
        name="PyPangolinViewer",
    )

    # @performance_timer()
    def prepare_task(frames, new_points, ground_truth):
        last_frame = frames[-1]
        map_points = [(p.coords, p.color) for p in new_points[::5]]
        return worker(
            last_frame.pose,
            ground_truth[last_frame.id] if last_frame.id < len(ground_truth) else None,
            map_points,
        )

    return prepare_task


def draw_map(
    render_context,
    Kinv,
    video_size,
    pose=None,
    ground_truth=None,
    new_points=None,
):
    (
        render_state,
        display,
        poses,
        true_trajectory,
        positions,
        map_points,
    ) = render_context()
    if pose is not None:
        poses += [pose]
        if ground_truth is not None:
            true_trajectory += [ground_truth[:3, 3:]]
        positions += [pose[:3, 3:]]
        if len(new_points) > 0:
            if visualization_params["follow_camera"]:
                render_state.Follow(pango.OpenGlMatrix(pose), True)
            new_coords, new_colors = zip(*new_points)
            map_points["coords"].extend(new_coords)
            map_points["colors"].extend(new_colors)

    if len(poses) == 0:
        return
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    display.Activate(render_state)
    gl.glLineWidth(3)
    gl.glColor(1.0, 0.85, 0.3)
    pango.glDrawLineStrip(positions)
    gl.glColor(0.0, 0.65, 0.1)
    pango.glDrawLineStrip(true_trajectory)
    gl.glLineWidth(3)
    gl.glColor(0.4, 0.0, 1.0)
    pango.glDrawFrustum(
        Kinv,
        *video_size,
        poses[-1],
        frontend_params["epipolar_scale"],
    )
    if len(map_points["coords"]) > 0:
        gl.glPointSize(1)
        pango.DrawPoints(
            map_points["coords"],
            map_points["colors"],
        )
    pango.FinishFrame()


def setup_pangolin(
    width,
    height,
    context,
    focal_length=750,
    z_near=0.001,
    z_far=10000,
    camera_pos=[0, -1, -15],
    target=[0.0, 0.0, 1.5],
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
    context["render_state"] = render_state
    context["display"] = display
