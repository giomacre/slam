#!/usr/bin/env python3
#%%
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
os.environ["LD_LIBRARY_PATH"] = os.getcwd() + "lib/"
import pypangolin as pango
from threading import Thread
from OpenGL.GL import *


def setup_pangolin(
    width,
    height,
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
    render_state = pango.OpenGlRenderState(projection_matrix, model_view)
    handler = pango.Handler3D(render_state)
    display = pango.CreateDisplay().SetAspect(width / height).SetHandler(handler)

    return render_state, display


def render(width, height):
    render_state, display = setup_pangolin(width, height)
    while not pango.ShouldQuit():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        display.Activate(render_state)
        pango.glDrawColouredCube()
        pango.FinishFrame()


if __name__ == "__main__":
    render_loop = Thread(target=render, args=(1280, 720))
    render_loop.start()
    render_loop.join()

# %%
