import numpy as np
import trimesh

# attach to logger so trimesh messages will be printed to console
# trimesh.util.attach_to_log()

# mesh = trimesh.load('./quadcopter_large.stl')
mesh = trimesh.load('./quadcopter.stl')

# is the current mesh watertight?
mesh.is_watertight

# what's the euler number for the mesh?
mesh.euler_number

# facets are groups of coplanar adjacent faces
# set each facet to a random color
# colors are 8 bit RGBA by default (n, 4) np.uint8
red = np.array([255, 0, 0, 255], dtype=np.uint8)
green = np.array([0, 255, 0, 255], dtype=np.uint8)
blue = np.array([0, 0, 255, 255], dtype=np.uint8)
color = np.array([73,  73,   73, 255], dtype=np.uint8)
for i, facet in enumerate(mesh.facets):
    if i < 30:
        mesh.visual.face_colors[facet] = red
    elif i < 42:
        mesh.visual.face_colors[facet] = green
    elif i < 54:
        mesh.visual.face_colors[facet] = blue
    elif i < 66:
        mesh.visual.face_colors[facet] = red
    else:
        mesh.visual.face_colors[facet] = color

    # red[0] = int(1.5 * i)
    mesh.visual.face_colors[facet] = color

    # color = trimesh.visual.random_color()

    # if i == 0:
    #     mesh.visual.face_colors[facet] = red
    # else:
    #     mesh.visual.face_colors[facet] = color

# preview mesh in an opengl window if you installed pyglet with pip
# import ipdb; ipdb.set_trace()
mesh.show()
