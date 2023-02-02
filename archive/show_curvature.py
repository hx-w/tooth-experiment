# -*- coding: utf-8 -*-

import trimesh

import sys
sys.path.append('.')

import toothlib

from trimesh.curvature import (
    discrete_mean_curvature_measure,
    sphere_ball_intersection
)

def unify(msh, angle, direction):
    msh.apply_transform(trimesh.transformations.rotation_matrix(
        angle=angle, direction=direction, point=[0, 0, 0]))
    return msh

th = toothlib.ToothComp('CBCT', 'N5')

curv = th.get_features(['curv'])[0].flatten()

uns_msh = trimesh.load('static/CBCT/N5/face1.obj')

curv_map = discrete_mean_curvature_measure(
    uns_msh,
    uns_msh.vertices,
    1e-3
)

curv_map /= sphere_ball_intersection(1, 1e-3)


# cmap unify
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np

style = 'YlGnBu'
def make_heatmap(data: np.array, vmin: float, vmax: float, style: str='YlGnBu') -> np.array:
    norm = Normalize()
    cmap = cm.get_cmap(style)
    return cmap(norm(data))


vmin = np.min([np.min(curv_map), np.min(curv)])
vmax = np.max([np.max(curv_map), np.max(curv)])

cmap_str = make_heatmap(curv, vmin, vmax, style)
cmap_uns = make_heatmap(curv_map, vmin, vmax, style)

uns_msh.visual.vertex_colors = cmap_uns
th.meshes[0].m.visual.vertex_colors = cmap_str
# uns_msh.show()

uns_msh.vertices += 300
th.meshes[0].m.vertices += 300

unify(uns_msh, -1.6, [0.7, -0.15, -0.10]).show()
unify(th.meshes[0].m, -1.6, [0.7, -0.15, -0.10]).show()

print(vmin, vmax)


