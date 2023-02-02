# %%
# -*- coding: utf-8 -*-

import trimesh
import numpy as np
import pandas as pd


msh = trimesh.load('static/CBCT/N5/face1.obj.remesh.obj')
msh_p4 = trimesh.load('static/CBCT/N5/face4.obj')


df = pd.read_csv('static/dataset/depth.csv')
depth_raw = df.iloc[0:10000, -1].to_numpy()


from matplotlib.colors import Normalize
import matplotlib.cm as cm

def generate_p2(depth_map_p1_raw: np.array, mesh0: trimesh.Trimesh, mesh1: trimesh.Trimesh) -> np.array:
    depth_map_p1 = depth_map_p1_raw.reshape(100, 100)
    # find the non-zero boundary of depth_map_p1(shape 100, 100)
    zero_boundary_index = []
    for row in range(100):
        nonz = np.nonzero(depth_map_p1[row, :])[0]
        if len(nonz) == 0:
            continue
        _min, _max = min(nonz), max(nonz)
        if _min > 0: zero_boundary_index.append(row * 100 + _min)
        if _max < 99: zero_boundary_index.append(row * 100 + _max)
    
    index_p2 = np.argwhere(depth_map_p1_raw == 0.)
    avg_norm = np.sum(mesh0.vertex_normals[zero_boundary_index], axis=0)
    avg_norm /= np.linalg.norm(avg_norm)
    vts_p2 = mesh0.vertices[index_p2]
    # vts_p2 shape(n, 1, 3), reshape to (n, 3)
    vts_p2 = vts_p2.reshape(-1, 3)
    # print(zero_boundary_index)

    _dirs = -np.array([avg_norm] * len(vts_p2))
    # print(vts_p2[1])
    # exit(-1)
    loc, index_ray, _ = mesh1.ray.intersects_location(ray_origins=vts_p2, ray_directions=_dirs)
    _new_oris = vts_p2[index_ray]
    _dists = np.linalg.norm(_new_oris - loc, axis=1)
    ray_real_ind = np.array(range(10000))[index_p2][index_ray]
    depth_map_p1_raw[ray_real_ind] = _dists.reshape(-1, 1)
    return depth_map_p1_raw.reshape(100, 100)

depth_raw = generate_p2(depth_raw, msh, msh_p4)

from copy import deepcopy

ray_origins = deepcopy(msh.vertices)

ray_directions = np.array([0, 1, 0] * len(ray_origins)).reshape(-1, 3)

# ray_directions shape (n, 3), depth_raw shape (n, ), multiply by row of them
ray_directions = ray_directions * depth_raw.reshape(-1, 1)
# ray_visualize = trimesh.load_path(np.hstack((ray_origins, ray_origins + ray_directions * 0.95)).reshape(-1, 2, 3))
# ray_visualize.colors = [[50, 50, 50, 50] for e in ray_visualize.entities]
ray_targets = ray_origins + ray_directions * 0.95
from copy import deepcopy
ray_new = deepcopy(ray_targets)

# build new mesh with vertices[ray_origins, ray_targets, ray_new], faces need to be modified
# concat ray_* to vertices, faces need to be modified
ray_vertices = np.vstack((ray_origins, ray_targets, ray_new))
len_ray = len(ray_origins)
ray_faces = np.array([[i, i+len_ray, i+len_ray*2] for i in range(len_ray)])
ray_msh = trimesh.Trimesh(vertices=ray_vertices, faces=ray_faces)
ray_msh.visual.vertex_colors = [50, 50, 50, 50]

new_msh = trimesh.Trimesh(vertices=ray_origins+ray_directions, faces=msh.faces)


def make_heatmap(data: np.array, vmin: float, vmax: float, style: str='YlGnBu') -> np.array:
    norm = Normalize()
    cmap = cm.get_cmap(style)
    return cmap(norm(data))

cmx = make_heatmap(depth_raw.flatten(), np.min(depth_raw), np.max(depth_raw)) * 255

new_msh.visual.vertex_colors = cmx
# scene = trimesh.Scene([msh, ray_msh, new_msh])

# scene.export('comb.ply')

msh.export('comb_p1.ply')
ray_msh.export('comb_p2.ply')
new_msh.export('comb_p3.ply')

# trimesh.load('comb.ply').show()
# combine two mesh [msh, new_msh]
# mm = trimesh.util.concatenate([msh, new_msh])
# mm.show()
# mm.export('test.obj')