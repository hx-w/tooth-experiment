# -*- coding: utf-8 -*-

import sys
sys.path.append('.')

import toothlib
import numpy as np
import trimesh
import seaborn as sns
import matplotlib.pyplot as plt

th = toothlib.ToothComp('CBCT', 'N5')

gt = th.get_gt()
def unify(msh, angle, direction):
    msh.apply_transform(trimesh.transformations.rotation_matrix(
        angle=angle, direction=direction, point=[0, 0, 0]))
    return msh

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


gt_all = generate_p2(gt.flatten(), th.meshes[0].m, th.meshes[3].m)


def show_heatmap(data: np.array, vmin: float, vmax: float, style: str='YlGnBu') -> np.array:
    # rotate 180
    data = np.rot90(data, 2)
    sns.heatmap(data, vmin=vmin, vmax=vmax, cmap=style)
    plt.show()


from matplotlib.colors import Normalize
import matplotlib.cm as cm
def make_heatmap(data: np.array, vmin: float, vmax: float, style: str='YlGnBu') -> np.array:
    norm = Normalize()
    cmap = cm.get_cmap(style)
    return cmap(norm(data))



show_heatmap(gt_all, np.min(gt_all), np.max(gt_all))
# show_heatmap(gt, np.min(gt_all), np.max(gt_all))
gt_p2 = np.where(gt == 0., gt_all, 0)
# show_heatmap(gt_p2, np.min(gt_all), np.max(gt_all))

cmx = make_heatmap(gt_all.flatten(), np.min(gt_all), np.max(gt_all)) * 255

th.meshes[0].m.vertices += 300

th.meshes[0].m.visual.vertex_colors = cmx
# unify(th.meshes[0].m, -1.6, [0.7, -0.2, -0.10]).show()
