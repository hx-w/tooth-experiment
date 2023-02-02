# -*- coding: utf-8 -*-

import trimesh
import numpy as np

_type = 'IOS/N5'

uns_msh = trimesh.load(f'static/{_type}/face1.obj')
param_msh = trimesh.load(f'static/{_type}/face1.obj.param.obj')
str_msh = trimesh.load(f'static/{_type}/face1.obj.remesh.obj')

# rotate the mesh down

uns_msh.apply_transform(trimesh.transformations.rotation_matrix(
    angle=-1.6, direction=[0.7, -0.15, -0.10], point=[0, 0, 0]))

# uns_msh.show()
sample_nums = 10
scale = 2

sample_pnts = []
for ic in range(sample_nums):
    for ir in range(sample_nums):
        # sample_pnts.append([scale * ic / (sample_nums - 1) - scale / 2,scale * ir / (sample_nums - 1) - scale / 2, 0.])
        sample_pnts.append([-scale * ir / (sample_nums - 1) + scale / 2, -scale * ic / (sample_nums - 1) + scale / 2,  0.])

half_trias1 = [
    [ir * sample_nums + ic, ir * sample_nums + ic - sample_nums, ir * sample_nums + ic - 1]
    for ir in range(1, sample_nums) for ic in range(1, sample_nums)
]
half_trias2 = [
    [ir * sample_nums + ic - 1, ir * sample_nums + ic - sample_nums, ir * sample_nums + ic - sample_nums - 1]
    for ir in range(1, sample_nums) for ic in range(1, sample_nums)
]

sample_msh = trimesh.Trimesh(
    vertices=sample_pnts,
    faces= np.vstack((half_trias1, half_trias2))
)

sample_msh.visual.vertex_colors = [255, 0, 0, 255]
# light grey color
param_msh.visual.vertex_colors = [150, 150, 150, 255]

# sample_msh.show()
# param_msh.show()
# trimesh.Scene([sample_msh, param_msh]).show()

# str_msh.show()
str_msh.apply_transform(trimesh.transformations.rotation_matrix(
    angle=-1.6, direction=[0.7, -0.15, -0.10], point=[0, 0, 0]))
# uns_msh.apply_transform(trimesh.transformations.rotation_matrix(
    # angle=-1.6, direction=[0.7, -0.15, -0.10], point=[0, 0, 0]))

uns_msh.visual.vertex_colors = [100, 100, 100, 255]
str_msh.visual.vertex_colors = [255, 0, 0, 255]

# line width

str_msh.export('archive/remesh_inst/IOS_N5_str.ply')
uns_msh.export('archive/remesh_inst/IOS_N5_uns.ply')

str_msh = trimesh.load('archive/remesh_inst/IOS_N5_str.ply')
uns_msh = trimesh.load('archive/remesh_inst/IOS_N5_uns.ply')

trimesh.Scene([uns_msh, str_msh]).show()

# str_msh.show()