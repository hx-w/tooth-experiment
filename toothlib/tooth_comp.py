# -*- coding: utf-8 -*-

'''
Tooth component is formed with 4 faces(CBCT) or 2 faces(IOS)
and face1's remeshed mesh `face1.obj.remesh.obj`
providing methods for
    - get_features [u, v, curv, metric_u, metric_v]
    - get_gt (ground truth)
    - get_geometry_image
'''

import os
from typing import NoReturn, NewType, List

import trimesh
from trimesh.curvature import (
    discrete_gaussian_curvature_measure,
    discrete_mean_curvature_measure,
    sphere_ball_intersection
)

import numpy as np
import pandas as pd

from .config import config
from .geoimg import create_RGB_GM

MESH_ROOT = 'static/'
MeshId = NewType('MeshId', int)


def tooth_confg_check(sample_type: str, sample_tag: str) -> NoReturn:
    '''
    check if the sample_type and sample_tag are in config
    '''
    assert sample_type in config['dataset']['raw']['sample_types'].keys()
    _all_samples = config['dataset']['raw']['samples']
    _inv_samples = config['dataset']['raw']['invalid_samples']
    assert sample_tag in _all_samples and sample_tag not in _inv_samples


class Mesh:
    '''trimesh type
    '''

    def __init__(self, mesh_path: str):
        self.m = self.__load_mesh(mesh_path)
        self.path = mesh_path
        self.is_remeshed = False

    def __load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        msh = trimesh.load_mesh(mesh_path)
        msh.fix_normals()
        msh.remove_duplicate_faces()
        return msh

    def __get__(self):
        return self.m

    def remesh(self) -> NoReturn:
        __str_msh_path = f'{self.path}{config["dataset"]["raw"]["remesh_suffix"]}'
        self.m = self.__load_mesh(__str_msh_path)
        self.is_remeshed = True
        _sz = int(self.m.vertices.shape[0] ** 0.5)
        self.topo_shape = (_sz, _sz)

    def feature_index(self, idx_type: str) -> np.array:
        '''feature: index ['U', 'V']; shape(n, n)
        '''
        _sap = self.topo_shape[0]
        if 'U' == idx_type:
            feat = np.repeat(np.arange(_sap), _sap)
        elif 'V' == idx_type:
            feat = np.tile(np.arange(_sap), _sap)
        else:
            raise ValueError('index_type must be `U` or `V`')
        return (feat / _sap).reshape(self.topo_shape)

    def feature_curv(self, curv_type: str) -> np.array:
        '''feature: curvature['gaussian', 'mean']; shape(n, n)
        '''
        radius = 1e-3
        if curv_type == 'mean':
            curv_map = discrete_mean_curvature_measure(
                self.m, self.m.vertices, radius)
        elif curv_type == 'gaussian':
            curv_map = discrete_gaussian_curvature_measure(
                self.m, self.m.vertices, radius)
        else:
            raise ValueError('curv_type must be `mean` or `gaussian`')
        curv_map /= sphere_ball_intersection(1, radius)
        return curv_map.reshape(self.topo_shape)

    def feature_metric(self, metric_type: str) -> np.array:
        '''feature: metric['U', 'V']; shape(n, n)
        '''
        _shape_0 = self.topo_shape[0]

        def __metric_center(v: int) -> float:
            _theta = 1 if metric_type == 'U' else _shape_0
            if (
                metric_type == 'U' and v % _shape_0 == 0 or
                metric_type == 'V' and v < _shape_0
            ):
                neighbors = [v + _theta]
            elif (
                metric_type == 'U' and v % _shape_0 == _shape_0 - 1 or
                metric_type == 'V' and v >= _shape_0 * (_shape_0 - 1)
            ):
                neighbors = [v - _theta]
            else:
                neighbors = [v - _theta, v + _theta]
            # compute metric(distance) between v and its neighbors
            _metric = np.mean(
                [np.linalg.norm(self.m.vertices[v] - self.m.vertices[_])
                 for _ in neighbors]
            )
            return _metric

        metric_map = np.array([__metric_center(_)
                              for _ in range(_shape_0 ** 2)])
        return metric_map.reshape(self.topo_shape)


class ToothComp:
    def __init__(self, sample_type: str, sample_tag: str):
        '''
        @param sample_type: 'CBCT' or 'IOS'
        @param sample_tag: 'N1', 'N2' or 'N{number}'
        '''
        tooth_confg_check(sample_type, sample_tag)
        self.sample_type = sample_type
        self.sample_tag = sample_tag
        # load meshes & remesh mesh[0]
        _basic_file = config['dataset']['raw']['sample_types'][sample_type]
        _sample_root = os.path.join(MESH_ROOT, sample_type)
        self.meshes = [
            Mesh(os.path.join(_sample_root, sample_tag, _f))
            for _f in _basic_file
        ]
        self.meshes[0].remesh()
        # may not be needed to calculate the rotation
        self.__determine_rot()

    def __determine_rot(self):
        '''features & depth will be output as a matrix,
        but the matrix direction is unordered, we want to unify the direction
        '''
        self._rotate_times = 0  # may not be needed
        return
        __thred = 1e-3
        f14_ndists = self.__compute_nearest(0, 3)
        f14_ndists = np.where(f14_ndists < __thred, 0., f14_ndists)
        self._rotate_times = np.argmin([
            len(np.nonzero(f14_ndists[:, -1])[0]),  # t0
            len(np.nonzero(f14_ndists[-1, :])[0]),  # t90
            len(np.nonzero(f14_ndists[:, 0])[0]),  # t180
            len(np.nonzero(f14_ndists[0, :])[0]),  # t270
        ])

    def __compute_nearest(self, source_id: MeshId, target_id: MeshId) -> np.array:
        '''compute the nearest distance for each vertices in source_id mesh to target
        '''
        source_vertices = self.meshes[source_id].m.vertices
        ndists = self.meshes[target_id].m.nearest.signed_distance(
            source_vertices)
        # in exp, only mesh-0 will be remeshed
        return np.absolute(ndists).reshape(self.meshes[0].topo_shape)

    def get_features(self, features: List[str], as_df: bool = False) -> List[np.array]:
        '''
        @param features: ['u', 'v', 'curv', 'metric_u', 'metric_v']
        '''
        feats_map = {
            'u': (self.meshes[0].feature_index, 'U'),
            'v': (self.meshes[0].feature_index, 'V'),
            'curv': (self.meshes[0].feature_curv, 'mean'),
            'metric_u': (self.meshes[0].feature_metric, 'U'),
            'metric_v': (self.meshes[0].feature_metric, 'V')
        }
        feats = list(map(lambda feat: feats_map[feat][0](feats_map[feat][1]), features))
        if as_df:
            feats = list(map(lambda feat: feat.flatten(), feats))
            df = pd.DataFrame(dict(zip(features, feats)))
            return df
        return feats

    def get_gt(self, thr: float = 0.3) -> np.array:
        # get ground truth
        f2_ndists = self.__compute_nearest(0, 1)
        f3_ndists = self.__compute_nearest(0, 2)
        diff_f3_f2 = np.abs(f3_ndists - f2_ndists)

        depth_map = np.where(diff_f3_f2 > thr, f2_ndists, 0.)
        return depth_map

    def save_geoimg(self) -> NoReturn:
        create_RGB_GM(
            self.meshes[0].m.vertices,
            f'{self.sample_type}_{self.sample_tag}'
        )
