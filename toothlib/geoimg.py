# -*- coding: utf-8 -*-
'''
implement geometry image method
'''

import os
import numpy as np
from PIL import Image
from .config import config

def norm_dim(seq: np.array, imin: float, imax: float) -> np.array:
    # normalize all values in seq (min imin, max imax) to [0, 255]
    _norm = lambda x, imin, imax: int((x - imin) / (imax - imin) * 255)
    return np.array([_norm(_, imin, imax) for _ in seq])


# create geometry image
def create_RGB_GM(vertices: np.array, tag: str):
    # msh must be structured
    _shp = int(vertices.shape[0] ** 0.5)
    im = Image.new('RGB', [_shp, _shp])

    Xmin, Xmax = vertices[:, 0].min(), vertices[:, 0].max()
    _R = norm_dim(vertices[:, 0], Xmin, Xmax)
    Ymin, Ymax = vertices[:, 1].min(), vertices[:, 1].max()
    _G = norm_dim(vertices[:, 1], Ymin, Ymax)
    Zmin, Zmax = vertices[:, 2].min(), vertices[:, 2].max()
    _B = norm_dim(vertices[:, 2], Zmin, Zmax)

    # form values in _R, _G, _B as [(_r, _g, _b, 255), ... ]
    _RGB = [(_r, _g, _b, 255) for _r, _g, _b in zip(_R, _G, _B)]
    im.putdata(_RGB)
    im.save(os.path.join(
        'static',
        config['dataset']['processed']['geometry_image'],
        f'{tag}.png'
    ))
