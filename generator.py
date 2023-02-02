# -*- coding: utf-8 -*-
'''
generate dataset from raw data(CBCT)
save to:
    - csv: config['dataset']['processed']['depth_csv']
    - features: config['dataset']['processed']['features']
    - ground truth: config['dataset']['processed']['gt']
'''

import os
import pandas as pd
from tqdm import tqdm
import toothlib

from typing import Tuple

def generate_from_toothcomps(tooth_comps: toothlib.ToothComp) -> Tuple[pd.DataFrame]:
    feat_names = ['u', 'v', 'curv', 'metric_u', 'metric_v']
    depth_df, edge_df = pd.DataFrame(), pd.DataFrame()
    for toc in tqdm(tooth_comps, desc='generating'):
        gt = toc.get_gt()
        _depth_df = toc.get_features(feat_names, as_df=True)
        # reshape each col of _depth_df to 100,100 and calc mean of each col
        _raw_depth_df = _depth_df.drop(columns=['u', 'metric_v'])
        _edge_df = _raw_depth_df.apply(lambda x: x.values.reshape(gt.shape).mean(axis=0))
        _edge_df['gt'] = [gt[:, i].nonzero()[0][0] for i in range(gt.shape[1])]
        _depth_df['gt'] = gt.flatten()
        depth_df = pd.concat([depth_df, _depth_df])
        edge_df = pd.concat([edge_df, _edge_df])

    return depth_df, edge_df

def generate_all_CBCT():
    depth_csv_path = os.path.join(
        'static',
        toothlib.config['dataset']['processed']['depth_csv']
    )
    edge_csv_path = os.path.join(
        'static',
        toothlib.config['dataset']['processed']['edge_csv']
    )

    _all = toothlib.config['dataset']['raw']['samples'][:15]
    _invalid = toothlib.config['dataset']['raw']['invalid_samples']
    tooth_comps = [
        toothlib.ToothComp('CBCT', tag)
        for tag in _all if tag not in _invalid
    ]

    depth_df, edge_df = generate_from_toothcomps(tooth_comps)
    depth_df.to_csv(depth_csv_path, index=False)
    edge_df.to_csv(edge_csv_path, index=False)

if __name__ == '__main__':
    generate_all_CBCT()


