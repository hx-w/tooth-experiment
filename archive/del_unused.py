# -*- coding: utf-8 -*-

import os

# walk through the directory: static/CBCT/
def del_unused(dir: str) -> None:
    for root, dirs, files in os.walk(dir):
        if 'N' not in root: continue
        invalid = list(filter(lambda fn: 'remesh_' in fn, files))
        if '.DS_Store' in files:
            invalid.append('.DS_Store')
        
        rmfile = lambda fn: os.remove(os.path.join(root, fn))
        list(map(rmfile, invalid))

del_unused('static/CBCT/')
del_unused('static/IOS/')