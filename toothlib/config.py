# -*- coding: utf-8 -*-

import yaml

class Config:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config = yaml.load(f, yaml.SafeLoader)
        self.config_file = config_file

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def save(self):
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)

    def __repr__(self):
        return str(self.config)

# singleton
try:
    config = Config('config.yml')
except Exception as err:
    raise err