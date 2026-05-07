"""YAML configuration loader used by detector/tracker entrypoints."""

import os
import yaml
from easydict import EasyDict as edict


class YamlParser(edict):
    """EasyDict wrapper that can merge one or more YAML config files."""

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert (os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.safe_load(fo.read()) or {})

        super(YamlParser, self).__init__(cfg_dict)


    def merge_from_file(self, config_file):
        """Merge values from a YAML file into the current config."""
        with open(config_file, 'r') as fo:
            self.update(yaml.safe_load(fo.read()) or {})


    def merge_from_dict(self, config_dict):
        """Merge values from a dictionary into the current config."""
        self.update(config_dict)


def get_config(config_file=None):
    """Create a config object, optionally preloaded from YAML."""
    return YamlParser(config_file=config_file)
