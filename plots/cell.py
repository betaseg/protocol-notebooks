""" Helper classes to wrap a cellsketch project
"""

import z5py
from pathlib import Path
import os
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import os
import re
from pathlib import Path


class Cell:
    def __init__(self, path, linestyle='-'):
        self.file = z5py.File(str(path))
        self.name = Path(path).name.rstrip('.n5')
        self.path = path
        self.linestyle = linestyle

    def contains(self, volume_name):
        return self.name + '_' + volume_name in self.file

    def keys(self):
        matches = tuple(re.match(f'{self.name}_(.*)', k) for k in self.file.keys())
        return tuple(m.groups()[0] for m in matches if m is not None and len(m.groups()) > 0)

    def get_bounds(self):
        for cellbound in self.file.attrs["cellbounds"]:
            return cellbound

    def read_volume(self, volume_name):
        if volume_name is None:
            return None
        return np.array(self.file[(self.name + '_' + volume_name)])

    def read_volume_from_group(self, group, volume_name):
        if volume_name is None:
            return None
        return np.array(self.file[group][(self.name + '_' + volume_name)])

    def read_individual_table(self, labelmap_name):
        return self._read_table('analysis' + os.sep + self.name + '_' + labelmap_name + '_individual.csv')

    def read_overall_table(self, labelmap_name):
        return self._read_table('analysis' + os.sep + self.name + '_' + labelmap_name + '.csv')

    def _read_table(cell, table_name):
        return pd.read_table(os.path.join(cell.path, table_name), delimiter=',')

    def read_distance_map(self, name):
        return self.read_volume_from_group("analysis", name + "_distance_map")

    def get_masks(self):
        return self.get_data_type("masks")

    def get_labelmaps(self):
        return self.get_data_type("labelmaps")

    def get_filaments(self):
        return self.get_data_type("filaments")

    def get_data_type(self, data_type):
        res = []
        if data_type in self.file.attrs:
            for name, path in self.file.attrs[data_type].items():
                res.append(name)
        return res

    def compute_volumes(self):
        voxel = self.file.attrs['pixelToUM']**3
        all_volumes = {}

        # for masks, labelmaps and filaments
        for data_type in ['masks', 'labelmaps', 'filaments']:
            if data_type in self.file.attrs:
                for orga, path in self.file.attrs[data_type].items():
                    if self.get_bounds() + " border" in orga:
                        continue
                    v = np.count_nonzero(self.read_volume(orga))*voxel
                    all_volumes[orga] = v
        cellbounds = self.get_bounds()
        if cellbounds:
            vol_all = np.count_nonzero(self.read_volume(cellbounds))*voxel
        rows = []
        for orga, v in all_volumes.items():
            rows.append(dict(organelle=orga, volume=v, volume_fraction=v/vol_all if vol_all else 0))
        return pd.DataFrame.from_records(rows)
