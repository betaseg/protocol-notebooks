import os
from pathlib import Path

import numpy as np
import pandas
import z5py


class Cell:
    def __init__(self, path, linestyle='-'):
        self.file = z5py.File(str(path))
        self.name = Path(path).name.rstrip('.n5')
        self.path = path
        self.linestyle = linestyle

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
        return pandas.read_table(os.path.join(cell.path, table_name), delimiter=',')

    def read_distance_map(self, name):
        return self.read_volume_from_group("analysis", name + "_distance_map")
