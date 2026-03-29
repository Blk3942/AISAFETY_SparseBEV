# nuScenes database loader — offline subset for detection evaluation only.
# Derived from nuscenes/nuscenes.py (Apache-2.0, Motional): tables + get + box_velocity + reverse index.

import json
import os.path as osp
import time
from typing import List

import numpy as np


class NuScenes:
    """Minimal NuScenes: loads JSON tables only (no maps, no lidarseg, no explorer)."""

    def __init__(self, version: str = 'v1.0-mini', dataroot: str = '/data/sets/nuscenes', verbose: bool = False):
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.table_names = [
            'category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
            'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map',
        ]

        assert osp.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)

        t0 = time.time()
        if verbose:
            print('======\nLoading NuScenes tables for version {}...'.format(self.version))

        self.category = self.__load_table__('category')
        self.attribute = self.__load_table__('attribute')
        self.visibility = self.__load_table__('visibility')
        self.instance = self.__load_table__('instance')
        self.sensor = self.__load_table__('sensor')
        self.calibrated_sensor = self.__load_table__('calibrated_sensor')
        self.ego_pose = self.__load_table__('ego_pose')
        self.log = self.__load_table__('log')
        self.scene = self.__load_table__('scene')
        self.sample = self.__load_table__('sample')
        self.sample_data = self.__load_table__('sample_data')
        self.sample_annotation = self.__load_table__('sample_annotation')
        self.map = self.__load_table__('map')

        if verbose:
            for table in self.table_names:
                print('{} {},'.format(len(getattr(self, table)), table))
            print('Done loading in {:.3f} seconds.\n======'.format(time.time() - t0))

        self.__make_reverse_index__(verbose)

        # Explorer not used by evaluation.
        self.explorer = None

    @property
    def table_root(self) -> str:
        return osp.join(self.dataroot, self.version)

    def __load_table__(self, table_name) -> list:
        with open(osp.join(self.table_root, '{}.json'.format(table_name))) as f:
            return json.load(f)

    def __make_reverse_index__(self, verbose: bool) -> None:
        start_time = time.time()
        if verbose:
            print('Reverse indexing ...')

        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()
            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member['token']] = ind

        for record in self.sample_annotation:
            inst = self.get('instance', record['instance_token'])
            record['category_name'] = self.get('category', inst['category_token'])['name']

        for record in self.sample_data:
            cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self.get('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']

        for record in self.sample:
            record['data'] = {}
            record['anns'] = []

        for record in self.sample_data:
            if record['is_key_frame']:
                sample_record = self.get('sample', record['sample_token'])
                sample_record['data'][record['channel']] = record['token']

        for ann_record in self.sample_annotation:
            sample_record = self.get('sample', ann_record['sample_token'])
            sample_record['anns'].append(ann_record['token'])

        if 'log_tokens' not in self.map[0].keys():
            raise Exception('Error: log_tokens not in map table.')
        log_to_map = dict()
        for map_record in self.map:
            for log_token in map_record['log_tokens']:
                log_to_map[log_token] = map_record['token']
        for log_record in self.log:
            log_record['map_token'] = log_to_map[log_record['token']]

        if verbose:
            print('Done reverse indexing in {:.1f} seconds.\n======'.format(time.time() - start_time))

    def get(self, table_name: str, token: str) -> dict:
        assert table_name in self.table_names, 'Table {} not found'.format(table_name)
        return getattr(self, table_name)[self.getind(table_name, token)]

    def getind(self, table_name: str, token: str) -> int:
        return self._token2ind[table_name][token]

    def field2token(self, table_name: str, field: str, query) -> List[str]:
        matches = []
        for member in getattr(self, table_name):
            if member[field] == query:
                matches.append(member['token'])
        return matches

    def box_velocity(self, sample_annotation_token: str, max_time_diff: float = 1.5) -> np.ndarray:
        current = self.get('sample_annotation', sample_annotation_token)
        has_prev = current['prev'] != ''
        has_next = current['next'] != ''

        if not has_prev and not has_next:
            return np.array([np.nan, np.nan, np.nan])

        if has_prev:
            first = self.get('sample_annotation', current['prev'])
        else:
            first = current

        if has_next:
            last = self.get('sample_annotation', current['next'])
        else:
            last = current

        pos_last = np.array(last['translation'])
        pos_first = np.array(first['translation'])
        pos_diff = pos_last - pos_first

        time_last = 1e-6 * self.get('sample', last['sample_token'])['timestamp']
        time_first = 1e-6 * self.get('sample', first['sample_token'])['timestamp']
        time_diff = time_last - time_first

        if has_next and has_prev:
            max_time_diff *= 2

        if time_diff > max_time_diff:
            return np.array([np.nan, np.nan, np.nan])
        return pos_diff / time_diff
