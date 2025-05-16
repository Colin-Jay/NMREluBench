
import numpy as np
import torch
from torch.utils.data import Dataset
import math

import lmdb
import os
import gzip
import pickle
from functools import lru_cache

class LMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            # self._keys = list(txn.cursor().iternext(values=False))
            self._keys = sorted(
                txn.cursor().iternext(values=False),
                key=lambda x: int(x.decode("ascii"))  
            )

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, 'env'):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)
        return data


def generate_nmr_spectrum(
    shifts, num_points=4000, shift_min=-20, shift_max=220, sigma=0.03
):
    x = np.linspace(shift_min, shift_max, num_points)
    spectrum = np.zeros_like(x)
    for c in shifts:
        spectrum += np.exp(-((x - c) ** 2) / (2 * sigma**2))
    if len(shifts) > 0 and np.max(spectrum) != np.min(spectrum):
        spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
    return spectrum.astype(np.float32)


class NMREluBenchDataset(Dataset):
    def __init__(self, db_path, use_h=False, use_c=True, num_points=4000):
        assert use_h or use_c, "You must enable at least one of use_h or use_c."

        self.dataset = LMDBDataset(db_path)
        self.use_h = use_h
        self.use_c = use_c
        self.num_points = num_points

        self.filtered_indices = []
        self.h_shifts_all = []
        self.c_shifts_all = []

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            smiles = sample.get("smiles", None)
            h_shifts = sample.get("nmr_1h", [])
            c_shifts = sample.get("nmr_13c", [])

            has_h = len(h_shifts) > 0
            has_c = len(c_shifts) > 0

            if (use_h and not has_h) or (use_c and not has_c) or (smiles is None):
                continue

            self.filtered_indices.append(idx)

            if use_h:
                self.h_shifts_all.append(max(h_shifts))
                self.h_shifts_all.append(min(h_shifts))
            if use_c:
                self.c_shifts_all.append(max(c_shifts))
                self.c_shifts_all.append(min(c_shifts))

        self.h_min = math.floor(min(self.h_shifts_all)) - 1 if self.h_shifts_all else -1.0
        self.h_max = math.ceil(max(self.h_shifts_all)) + 1 if self.h_shifts_all else 12.0
        self.c_min = math.floor(min(self.c_shifts_all)) - 5 if self.c_shifts_all else -20.0
        self.c_max = math.ceil(max(self.c_shifts_all)) + 5 if self.c_shifts_all else 220.0

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        real_idx = self.filtered_indices[idx]
        sample = self.dataset[real_idx]

        smiles = sample.get("smiles", None)
        h_shifts = sample.get("nmr_1h", []) if self.use_h else []
        c_shifts = sample.get("nmr_13c", []) if self.use_c else []

        h_spectrum = generate_nmr_spectrum(
            h_shifts, num_points=self.num_points, shift_min=self.h_min, shift_max=self.h_max, sigma=0.01
        ) if self.use_h else np.zeros(self.num_points, dtype=np.float32)

        c_spectrum = generate_nmr_spectrum(
            c_shifts, num_points=self.num_points, shift_min=self.c_min, shift_max=self.c_max, sigma=0.03
        ) if self.use_c else np.zeros(self.num_points, dtype=np.float32)

        ret = {}
        ret['smiles'] = smiles
        # ret['nmr_1h'] = torch.from_numpy(np.array(h_shifts)).float()
        # ret['nmr_13c'] = torch.from_numpy(np.array(c_shifts)).float()
        ret['nmr_1h_spec'] = torch.from_numpy(np.array(h_spectrum)).float()
        ret['nmr_13c_spec'] = torch.from_numpy(np.array(c_spectrum)).float()

        return ret

