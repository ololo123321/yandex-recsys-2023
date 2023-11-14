from typing import List, Tuple
import random
import numpy as np
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    def __init__(
            self,
            data: List[Tuple[int, str, List[int]]],
            p_catmix: float = 0.0,
            p_cutmix: float = 0.0,
            min_crop_size: int = 50,
            cache_emb: bool = False
    ):
        super().__init__()
        self.data = None
        if data is not None:
            self.set_data(data)
        self.p_catmix = p_catmix
        self.p_cutmix = p_cutmix
        self.min_crop_size = min_crop_size
        self.cache_emb = cache_emb

        self.index2emb = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        track, path, labels = self.data[index]
        emb = self._get_emb(index)

        # cutmix
        if random.random() < self.p_cutmix:
            # выбираем случайный спан размера [min_crop_size, sequence_length]
            n = emb.shape[0]
            j = random.randint(self.min_crop_size, max(n, self.min_crop_size))
            i = random.randint(0, max(0, j - self.min_crop_size))
            emb = emb[i:j]

        # catmix
        if random.random() < self.p_catmix:
            i = random.randint(0, len(self.data) - 1)
            _, path, labels_i = self.data[i]
            emb_i = self._get_emb(i)
            labels = sorted(set(labels + labels_i))
            if random.random() < 0.5:
                emb = np.concatenate([emb, emb_i], axis=0)
            else:
                emb = np.concatenate([emb_i, emb], axis=0)
        return emb, labels

    def _get_emb(self, index):
        _, path, _ = self.data[index]
        if self.cache_emb:
            emb = self.index2emb.setdefault(index, np.load(path))
        else:
            emb = np.load(path)
        return emb

    def set_data(self, data):
        self.data = data


class TrainingDatasetV2(Dataset):
    """
    new data layout. мб поможет избежать ликов памяти на даталоадере. UPD: не помогло
    """
    def __init__(self, data: List[Tuple[int, str, List[int]]], **kwargs):
        super().__init__()
        self.tracks = None
        self.paths = None
        self.labels = None
        self.indptr = None
        if data is not None:
            self.set_data(data)
        self.index2emb = {}

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index: int):
        emb = np.load(self.paths[index])
        labels = self.labels[self.indptr[index]:self.indptr[index + 1]]
        return emb, labels

    def set_data(self, data):
        self.tracks = np.array([x[0] for x in data])
        self.paths = np.array([x[1] for x in data])
        self.labels = []
        self.indptr = [0]
        for i in range(len(data)):
            self.labels += data[i][2]
            self.indptr.append(self.indptr[-1] + len(data[i][2]))
        self.labels = np.array(self.labels)
        self.indptr = np.array(self.indptr)


class InferenceDataset(Dataset):
    def __init__(self, data: List[str]):
        super().__init__()
        self.data = data
        self.index2emb = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        path = self.data[index]
        emb = self.index2emb.setdefault(index, np.load(path))
        return emb
