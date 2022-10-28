from abc import ABC, abstractmethod

import faiss
import numpy as np
import torch


class NumpyIndexL2:
    def __init__(self, dim, batch_size=16):
        self._batch_size = batch_size
        self.reset()

    def reset(self):
        self._items = None

    def add(self, items):
        if self._items is not None:
            items = np.concatenate((self._items, items), 0)
        self._items = items

    def search(self, queries, k):
        if k < 1:
            raise ValueError("Expected k > 0, got {}.".format(k))
        if (self._items is None) or (len(self._items) == 0):
            raise RuntimeError("Empty index")
        if len(self._items) == 1:
            indices = np.zeros(len(queries), dtype=np.int64)
            distances = np.linalg.norm(queries - self._items[0], axis=1)
        else:
            k = min(k, len(self._items))  # [1, I].
            indices = []
            distances = []
            for i in range(0, len(queries), self._batch_size):
                batch = queries[i:i + self._batch_size]
                scores = np.linalg.norm(batch[:, None, :] - self._items[None, :, :], axis=2)  # (Q, I).
                if k == 1:
                    indices.append(np.argmin(scores, axis=1)[:, None])
                else:
                    indices.append(np.argpartition(scores, (1, k - 1), axis=1)[:, :k])
                distances.append(np.take_along_axis(scores, indices[-1], 1))
            indices = np.concatenate(indices, 0)
            distances = np.concatenate(distances, 0)
        return distances, indices


class TorchIndexL2:
    def __init__(self, dim, batch_size=16):
        self._batch_size = batch_size
        self.reset()

    def reset(self):
        self._items = None

    def add(self, items):
        items = self._from_numpy(items)
        if self._items is not None:
            items = torch.cat((self._items, items), 0)
        self._items = items

    def search(self, queries, k):
        queries = self._from_numpy(queries)
        if k < 1:
            raise ValueError("Expected k > 0, got {}.".format(k))
        if (self._items is None) or (len(self._items) == 0):
            raise RuntimeError("Empty index")
        if len(self._items) == 1:
            indices = torch.zeros(len(queries), dtype=torch.long)
            distances = torch.linalg.norm(queries - self._items[0], dim=1)
        else:
            k = min(k, len(self._items))  # [1, I].
            indices = []
            distances = []
            for i in range(0, len(queries), self._batch_size):
                batch = queries[i:i + self._batch_size]
                scores = torch.linalg.norm(batch[:, None, :] - self._items[None, :, :], dim=2)  # (Q, I).
                if k == 1:
                    batch_scores, batch_indices = torch.min(scores, dim=1)
                    indices.append(batch_indices[:, None])
                    distances.append(batch_scores[:, None])
                else:
                    batch_scores, batch_indices = torch.topk(scores, k, dim=1, largest=False)
                    indices.append(batch_indices)
                    distances.append(batch_scores)
            indices = torch.cat(indices, 0)
            distances = torch.cat(distances, 0)
        return distances.cpu().numpy(), indices.cpu().numpy()

    @staticmethod
    def _from_numpy(array):
        tensor = torch.from_numpy(array)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor


class KNNIndex:
    BACKENDS = {
        "faiss": faiss.IndexFlatL2,
        "numpy": NumpyIndexL2,
        "torch": TorchIndexL2
    }

    def __init__(self, dim, backend="torch"):
        self._index = self.BACKENDS[backend](dim)

    def __enter__(self):
        if self._index is None:
            raise RuntimeError("Can't create context multiple times.")
        return self._index

    def __exit__(self, exc_type, exc_value, traceback):
        self._index.reset()
        self._index = None
