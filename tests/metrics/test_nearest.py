import numpy as np
import torch
from unittest import TestCase, main

from probabilistic_embeddings.layers.distribution import DiracDistribution, NormalDistribution
from probabilistic_embeddings.layers.scorer import NegativeL2Scorer
from probabilistic_embeddings.metrics.nearest import NearestNeighboursMetrics, MAPR


class TestUtils(TestCase):
    def test_knn(self):
        for backend in ["faiss", "numpy", "torch"]:
            d = DiracDistribution(config={"dim": 2, "spherical": False})
            s = NegativeL2Scorer(d)
            metric = NearestNeighboursMetrics(d, s, config={"backend": backend})

            # Test unimodal.
            x = torch.tensor([
                [2, 0],
                [2.1, 0],
                [1.1, 0],
                [0, 1],
                [0, 0]
            ]).float().reshape((-1, 1, 2))

            indices = metric._multimodal_knn(x, 2)
            indices_gt = np.array([
                [0, 1],
                [1, 0],
                [2, 0],
                [3, 4],
                [4, 3]
            ]).reshape((-1, 1, 2))
            self.assertTrue(np.allclose(indices, indices_gt))

    def test_gather(self):
        x = torch.tensor([
            [1, 2],
            [3, 4],
            [5, 6]
        ])  # (3, 2).
        index = torch.tensor([
            [0, 2],
            [2, 1]
        ])  # (2, 2).
        result = NearestNeighboursMetrics._gather_broadcast(x[None], 1, index[..., None])  # (2, 2, 2).
        result_gt = [
            [[1, 2], [5, 6]],
            [[5, 6], [3, 4]]
        ]
        self.assertTrue(np.allclose(result, result_gt))

    def test_remove_duplicates(self):
        # K unique values are available.
        x = torch.tensor([
            [5, 3, 2, 5, 1, 1, 5],
            [5, 4, 3, 2, 1, 1, 1],
            [5, 4, 2, 2, 2, 2, 4],
        ])
        result = NearestNeighboursMetrics._remove_duplicates(x, 3)
        result_gt = [
            [5, 3, 2],
            [5, 4, 3],
            [5, 4, 2]
        ]
        self.assertTrue(np.allclose(result, result_gt))

        # The number of unique values is less than K.
        result = NearestNeighboursMetrics._remove_duplicates(x, 6)
        result_gt = [
            [5, 3, 2, 1, 1, 5],
            [5, 4, 3, 2, 1, 1],
            [5, 4, 2, 2, 2, 4]
        ]
        self.assertTrue(np.allclose(result, result_gt))

    def test_get_positives(self):
        d = DiracDistribution(config={"dim": 1, "spherical": False})
        s = NegativeL2Scorer(d)
        metric = NearestNeighboursMetrics(d, s)
        labels = torch.tensor([1, 0, 1, 2, 0])
        parameters = torch.tensor([
            [0],
            [0],
            [1],
            [3],
            [5]
        ]).float()
        scores, counts, same_mask = metric._get_positives(parameters, labels)
        scores_gt = torch.tensor([
            [0, -1],
            [0, -25],
            [0, -1],
            [0, -26],  # Second is dummy score with minimum value.
            [0, -25]
        ])
        counts_gt = torch.tensor([2, 2, 2, 1, 2])
        same_mask_gt = torch.tensor([
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0]
        ])
        self.assertTrue((scores == scores_gt).all())
        self.assertTrue((counts == counts_gt).all())
        self.assertTrue((same_mask == same_mask_gt).all())


class TestMAPR(TestCase):
    def test_simple(self):
        d = DiracDistribution(config={"dim": 1, "spherical": False})
        s = NegativeL2Scorer(d)

        m = NearestNeighboursMetrics(d, s, config={"metrics": ["mapr-ms"], "prefetch_factor": 1})
        labels = torch.tensor([1, 1, 0, 1, 2, 2, 0, 0, 1])  # (B).
        parameters = torch.arange(len(labels))[:, None] ** 1.01  # (B, 1).

        # N  R   Nearest      Same      P            MAP@R
        # 0  4   0, 1, 2, 3   1 1 0 1   1, 1, 3/4    11/16
        # 1  4   1, 0, 2, 3   1 1 0 1   1, 1, 3/4    11/16
        # 2  3   2, 1, 3      1 0 0     1            1/3
        # 3  4   3, 2, 4, 1   1 0 0 1   1, 1/2       3/8
        # 4  2   4, 3         1 0                    1/2
        # 5  2   5, 4         1 1       1, 1         1
        # 6  3   6, 5, 7      1 0 1     1, 2/3       5/9
        # 7  3   7, 6, 8      1 1 0     1, 1         2/3
        # 8  4   8, 7, 6, 5   1 0 0 0   1            1/4

        result = m(parameters, labels)["mapr-ms"].item()
        result_gt = np.mean([11 / 16, 11 / 16, 1 / 3, 3 / 8, 1 / 2, 1, 5 / 9, 2 / 3, 1 / 4])
        self.assertAlmostEqual(result, result_gt)

        m = NearestNeighboursMetrics(d, s, config={"metrics": ["mapr"], "prefetch_factor": 1})
        labels = torch.tensor([1, 1, 0, 1, 2, 2, 0, 0, 1])  # (B).
        parameters = torch.arange(len(labels))[:, None] ** 1.01  # (B, 1).

        # N  R   Nearest      Same      P       MAP@R
        # 0  4   1, 2, 3      1 0 1   1, 2/3    5/9
        # 1  4   0, 2, 3      1 0 1   1, 2/3    5/9
        # 2  3   1, 3         0 0               0
        # 3  4   2, 4, 1      0 0 1   1/3       1/9
        # 4  2   3            0                 0
        # 5  2   4            1       1         1
        # 6  3   5, 7         0 1     1/2       1/4
        # 7  3   6, 8         1 0     1         1/2
        # 8  4   7, 6, 5      0 0 0             0

        result = m(parameters, labels)["mapr"].item()
        result_gt = np.mean([5 / 9, 5 / 9, 0, 1 / 9, 0, 1, 1 / 4, 1 / 2, 0])
        self.assertAlmostEqual(result, result_gt)

    def test_toy(self):
        """Eval MAP@R on toy examples from original paper."""
        sample_size = 1000
        mapr_gt = {
            (0, 1): 0.779,
            (0, None, 1): 0.998,
            (0, None, 0, 1, None, 1): 0.714
        }
        d = DiracDistribution(config={"dim": 2, "spherical": False})
        s = NegativeL2Scorer(d)
        m = NearestNeighboursMetrics(d, s, config={"metrics": ["mapr-ms"], "prefetch_factor": 1})
        for pattern, gt in mapr_gt.items():
            embeddings1 = torch.rand(sample_size, 2)
            embeddings2 = torch.rand(sample_size, 2)
            self._apply_pattern_inplace(embeddings1, pattern, 0)
            self._apply_pattern_inplace(embeddings2, pattern, 1)
            embeddings = torch.cat((embeddings1, embeddings2))
            labels = torch.cat((torch.zeros(sample_size).long(), torch.ones(sample_size).long()))
            result = m(embeddings, labels)["mapr-ms"].item()
            self.assertTrue(abs(result - gt) < 0.05)

    def _apply_pattern_inplace(self, sample, pattern, label):
        """Apply pattern to uniform distribution."""
        pattern = tuple(p == label for p in pattern)
        num_bins = sum(pattern)
        sample[:, 0] *= num_bins
        for i, j in reversed(list(enumerate(np.nonzero(pattern)[0]))):
            mask = (sample[:, 0] >= i) & (sample[:, 0] <= i + 1)
            sample[mask, 0] += j - i
        sample[:, 0] *= 2 / len(pattern)


class TestRecallK(TestCase):
    def test_simple(self):
        d = DiracDistribution(config={"dim": 1, "spherical": False})
        s = NegativeL2Scorer(d)
        labels = torch.tensor([1, 1, 0, 2, 0, 1])  # (B).
        parameters = torch.tensor([0, 0.5, 1.5, 3.5, 4, 5])[:, None]  # (B, 1).

        config = {"metrics": ["recall"], "recall_k_values": (1, 2, 3, 4, 5, 10), "prefetch_factor": 1}
        m = NearestNeighboursMetrics(d, s, config=config)(parameters, labels)
        # Item    Nearest   Same
        #
        # 0       12345     10001
        # 1       02345     10001
        # 2       10345     00010
        # 3       45210     00000 (excluded as one-element class).
        # 4       35210     00100
        # 5       43210     00011
        self.assertAlmostEqual(m["recall@1"], 2 / 5)
        self.assertAlmostEqual(m["recall@2"], 2 / 5)
        self.assertAlmostEqual(m["recall@3"], 3 / 5)
        self.assertAlmostEqual(m["recall@4"], 1)
        self.assertAlmostEqual(m["recall@5"], 1)
        self.assertAlmostEqual(m["recall@10"], 1)


class TestERCRecallK(TestCase):
    def test_simple(self):
        d = NormalDistribution(config={"dim": 1})
        s = NegativeL2Scorer(d)
        labels = torch.tensor([1, 1, 0, 2, 0, 1])  # (B).
        centers = torch.tensor([0, 0.5, 1.5, 3.5, 4, 5])  # (B).
        confidences = torch.tensor([0, 2, 4, 5, 3, 1]).float()  # (B).
        parameters = torch.stack([centers, -confidences], 1)  # (B, 2).

        config = {"metrics": ["erc-recall@1"], "recall_k_values": (1, 2, 3, 4, 5, 10), "prefetch_factor": 1}
        m = NearestNeighboursMetrics(d, s, config=config)(parameters, labels)
        # Item    Nearest   Confidence   Same
        #
        # 0       12345         0        1
        # 1       02345         2        1
        # 2       10345         4        0
        # 3       45210         5        0 (excluded as one-element class).
        # 4       35210         3        0
        # 5       43210         1        0
        #
        # Same ordered by descending confidence:
        # 0 0 1 0 1
        #
        # Metrics
        # 0/1 0/2 1/3 1/4 2/5
        #
        self.assertAlmostEqual(m["erc-recall@1"], 1 - np.mean([0, 0, 1/3, 1/4, 2/5]), places=6)


class TestERCMAPR(TestCase):
    def test_simple(self):
        d = NormalDistribution(config={"dim": 1})
        s = NegativeL2Scorer(d)

        labels = torch.tensor([1, 1, 0, 1, 2, 2, 0, 0, 1])  # (B).
        centers = torch.arange(len(labels)) ** 1.01  # (B, 1).
        confidences = torch.tensor([0, 2, 4, 6, 8, 7, 5, 3, 1]).float()  # (B).
        parameters = torch.stack([centers, -confidences], 1)  # (B, 2).
        m = NearestNeighboursMetrics(d, s, config={"metrics": ["erc-mapr"], "prefetch_factor": 1})(parameters, labels)

        # N  R   Nearest      Same      P       MAP@R   Confidence
        # 0  4   1, 2, 3      1 0 1   1, 2/3    5/9     0
        # 1  4   0, 2, 3      1 0 1   1, 2/3    5/9     2
        # 2  3   1, 3         0 0               0       4
        # 3  4   2, 4, 1      0 0 1   1/3       1/9     6
        # 4  2   3            0                 0       8
        # 5  2   4            1       1         1       7
        # 6  3   5, 7         0 1     1/2       1/4     5
        # 7  3   6, 8         1 0     1         1/2     3
        # 8  4   7, 6, 5      0 0 0             0       1
        #
        # MAP@R ordered by descending confidence:
        # 0 1 1/9 1/4 0 1/2 5/9 0 5/9
        #

        maprs = np.array([0, 1, 1/9, 1/4, 0, 1/2, 5/9, 0, 5/9])
        erc_mapr_gt = 1 - (np.cumsum(maprs) / np.arange(1, len(maprs) + 1)).mean()

        self.assertAlmostEqual(m["erc-mapr"], erc_mapr_gt, places=6)


if __name__ == "__main__":
    main()
