import torch


class DotProductScorer(torch.nn.Module):
    """Compare two embeddings using dot product.

    Args:
        distribution: Distribution used in the model.

    Inputs:
        - parameters1: First group of distributions with shape (..., K).
        - parameters2: Second group of distributions with shape (..., K).

    Outputs:
        - scores: Similarities with shape (...).
    """

    def __init__(self, distribution):
        super().__init__()
        self._distribution = distribution

    def __call__(self, parameters1, parameters2):
        embeddings1 = self._distribution.mean(parameters1)
        embeddings2 = self._distribution.mean(parameters2)
        products = (embeddings1 * embeddings2).sum(-1)
        return products

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}


class CosineScorer(torch.nn.Module):
    """Compare two embeddings using cosine similarity.

    Args:
        distribution: Distribution used in the model.

    Inputs:
        - parameters1: First group of distributions with shape (..., K).
        - parameters2: Second group of distributions with shape (..., K).

    Outputs:
        - scores: Similarities with shape (...).
    """

    def __init__(self, distribution):
        super().__init__()
        self._distribution = distribution

    def __call__(self, parameters1, parameters2):
        embeddings1 = torch.nn.functional.normalize(self._distribution.mean(parameters1), dim=-1)
        embeddings2 = torch.nn.functional.normalize(self._distribution.mean(parameters2), dim=-1)
        cosines = (embeddings1 * embeddings2).sum(-1)
        return cosines

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}


class ExpectedCosineScorer(torch.nn.Module):
    """Compare two embeddings using expected cosine similarity.

    Args:
        distribution: Distribution used in the model.

    Inputs:
        - parameters1: First group of distributions with shape (..., K).
        - parameters2: Second group of distributions with shape (..., K).

    Outputs:
        - scores: Similarities with shape (...).
    """

    SAMPLE_SIZE = 10
    BATCH_SIZE = 128

    def __init__(self, distribution):
        super().__init__()
        self._distribution = distribution

    def __call__(self, parameters1, parameters2):
        if (len(parameters1) == len(parameters2)) and (len(parameters1) > self.BATCH_SIZE):
            batch_size = len(parameters1)
            scores = []
            for i in range(0, batch_size, self.BATCH_SIZE):
                scores.append(self(parameters1[i:i + self.BATCH_SIZE], parameters2[i:i + self.BATCH_SIZE]))
            return torch.cat(scores)
        shape1 = list(parameters1.shape[:-1]) + [self.SAMPLE_SIZE, 1]
        shape2 = list(parameters2.shape[:-1]) + [1, self.SAMPLE_SIZE]
        print(shape1, shape2)
        embeddings1 = torch.nn.functional.normalize(self._distribution.sample(parameters1, shape1)[0], dim=-1)  # (..., K, 1, D).
        embeddings2 = torch.nn.functional.normalize(self._distribution.sample(parameters2, shape2)[0], dim=-1)  # (..., 1, K, D).
        cosines = (embeddings1 * embeddings2).sum(-1)  # (..., K, K).
        return cosines.mean((-1, -2))  # (...).

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}


class NegativeL2Scorer(torch.nn.Module):
    """Compare two embeddings using similarity based on euclidean distance.

    Args:
        distribution: Distribution used in the model.

    Inputs:
        - parameters1: First group of distributions with shape (..., K).
        - parameters2: Second group of distributions with shape (..., K).

    Outputs:
        - scores: Similarities with shape (...).
    """

    def __init__(self, distribution):
        super().__init__()
        self._distribution = distribution

    def __call__(self, parameters1, parameters2):
        embeddings1 = self._distribution.mean(parameters1)
        embeddings2 = self._distribution.mean(parameters2)
        distances = torch.square(embeddings1 - embeddings2).sum(-1)
        return -distances

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}


class MutualLikelihoodScorer(torch.nn.Module):
    """Compare two embeddings using MLS.

    Args:
        distribution: Distribution used in the model.

    Inputs:
        - parameters1: First group of distributions with shape (..., K).
        - parameters2: Second group of distributions with shape (..., K).

    Outputs:
        - scores: Similarities with shape (...).
    """

    def __init__(self, distribution):
        super().__init__()
        self._distribution = distribution

    def __call__(self, parameters1, parameters2):
        return self._distribution.logmls(parameters1, parameters2)

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}


class HIBScorer(torch.nn.Module):
    """Compare two embeddings using expectation of L2 sigmoid with trainable scale and bias.

    Scorer is used by HIB: https://arxiv.org/pdf/1810.00319.pdf

    Args:
        distribution: Distribution used in the model.

    Inputs:
        - parameters1: First group of distributions with shape (..., K).
        - parameters2: Second group of distributions with shape (..., K).

    Outputs:
        - scores: Similarities with shape (...).
    """

    NUM_SAMPLES = 8
    BATCH_SIZE = 128

    def __init__(self, distribution):
        super().__init__()
        self._distribution = distribution
        self.scale = torch.nn.Parameter(torch.ones([]))
        self.bias = torch.nn.Parameter(torch.zeros([]))

    def __call__(self, parameters1, parameters2):
        if (len(parameters1) == len(parameters2)) and (len(parameters1) > self.BATCH_SIZE):
            batch_size = len(parameters1)
            scores = []
            for i in range(0, batch_size, self.BATCH_SIZE):
                scores.append(self(parameters1[i:i + self.BATCH_SIZE], parameters2[i:i + self.BATCH_SIZE]))
            return torch.cat(scores)
        samples1 = self._distribution.sample(parameters1, list(parameters1.shape)[:-1] + [self.NUM_SAMPLES])[0]  # (..., K, D).
        samples2 = self._distribution.sample(parameters2, list(parameters2.shape)[:-1] + [self.NUM_SAMPLES])[0]  # (..., K, D).
        # ||a - b|| = sqrt(||a||^2 + ||b|| ^ 2 - 2(a, b)).
        norm1sq = (samples1 ** 2).sum(-1)  # (..., K).
        norm2sq = (samples2 ** 2).sum(-1)  # (..., K).
        dot = torch.matmul(samples1, samples2.transpose(-1, -2))  # (..., K, K).
        distances = (norm1sq.unsqueeze(-1) + norm2sq.unsqueeze(-2) - 2 * dot).sqrt()
        scores = torch.sigmoid(-self.scale * distances + self.bias).mean(dim=(-1, -2))  # (...).
        return scores

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {
            "scorer_scale": self.scale.item(),
            "scorer_bias": self.bias.item()
        }
