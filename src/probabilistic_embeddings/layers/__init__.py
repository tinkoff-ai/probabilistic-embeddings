"""Configurable PyTorch layers and modules."""

from .embedder import IdentityEmbedder, CNNEmbedder
from .distribution import DiracDistribution, NormalDistribution, VMFDistribution
from .classifier import LinearClassifier, ArcFaceClassifier, CosFaceClassifier, LogLikeClassifier, VMFClassifier, SPEClassifier, ScorerClassifier
from .scorer import DotProductScorer, CosineScorer, ExpectedCosineScorer, NegativeL2Scorer, MutualLikelihoodScorer, HIBScorer
