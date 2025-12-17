"""
Audio retrieval methods.

This module implements various audio retrieval approaches:
- M1: MFCC + Global Pooling + Cosine Distance
- M2: MFCC + Delta + Global Pooling + Cosine Distance
- M3: Log-Mel + Global Pooling + Cosine Distance
- M4: Spectral Statistics + L2 Distance
- M5: MFCC + DTW Distance
- M6: MFCC + Bag-of-Audio-Words + Chi-squared Distance
- M7: Multi-resolution Fusion
- M8: CLAP Embedding + Cosine Distance
- M9: Hybrid CLAP + MFCC Late Fusion
- BEATs: BEATs Embedding + Cosine Distance
- Autoencoder: Autoencoder latent space retrieval
- CNN: CNN penultimate layer retrieval
- Contrastive: Supervised contrastive embedding retrieval
- LateFusion: Weighted distance combination
- RankFusion: Reciprocal Rank Fusion
"""

from .base import BaseRetriever
from .pool_retriever import PoolRetriever, create_method_m1, create_method_m2, create_method_m3, create_method_m4
from .dtw_retriever import DTWRetriever, create_method_m5
from .boaw_retriever import BoAWRetriever, create_method_m6
from .multires_retriever import MultiResRetriever, create_method_m7
from .clap_retriever import CLAPRetriever, create_method_m8
from .hybrid_retriever import HybridRetriever, create_method_m9
from .beats_retriever import BEATsRetriever, create_beats_retriever
from .autoencoder_retriever import AutoencoderRetriever, create_autoencoder_retriever
from .cnn_retriever import CNNRetriever, create_cnn_retriever
from .contrastive_retriever import ContrastiveRetriever, create_contrastive_retriever
from .fusion_retriever import LateFusionRetriever, RankFusionRetriever, create_late_fusion, create_rank_fusion
from .twostage_retriever import TwoStageRetriever, create_twostage_retriever
from .partial_retriever import PartialQueryRetriever, create_partial_retriever

__all__ = [
    'BaseRetriever',
    'PoolRetriever',
    'DTWRetriever',
    'BoAWRetriever',
    'MultiResRetriever',
    'CLAPRetriever',
    'HybridRetriever',
    'BEATsRetriever',
    'AutoencoderRetriever',
    'CNNRetriever',
    'ContrastiveRetriever',
    'LateFusionRetriever',
    'RankFusionRetriever',
    'TwoStageRetriever',
    'PartialQueryRetriever',
    # Factory functions
    'create_method_m1',
    'create_method_m2',
    'create_method_m3',
    'create_method_m4',
    'create_method_m5',
    'create_method_m6',
    'create_method_m7',
    'create_method_m8',
    'create_method_m9',
    'create_beats_retriever',
    'create_autoencoder_retriever',
    'create_cnn_retriever',
    'create_contrastive_retriever',
    'create_late_fusion',
    'create_rank_fusion',
    'create_twostage_retriever',
    'create_partial_retriever',
]
