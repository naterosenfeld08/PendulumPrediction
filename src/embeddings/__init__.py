"""Embedding implementations and registry."""

from embeddings.fft_features import FFTFeatureEmbedder
from embeddings.hybrid_features import HybridPhysicsFFTEmbedder
from embeddings.physics_features import PhysicsFeatureEmbedder
from embeddings.raw_window import RawWindowEmbedder

_REGISTRY = {
    "physics_features_v1": PhysicsFeatureEmbedder,
    "raw_window": RawWindowEmbedder,
    "fft_features": FFTFeatureEmbedder,
    "hybrid_physics_fft": HybridPhysicsFFTEmbedder,
}


def available_embeddings() -> list[str]:
    """Return sorted embedding names."""
    return sorted(_REGISTRY.keys())


def build_embedder(name: str):
    """Construct embedder by registry name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown embedding '{name}'. Available: {available_embeddings()}")
    return _REGISTRY[name]()


__all__ = [
    "FFTFeatureEmbedder",
    "HybridPhysicsFFTEmbedder",
    "PhysicsFeatureEmbedder",
    "RawWindowEmbedder",
    "available_embeddings",
    "build_embedder",
]
