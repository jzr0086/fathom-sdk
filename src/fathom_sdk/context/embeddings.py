"""CodeBERT embedding utilities for semantic code analysis.

Provides code embedding generation and similarity computation.
Falls back gracefully when torch/transformers are unavailable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Lazy-loaded model instance
_model = None
_model_load_attempted = False


def _load_model(model_name: str = "microsoft/codebert-base"):
    """Lazily load the sentence-transformers model."""
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)
    except (ImportError, Exception):
        _model = None
    return _model


def get_code_embeddings(
    chunks: list[str],
    model_name: str = "microsoft/codebert-base",
) -> Optional[np.ndarray]:
    """Compute embeddings for code chunks.

    Returns an (N, D) array where N is the number of chunks and D is the
    embedding dimension.  Returns None if the ML dependencies are missing.
    """
    if not chunks:
        return np.empty((0, 0))

    model = _load_model(model_name)
    if model is None:
        return None

    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(embeddings)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for embeddings.

    Args:
        embeddings: (N, D) array of embeddings.

    Returns:
        (N, N) similarity matrix with values in [-1, 1].
    """
    if embeddings.size == 0:
        return np.empty((0, 0))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    normalized = embeddings / norms
    return normalized @ normalized.T


def find_similar_pairs(
    embeddings: np.ndarray,
    threshold: float = 0.85,
) -> list[tuple[int, int, float]]:
    """Find pairs of code chunks with cosine similarity above threshold.

    Returns list of (index_i, index_j, similarity) tuples, i < j.
    """
    if embeddings.size == 0:
        return []
    sim_matrix = cosine_similarity_matrix(embeddings)
    n = sim_matrix.shape[0]
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                pairs.append((i, j, float(sim_matrix[i, j])))
    return pairs
