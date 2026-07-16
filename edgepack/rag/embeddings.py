"""On-device text embeddings.

Default implementation is a deterministic **feature-hashing n-gram embedder**:
no model download, no network, identical output on every device — exactly what
a privacy-first edge RAG needs as a baseline, and fast enough for thousands of
docs on a phone.  The `Embedder` protocol lets you drop in a real sentence
encoder (e.g. a 4-bit MiniLM via MLX) without touching the store or retriever.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Feature-hashed unigrams + bigrams, L2-normalized, sublinear TF."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _slot(self, feature: str) -> tuple[int, float]:
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] & 1 else -1.0
        return idx, sign

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        toks = _tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            idx, sign = self._slot(f)
            vec[idx] += sign
        # sublinear scaling + L2 norm
        vec = np.sign(vec) * np.log1p(np.abs(vec))
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


def cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Rows of ``matrix`` are assumed L2-normalized; returns (row, score) best-first."""
    if matrix.size == 0:
        return []
    qn = np.linalg.norm(query)
    if qn == 0:
        return []
    scores = matrix @ (query / qn)
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]
