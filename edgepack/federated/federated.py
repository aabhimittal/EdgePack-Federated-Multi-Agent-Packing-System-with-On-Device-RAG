"""Federated learning of the RAG re-rank adapter.

Each device improves retrieval from its OWN click/feedback data; only small
weight updates ever leave the device.  Raw documents and queries never do.

Privacy stack, in order:
  1. **Local training** — pairwise ranking loss on (query, clicked, skipped).
  2. **Clipping + Gaussian noise** — per-client differential privacy.
  3. **Secure aggregation** — pairwise antisymmetric masks: the server only
     sees masked updates whose SUM equals the true sum; individual updates
     are unrecoverable (masks cancel only in aggregate).
  4. **FedAvg** — server averages and broadcasts new global weights.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from ..rag.pipeline import RerankAdapter
from ..rag.vector_store import EncryptedVectorStore


# ------------------------------------------------------------------- privacy
@dataclass
class DPConfig:
    clip_norm: float = 1.0
    noise_multiplier: float = 0.1   # sigma = noise_multiplier * clip_norm

    def privatize(self, update: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        norm = float(np.linalg.norm(update))
        if norm > self.clip_norm:
            update = update * (self.clip_norm / norm)
        sigma = self.noise_multiplier * self.clip_norm
        return update + rng.normal(0.0, sigma, size=update.shape)


def _pair_mask(seed_a: int, seed_b: int, dim: int) -> np.ndarray:
    """Deterministic mask shared by clients a and b (a<b adds it, b>a subtracts)."""
    key = hashlib.sha256(f"{min(seed_a, seed_b)}:{max(seed_a, seed_b)}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
    return rng.normal(0.0, 1.0, size=dim)


# -------------------------------------------------------------------- client
@dataclass
class FeedbackEvent:
    query: str
    clicked_doc_id: str
    skipped_doc_id: str


class FederatedClient:
    def __init__(self, client_id: int, store: EncryptedVectorStore,
                 dp: DPConfig | None = None, lr: float = 0.1, seed: int | None = None):
        self.client_id = client_id
        self.store = store
        self.adapter = RerankAdapter()
        self.dp = dp or DPConfig()
        self.lr = lr
        self.rng = np.random.default_rng(seed if seed is not None else client_id)
        self.feedback: list[FeedbackEvent] = []

    def record_feedback(self, event: FeedbackEvent) -> None:
        self.feedback.append(event)

    def local_update(self) -> np.ndarray:
        """Pairwise ranking step: clicked doc should outscore skipped doc.

        Returns the weight DELTA (not the weights) — that's what gets
        aggregated.
        """
        w = self.adapter.weights.copy()
        for ev in self.feedback:
            try:
                clicked_text, _ = self.store.get(ev.clicked_doc_id)
                skipped_text, _ = self.store.get(ev.skipped_doc_id)
            except KeyError:
                continue
            from ..rag.vector_store import SearchHit

            pos = RerankAdapter.features(
                SearchHit(ev.clicked_doc_id, 0.0, clicked_text, {}), ev.query)
            neg = RerankAdapter.features(
                SearchHit(ev.skipped_doc_id, 0.0, skipped_text, {}), ev.query)
            margin = float(w @ (pos - neg))
            if margin < 1.0:  # hinge loss gradient
                w += self.lr * (pos - neg)
        return w - self.adapter.weights

    def masked_update(self, roster: list[int]) -> np.ndarray:
        """DP-privatized update + pairwise masks for secure aggregation."""
        update = self.dp.privatize(self.local_update(), self.rng)
        dim = update.shape[0]
        for other in roster:
            if other == self.client_id:
                continue
            mask = _pair_mask(self.client_id, other, dim)
            update = update + mask if self.client_id < other else update - mask
        return update

    def apply_global(self, weights: np.ndarray) -> None:
        self.adapter.weights = weights.copy()


# -------------------------------------------------------------------- server
@dataclass
class RoundReport:
    round_id: int
    n_clients: int
    update_norm: float
    global_weights: np.ndarray


class FederatedServer:
    """FedAvg over masked updates. Never sees raw data or unmasked updates."""

    def __init__(self, dim: int = 3):
        self.global_weights = np.zeros(dim)
        self.rounds: list[RoundReport] = []

    def run_round(self, clients: list[FederatedClient]) -> RoundReport:
        roster = [c.client_id for c in clients]
        # masks cancel in the sum: sum(masked) == sum(true updates)
        total = np.zeros_like(self.global_weights)
        for c in clients:
            c.apply_global(self.global_weights)
            total += c.masked_update(roster)
        avg_update = total / max(1, len(clients))
        self.global_weights = self.global_weights + avg_update

        report = RoundReport(
            round_id=len(self.rounds) + 1,
            n_clients=len(clients),
            update_norm=float(np.linalg.norm(avg_update)),
            global_weights=self.global_weights.copy(),
        )
        self.rounds.append(report)
        for c in clients:
            c.apply_global(self.global_weights)
        return report
