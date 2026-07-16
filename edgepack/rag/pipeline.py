"""Federated RAG pipeline: encrypted local retrieval -> routed generation.

Flow per query:
  1. Embed the query on device, cosine top-k over the *encrypted* store
     (vectors decrypted only in RAM).
  2. Re-rank hits with the federated adapter (weights learned across devices
     via FedAvg — see ``edgepack.federated``).
  3. Score complexity (query + retrieval confidence) and route to the
     cheapest capable model tier within the token budget.
  4. Generate grounded in retrieved context; record actual token usage
     against the budget.

The ``LLMClient`` protocol keeps the pipeline model-agnostic: the default
``TemplateLLM`` is a deterministic extractive stub (CI-safe, zero deps); an
MLX-backed client can be dropped in on Apple Silicon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..router.model_router import ModelRouter, RoutingDecision
from .vector_store import EncryptedVectorStore, SearchHit


class LLMClient(Protocol):
    def generate(self, prompt: str, model: str, max_tokens: int) -> tuple[str, int]:
        """Returns (text, tokens_used)."""
        ...


class TemplateLLM:
    """Deterministic extractive 'generator' used for tests/demos and as the
    on-device fallback of last resort. Answers by quoting the best context."""

    def generate(self, prompt: str, model: str, max_tokens: int) -> tuple[str, int]:
        marker = "CONTEXT:"
        context = prompt.split(marker, 1)[1].strip() if marker in prompt else ""
        first = context.split("\n")[0].strip() if context else "No local context found."
        answer = f"[{model}] {first}"
        tokens = min(max_tokens, max(1, len(answer.split())))
        return answer, tokens


@dataclass
class RAGResponse:
    answer: str
    hits: list[SearchHit]
    routing: RoutingDecision
    tokens_used: int


@dataclass
class RerankAdapter:
    """Tiny federated-learnable re-ranker.

    score' = score + w · features(hit), with features cheap enough for any
    device.  ``weights`` is the vector that FedAvg aggregates across devices.
    """

    weights: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @staticmethod
    def features(hit: SearchHit, query: str) -> np.ndarray:
        q_terms = set(query.lower().split())
        d_terms = set(hit.text.lower().split())
        overlap = len(q_terms & d_terms) / max(1, len(q_terms))
        length_penalty = min(1.0, len(hit.text) / 500.0)
        return np.array([hit.score, overlap, length_penalty])

    def rerank(self, hits: list[SearchHit], query: str) -> list[SearchHit]:
        scored = [
            (hit.score + float(self.weights @ self.features(hit, query)), hit)
            for hit in hits
        ]
        return [h for _, h in sorted(scored, key=lambda t: -t[0])]


class RAGPipeline:
    def __init__(
        self,
        store: EncryptedVectorStore,
        router: ModelRouter,
        llm: LLMClient | None = None,
        adapter: RerankAdapter | None = None,
        top_k: int = 4,
    ):
        self.store = store
        self.router = router
        self.llm = llm or TemplateLLM()
        self.adapter = adapter or RerankAdapter()
        self.top_k = top_k

    def build_prompt(self, query: str, hits: list[SearchHit]) -> str:
        context = "\n".join(f"- {h.text}" for h in hits)
        return (
            "Answer the question using ONLY the context below.\n"
            f"QUESTION: {query}\n"
            f"CONTEXT:\n{context}\n"
        )

    def query(self, question: str) -> RAGResponse:
        hits = self.store.search(question, k=self.top_k)
        hits = self.adapter.rerank(hits, question)
        decision = self.router.route(question, retrieval_scores=[h.score for h in hits])
        prompt = self.build_prompt(question, hits)
        answer, tokens = self.llm.generate(
            prompt, model=decision.tier.name, max_tokens=decision.tier.max_output_tokens
        )
        self.router.record_usage(decision, tokens)
        return RAGResponse(answer=answer, hits=hits, routing=decision, tokens_used=tokens)
