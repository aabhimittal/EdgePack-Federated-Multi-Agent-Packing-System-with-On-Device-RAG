import pytest

from edgepack.rag import EncryptedVectorStore, RAGPipeline
from edgepack.router import (
    BudgetExhausted,
    ComplexityEstimator,
    ModelRouter,
    TokenBudget,
)


def test_complexity_simple_vs_hard():
    est = ComplexityEstimator()
    simple = est.estimate("What is the thermal budget?", retrieval_scores=[0.9])
    hard = est.estimate(
        "Explain why the federated aggregation converges, compare secure aggregation "
        "versus plain FedAvg tradeoffs, then design a rollout plan and analyze failure modes.",
        retrieval_scores=[0.1],
    )
    assert hard.score > simple.score


def test_router_picks_cheapest_covering_tier():
    router = ModelRouter(budget=TokenBudget(limit=1e9))
    d = router.route("What is the thermal budget?", retrieval_scores=[0.95])
    assert d.tier.name == "nano-1b-q4"
    d2 = router.route(
        "Explain how and why the reality gap changed, compare calibration approaches, "
        "analyze tradeoffs and design steps to verify; also derive the budget formula.",
        retrieval_scores=[0.05],
    )
    assert d2.tier.cost_multiplier > d.tier.cost_multiplier


def test_router_downgrades_under_budget_pressure():
    # budget only covers the cheapest tier's worst case (256 tokens * 1.0)
    router = ModelRouter(budget=TokenBudget(limit=300.0))
    d = router.route(
        "Explain and compare and analyze and design everything in exhaustive detail "
        "with tradeoffs, then derive proofs; " * 3,
        retrieval_scores=[0.0],
    )
    assert d.tier.name == "nano-1b-q4"
    assert d.downgraded


def test_router_raises_when_budget_exhausted():
    budget = TokenBudget(limit=100.0)
    budget.charge(100, 1.0, "nano-1b-q4")
    router = ModelRouter(budget=budget)
    with pytest.raises(BudgetExhausted):
        router.route("anything at all")


def test_router_offline_excludes_cloud():
    router = ModelRouter(offline=True, budget=TokenBudget(limit=1e9))
    d = router.route(
        "Explain, compare, analyze, design, derive and prove everything about the system "
        "including tradeoffs; " * 4,
        retrieval_scores=[0.0],
    )
    assert d.tier.on_device


def test_usage_accounting():
    router = ModelRouter(budget=TokenBudget(limit=1000.0))
    d = router.route("What is the bin size?", retrieval_scores=[0.9])
    router.record_usage(d, tokens_used=50)
    s = router.usage_summary()
    assert s["spent"] == 50 * d.tier.cost_multiplier
    assert s["per_tier"][d.tier.name]["calls"] == 1


# ------------------------------------------------------------------ pipeline
def test_rag_end_to_end(tmp_path):
    store = EncryptedVectorStore(tmp_path / "rag.db", passphrase="pw")
    store.add("Heavy items go in the bottom layer of the bin.")
    store.add("The cafeteria opens at nine in the morning.")
    router = ModelRouter(budget=TokenBudget(limit=10_000.0))
    rag = RAGPipeline(store, router)

    resp = rag.query("Where do heavy items go?")
    assert "bottom layer" in resp.answer
    assert resp.hits[0].score > 0
    assert resp.tokens_used > 0
    assert router.budget.spent > 0


def test_rag_routes_simple_query_to_cheap_tier(tmp_path):
    store = EncryptedVectorStore(tmp_path / "rag.db", passphrase="pw")
    store.add("Heavy items go in the bottom layer of the bin.")
    router = ModelRouter(budget=TokenBudget(limit=10_000.0))
    rag = RAGPipeline(store, router)
    resp = rag.query("Where do heavy items go?")
    assert resp.routing.tier.name == "nano-1b-q4"
