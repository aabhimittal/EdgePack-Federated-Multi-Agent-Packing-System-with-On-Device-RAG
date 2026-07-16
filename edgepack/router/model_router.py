"""Complexity-based model switching under a token budget.

The core cost-control idea: **never spend big-model tokens on a small-model
task**.  Every request is scored for complexity (0..1) from cheap lexical and
retrieval signals, then routed to the cheapest model tier whose capability
covers that score.  A global :class:`TokenBudget` meters spend (weighted by
each tier's cost multiplier) and forces graceful *downgrades* as the budget
runs out — the system gets cheaper, not broken, under pressure.

Tiers are declarative, so the same router drives on-device 4-bit models (from
the quantization workflow) and cloud fallbacks.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


# --------------------------------------------------------------- complexity
_WH_RE = re.compile(r"\b(why|how|explain|compare|derive|prove|analyze|design|tradeoffs?)\b", re.I)
_MULTI_RE = re.compile(r"\b(and|then|also|versus|vs\.?|steps?)\b|[;,]", re.I)
_CODE_RE = re.compile(r"```|\bdef\b|\bclass\b|[{}()\[\]]{2,}")


@dataclass
class ComplexityReport:
    score: float                 # 0 (trivial) .. 1 (hard)
    signals: dict[str, float]


class ComplexityEstimator:
    """Cheap, deterministic complexity scoring — costs ~0 tokens itself."""

    def estimate(self, query: str, retrieval_scores: list[float] | None = None) -> ComplexityReport:
        words = query.split()
        n = len(words)

        length = min(1.0, n / 60.0)                                  # long asks are harder
        reasoning = min(1.0, len(_WH_RE.findall(query)) * 0.45)      # why/how/compare...
        multi_part = min(1.0, len(_MULTI_RE.findall(query)) / 6.0)   # compound requests
        code = 1.0 if _CODE_RE.search(query) else 0.0
        vocab = min(1.0, (len(set(w.lower() for w in words)) / n) * (n / 25.0)) if n else 0.0

        # weak retrieval = the answer isn't sitting in the local corpus -> harder
        if retrieval_scores:
            retrieval_uncertainty = float(max(0.0, 1.0 - max(retrieval_scores)))
        else:
            retrieval_uncertainty = 0.5

        signals = {
            "length": length,
            "reasoning": reasoning,
            "multi_part": multi_part,
            "code": code,
            "vocab": vocab,
            "retrieval_uncertainty": retrieval_uncertainty,
        }
        weights = {
            "length": 0.15, "reasoning": 0.3, "multi_part": 0.15,
            "code": 0.15, "vocab": 0.1, "retrieval_uncertainty": 0.15,
        }
        score = sum(signals[k] * weights[k] for k in weights)
        return ComplexityReport(score=float(min(1.0, score)), signals=signals)


# -------------------------------------------------------------------- budget
class BudgetExhausted(Exception):
    pass


@dataclass
class TokenBudget:
    """Meters *weighted* token spend: big-model tokens cost more budget units."""

    limit: float
    spent: float = 0.0
    history: list[dict] = field(default_factory=list)

    def charge(self, tokens: int, cost_multiplier: float, tier: str) -> None:
        cost = tokens * cost_multiplier
        self.spent += cost
        self.history.append({"tier": tier, "tokens": tokens, "cost": cost})

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.spent)

    @property
    def fraction_remaining(self) -> float:
        return self.remaining / self.limit if self.limit > 0 else 0.0


# --------------------------------------------------------------------- tiers
@dataclass
class ModelTier:
    name: str
    capability: float        # max complexity this tier handles well (0..1)
    cost_multiplier: float   # budget units per token
    max_output_tokens: int
    on_device: bool = True


DEFAULT_TIERS = [
    ModelTier("nano-1b-q4", capability=0.35, cost_multiplier=1.0, max_output_tokens=256),
    ModelTier("edge-7b-q4", capability=0.7, cost_multiplier=4.0, max_output_tokens=512),
    ModelTier("cloud-large", capability=1.0, cost_multiplier=20.0, max_output_tokens=1024, on_device=False),
]


@dataclass
class RoutingDecision:
    tier: ModelTier
    complexity: ComplexityReport
    downgraded: bool
    reason: str


class ModelRouter:
    """Pick the cheapest tier that covers the task; downgrade under budget pressure."""

    def __init__(self, tiers: list[ModelTier] | None = None, budget: TokenBudget | None = None,
                 offline: bool = False):
        self.tiers = sorted(tiers or list(DEFAULT_TIERS), key=lambda t: t.cost_multiplier)
        self.budget = budget or TokenBudget(limit=float("inf"))
        self.offline = offline
        self.estimator = ComplexityEstimator()

    def route(self, query: str, retrieval_scores: list[float] | None = None) -> RoutingDecision:
        report = self.estimator.estimate(query, retrieval_scores)
        candidates = [t for t in self.tiers if not (self.offline and not t.on_device)]
        if not candidates:
            raise RuntimeError("no model tiers available")

        # cheapest tier whose capability covers the complexity score
        chosen = next((t for t in candidates if t.capability >= report.score), candidates[-1])
        downgraded, reason = False, f"complexity {report.score:.2f} -> cheapest covering tier"

        # budget pressure: estimated worst-case cost must fit in remaining budget;
        # otherwise step down tiers until it does.
        def worst_case(t: ModelTier) -> float:
            return t.max_output_tokens * t.cost_multiplier

        while worst_case(chosen) > self.budget.remaining:
            cheaper = [t for t in candidates if t.cost_multiplier < chosen.cost_multiplier]
            if not cheaper:
                if worst_case(chosen) > self.budget.remaining and math.isfinite(self.budget.limit):
                    raise BudgetExhausted(
                        f"budget remaining {self.budget.remaining:.0f} cannot cover "
                        f"cheapest tier ({worst_case(chosen):.0f})"
                    )
                break
            chosen = cheaper[-1]
            downgraded = True
            reason = f"downgraded under budget pressure ({self.budget.fraction_remaining:.0%} left)"

        return RoutingDecision(tier=chosen, complexity=report, downgraded=downgraded, reason=reason)

    def record_usage(self, decision: RoutingDecision, tokens_used: int) -> None:
        self.budget.charge(tokens_used, decision.tier.cost_multiplier, decision.tier.name)

    def usage_summary(self) -> dict:
        per_tier: dict[str, dict] = {}
        for h in self.budget.history:
            d = per_tier.setdefault(h["tier"], {"calls": 0, "tokens": 0, "cost": 0.0})
            d["calls"] += 1
            d["tokens"] += h["tokens"]
            d["cost"] += h["cost"]
        return {
            "spent": self.budget.spent,
            "remaining": self.budget.remaining,
            "per_tier": per_tier,
        }
