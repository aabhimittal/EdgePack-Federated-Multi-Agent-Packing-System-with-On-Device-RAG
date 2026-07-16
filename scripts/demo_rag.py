#!/usr/bin/env python3
"""Demo: encrypted on-device RAG with complexity-routed generation.

Usage:
    python scripts/demo_rag.py --db /tmp/edgepack.db --budget 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgepack.rag import EncryptedVectorStore, RAGPipeline
from edgepack.router import ModelRouter, TokenBudget

DOCS = [
    "Bin 4 accepts fragile items only when the physics agent stability score exceeds 0.8.",
    "The packing policy bundle v3 reduced the reality gap to 6 percent on the loading dock cell.",
    "Thermal budget for the warehouse tablets is 85 celsius; quantized models must stay under 6 GB RAM.",
    "Federated round 12 improved retrieval precision by 4 points without moving any raw documents.",
    "Items heavier than 2 kg must be placed in the bottom layer of the bin.",
    "The edge-7b-q4 model handles most packing questions; cloud-large is reserved for audits.",
]

QUERIES = [
    "Where do heavy items go?",                                            # trivial -> nano tier
    "Why did federated round 12 improve retrieval, and how does that interact with the reality gap?",  # complex
    "What is the thermal budget?",                                         # trivial
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="/tmp/edgepack_demo.db")
    ap.add_argument("--passphrase", default="demo-device-passphrase")
    ap.add_argument("--budget", type=float, default=5000.0)
    args = ap.parse_args()

    store = EncryptedVectorStore(args.db, passphrase=args.passphrase)
    if len(store) == 0:
        store.add_many(DOCS)
        print(f"indexed {len(DOCS)} encrypted docs -> {args.db}")

    router = ModelRouter(budget=TokenBudget(limit=args.budget))
    rag = RAGPipeline(store, router)

    for q in QUERIES:
        resp = rag.query(q)
        r = resp.routing
        print(f"\nQ: {q}")
        print(f"   complexity={r.complexity.score:.2f} -> tier={r.tier.name}"
              f"{' (downgraded)' if r.downgraded else ''}")
        print(f"   A: {resp.answer}")

    s = router.usage_summary()
    print(f"\ntoken budget: spent={s['spent']:.0f} remaining={s['remaining']:.0f}")
    for tier, d in s["per_tier"].items():
        print(f"   {tier}: {d['calls']} calls, {d['tokens']} tokens, {d['cost']:.0f} budget units")


if __name__ == "__main__":
    main()
