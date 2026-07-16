#!/usr/bin/env python3
"""Demo: federated learning rounds over per-device encrypted stores.

Each simulated device has its own encrypted DB and click feedback; only
DP-noised, secure-aggregated weight deltas reach the server.

Usage:
    python scripts/run_federated_round.py --devices 4 --rounds 3
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgepack.federated import FederatedClient, FederatedServer, FeedbackEvent
from edgepack.rag import EncryptedVectorStore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--devices", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="edgepack_fed_"))
    clients = []
    for i in range(args.devices):
        store = EncryptedVectorStore(tmp / f"device_{i}.db", passphrase=f"device-{i}-secret")
        good = store.add(f"Detailed packing guide for station {i}: heavy items bottom layer.")
        bad = store.add("Unrelated cafeteria menu for the week.")
        client = FederatedClient(client_id=i, store=store)
        client.record_feedback(FeedbackEvent(
            query="how to pack heavy items", clicked_doc_id=good, skipped_doc_id=bad))
        clients.append(client)

    server = FederatedServer(dim=3)
    for _ in range(args.rounds):
        report = server.run_round(clients)
        print(f"round {report.round_id}: {report.n_clients} devices, "
              f"update_norm={report.update_norm:.4f}, weights={report.global_weights.round(3)}")

    print("\nraw documents never left their device DBs; only masked weight deltas did.")


if __name__ == "__main__":
    main()
