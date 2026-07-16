#!/usr/bin/env python3
"""Demo: edge MLOps loop — register a MARL policy bundle, canary-roll it out.

Takes the manifest exported by the MARL-Sim2real repo (`run_bridge.py`),
registers it with promotion gates on the reality-gap metrics, and simulates a
health-gated OTA rollout across a device fleet.

Usage:
    python scripts/deploy_policy.py --manifest ../MARL-Sim2real/artifacts/packing_policy_real.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgepack.mlops import ModelRegistry, RolloutManager, TelemetryCollector, TelemetrySample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="policy bundle manifest JSON from MARL-Sim2real")
    ap.add_argument("--registry", default="registry")
    ap.add_argument("--fleet-size", type=int, default=20)
    ap.add_argument("--min-real-utilization", type=float, default=0.15)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    metrics = manifest.get("metrics", {})

    registry = ModelRegistry(args.registry)
    mv = registry.register(manifest["name"], manifest_path, metrics=metrics)
    print(f"registered {mv.name} v{mv.version}")

    try:
        registry.promote(mv.name, mv.version,
                         min_metrics={"real_utilization": args.min_real_utilization})
        print(f"promoted to production (real_utilization="
              f"{metrics.get('real_utilization', 0):.1%})")
    except ValueError as e:
        print(f"promotion blocked: {e}", file=sys.stderr)
        sys.exit(1)

    fleet = [f"device-{i:03d}" for i in range(args.fleet_size)]
    telemetry = TelemetryCollector()
    rollout = RolloutManager(registry, telemetry, fleet)
    state = rollout.start(mv.name, mv.version)
    print(f"rollout started: phase={state.phase}, devices={len(state.devices_updated)}")

    # healthy canary telemetry -> advance through phases
    for phase in ("staged", "full"):
        for d in state.devices_updated:
            telemetry.record(TelemetrySample(d, ram_used_gb=3.1, temperature_c=61.0, latency_ms=120.0))
        state = rollout.advance()
        print(f"advanced: phase={state.phase}, devices={len(state.devices_updated)}")


if __name__ == "__main__":
    main()
