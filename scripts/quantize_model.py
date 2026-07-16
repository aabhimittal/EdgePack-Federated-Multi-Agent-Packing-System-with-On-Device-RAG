#!/usr/bin/env python3
"""Quantize a large model to 4-bit for Apple Silicon under RAM/thermal budgets.

On an M-series Mac with `mlx-lm` installed this performs the real conversion;
elsewhere it runs the identical pipeline in dry-run mode (budget checks, plan
files, report) so the workflow is testable anywhere.

Usage:
    python scripts/quantize_model.py --model mistralai/Mistral-7B-Instruct-v0.3 \
        --ram-limit 6 --bits 4 --coreml
    python scripts/quantize_model.py --model meta-llama/Meta-Llama-3-8B-Instruct --ram-limit 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgepack.mlops import ModelRegistry
from edgepack.quantization import HAS_MLX, QuantizationConfig, QuantizationWorkflow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Hugging Face model id")
    ap.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--ram-limit", type=float, default=6.0, help="GB")
    ap.add_argument("--thermal-limit", type=float, default=85.0, help="celsius")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--coreml", action="store_true", help="also export a Core ML palettized plan/model")
    ap.add_argument("--out", default="quantized")
    ap.add_argument("--registry", default="registry")
    args = ap.parse_args()

    cfg = QuantizationConfig(
        model_id=args.model,
        out_dir=args.out,
        q_bits=args.bits,
        q_group_size=args.group_size,
        ram_limit_gb=args.ram_limit,
        thermal_limit_c=args.thermal_limit,
        context_len=args.context,
        export_coreml=args.coreml,
    )
    wf = QuantizationWorkflow(cfg)
    print(f"mode: {'REAL (mlx available)' if HAS_MLX and not wf.dry_run else 'DRY-RUN'}")
    result = wf.run()

    print(json.dumps(result.report(), indent=2))
    if not result.ok:
        print("\nworkflow FAILED (over budget or verification failed)", file=sys.stderr)
        sys.exit(1)

    report_path = result.artifact_dir / "REPORT.json"
    registry = ModelRegistry(args.registry)
    mv = registry.register(
        name=args.model.split("/")[-1] + f"-q{args.bits}",
        artifact_path=report_path,
        metrics={"ram_estimate_gb": result.stages[-1].detail.get("ram_estimate_gb", 0.0)},
    )
    print(f"\nregistered {mv.name} v{mv.version} (sha256 {mv.sha256[:12]}…) in {args.registry}/")


if __name__ == "__main__":
    main()
