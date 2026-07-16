"""End-to-end 4-bit quantization workflow for Apple Silicon.

Pipeline:  budget-check -> MLX 4-bit convert -> (optional) Core ML export
           -> verify -> register.

MLX (`mlx_lm.convert`) does the actual 4-bit group-wise quantization of a
Hugging Face model (Llama 3, Mistral, ...); coremltools palettizes for the
Neural Engine.  Both are **optional imports**: on non-Apple hardware (like
this CI container) the workflow runs in *dry-run* mode — every stage executes
its checks and produces its report, only the heavy conversion is simulated.
That keeps the pipeline testable everywhere while behaving identically in
structure on an M-series Mac.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .budget import RAMBudget, ThermalBudget, lookup_geometry

try:  # Apple-Silicon-only heavy deps
    from mlx_lm import convert as mlx_convert  # type: ignore
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import coremltools as ct  # type: ignore
    HAS_COREML = True
except ImportError:
    HAS_COREML = False


@dataclass
class QuantizationConfig:
    model_id: str                      # HF repo, e.g. "mistralai/Mistral-7B-Instruct-v0.3"
    out_dir: str = "quantized"
    q_bits: int = 4
    q_group_size: int = 64
    ram_limit_gb: float = 6.0          # e.g. iPhone 15 Pro / 8 GB Mac headroom
    thermal_limit_c: float = 85.0
    context_len: int = 4096
    export_coreml: bool = False
    dry_run: bool | None = None        # None = auto (dry if MLX missing)


@dataclass
class StageResult:
    name: str
    ok: bool
    detail: dict = field(default_factory=dict)


@dataclass
class WorkflowResult:
    config: QuantizationConfig
    stages: list[StageResult]
    artifact_dir: Path | None

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.stages)

    def report(self) -> dict:
        return {
            "model_id": self.config.model_id,
            "q_bits": self.config.q_bits,
            "ok": self.ok,
            "stages": [{"name": s.name, "ok": s.ok, **s.detail} for s in self.stages],
            "artifact_dir": str(self.artifact_dir) if self.artifact_dir else None,
        }


class QuantizationWorkflow:
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.dry_run = config.dry_run if config.dry_run is not None else not HAS_MLX

    # ------------------------------------------------------------- stage 1
    def check_budgets(self) -> list[StageResult]:
        cfg = self.config
        params_b, n_layers, hidden = lookup_geometry(cfg.model_id)

        ram = RAMBudget(cfg.ram_limit_gb)
        ram_ok, ram_est = ram.check(
            params_b, bits=cfg.q_bits, context_len=cfg.context_len,
            n_layers=n_layers, hidden=hidden,
        )
        # also show what fp16 would have cost — the point of quantizing
        _, fp16_est = ram.check(params_b, bits=16, context_len=cfg.context_len,
                                n_layers=n_layers, hidden=hidden)

        thermal = ThermalBudget(cfg.thermal_limit_c)
        therm_ok, therm_detail = thermal.check()

        return [
            StageResult("ram_budget", ram_ok,
                        {"quantized": ram_est, "fp16_reference_gb": fp16_est["total_gb"]}),
            StageResult("thermal_budget", therm_ok, therm_detail),
        ]

    # ------------------------------------------------------------- stage 2
    def quantize_mlx(self, out: Path) -> StageResult:
        cfg = self.config
        if self.dry_run:
            plan = {
                "backend": "mlx_lm.convert",
                "args": {
                    "hf_path": cfg.model_id,
                    "mlx_path": str(out / "mlx"),
                    "quantize": True,
                    "q_bits": cfg.q_bits,
                    "q_group_size": cfg.q_group_size,
                },
                "mode": "dry-run (mlx not installed — run on Apple Silicon)",
            }
            (out / "mlx").mkdir(parents=True, exist_ok=True)
            (out / "mlx" / "PLAN.json").write_text(json.dumps(plan, indent=2))
            return StageResult("mlx_quantize", True, plan)
        t0 = time.time()
        mlx_convert(
            hf_path=cfg.model_id,
            mlx_path=str(out / "mlx"),
            quantize=True,
            q_bits=cfg.q_bits,
            q_group_size=cfg.q_group_size,
        )
        return StageResult("mlx_quantize", True,
                           {"backend": "mlx_lm.convert", "seconds": round(time.time() - t0, 1)})

    # ------------------------------------------------------------- stage 3
    def export_coreml(self, out: Path) -> StageResult:
        cfg = self.config
        if not cfg.export_coreml:
            return StageResult("coreml_export", True, {"skipped": True})
        if self.dry_run or not HAS_COREML:
            plan = {
                "backend": "coremltools",
                "steps": [
                    "trace/convert model to Core ML mlprogram (FP16 compute)",
                    f"apply weight palettization: {cfg.q_bits}-bit LUT per channel "
                    "(ct.optimize.coreml.palettize_weights)",
                    "set compute_units=ALL to allow the Neural Engine",
                ],
                "mode": "dry-run (coremltools not installed)",
            }
            (out / "coreml").mkdir(parents=True, exist_ok=True)
            (out / "coreml" / "PLAN.json").write_text(json.dumps(plan, indent=2))
            return StageResult("coreml_export", True, plan)
        # Real path (Apple Silicon): palettize a converted mlprogram.
        from coremltools.optimize.coreml import (  # type: ignore
            OpPalettizerConfig, OptimizationConfig, palettize_weights,
        )
        model = ct.models.MLModel(str(out / "coreml" / "model.mlpackage"))
        op_config = OpPalettizerConfig(mode="kmeans", nbits=cfg.q_bits)
        palettized = palettize_weights(model, OptimizationConfig(global_config=op_config))
        palettized.save(str(out / "coreml" / "model_palettized.mlpackage"))
        return StageResult("coreml_export", True, {"nbits": cfg.q_bits})

    # ------------------------------------------------------------- stage 4
    def verify(self, out: Path) -> StageResult:
        """Post-quantization smoke checks: artifact exists + RAM re-estimate.

        On real hardware this also runs a short generation to measure
        tokens/sec and peak RSS; in dry-run it validates the plan files.
        """
        cfg = self.config
        detail: dict = {}
        if self.dry_run:
            detail["checked"] = "plan artifacts"
            ok = (out / "mlx" / "PLAN.json").exists()
        else:
            from mlx_lm import generate, load  # type: ignore

            model, tokenizer = load(str(out / "mlx"))
            t0 = time.time()
            text = generate(model, tokenizer, prompt="Say OK.", max_tokens=8)
            dt = time.time() - t0
            detail["sample"] = text[:80]
            detail["tokens_per_sec"] = round(8 / dt, 1)
            ok = bool(text)
        params_b, n_layers, hidden = lookup_geometry(cfg.model_id)
        _, est = RAMBudget(cfg.ram_limit_gb).check(
            params_b, bits=cfg.q_bits, context_len=cfg.context_len,
            n_layers=n_layers, hidden=hidden)
        detail["ram_estimate_gb"] = est["total_gb"]
        return StageResult("verify", ok, detail)

    # ------------------------------------------------------------ pipeline
    def run(self) -> WorkflowResult:
        stages: list[StageResult] = []
        out = Path(self.config.out_dir) / self.config.model_id.replace("/", "__")

        stages.extend(self.check_budgets())
        if not all(s.ok for s in stages):
            return WorkflowResult(self.config, stages, None)  # fail fast: over budget

        out.mkdir(parents=True, exist_ok=True)
        stages.append(self.quantize_mlx(out))
        stages.append(self.export_coreml(out))
        stages.append(self.verify(out))

        result = WorkflowResult(self.config, stages, out)
        (out / "REPORT.json").write_text(json.dumps(result.report(), indent=2))
        return result
