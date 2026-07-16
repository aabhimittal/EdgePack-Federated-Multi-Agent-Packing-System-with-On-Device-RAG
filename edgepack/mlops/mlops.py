"""Edge MLOps plane: model registry, device telemetry, staged OTA rollout.

- **ModelRegistry** — JSON-file store of versioned artifacts (packing-policy
  bundles from MARL-Sim2real, quantized LLMs from the quantization workflow)
  with SHA-256 hashes and metrics.  Promotion gates on the metrics.
- **TelemetryCollector** — per-device RAM / thermal / latency samples with
  simple health rules.
- **RolloutManager** — canary -> staged -> full OTA rollout with automatic
  rollback when canary health degrades.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path


# ------------------------------------------------------------------ registry
@dataclass
class ModelVersion:
    name: str
    version: int
    artifact_path: str
    sha256: str
    metrics: dict
    stage: str = "staging"   # staging -> production -> archived
    created_unix: int = 0


class ModelRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        self._index: dict[str, list[dict]] = {}
        if self.index_path.exists():
            self._index = json.loads(self.index_path.read_text())

    def _save(self) -> None:
        self.index_path.write_text(json.dumps(self._index, indent=2))

    def register(self, name: str, artifact_path: str | Path, metrics: dict | None = None) -> ModelVersion:
        artifact_path = Path(artifact_path)
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        versions = self._index.setdefault(name, [])
        mv = ModelVersion(
            name=name,
            version=len(versions) + 1,
            artifact_path=str(artifact_path),
            sha256=digest,
            metrics=metrics or {},
            created_unix=int(time.time()),
        )
        versions.append(mv.__dict__)
        self._save()
        return mv

    def get(self, name: str, version: int | None = None, stage: str | None = None) -> ModelVersion:
        versions = self._index.get(name, [])
        if not versions:
            raise KeyError(f"no versions of {name!r}")
        if version is not None:
            for v in versions:
                if v["version"] == version:
                    return ModelVersion(**v)
            raise KeyError(f"{name} v{version} not found")
        if stage is not None:
            staged = [v for v in versions if v["stage"] == stage]
            if not staged:
                raise KeyError(f"no {name!r} version in stage {stage!r}")
            return ModelVersion(**staged[-1])
        return ModelVersion(**versions[-1])

    def promote(self, name: str, version: int, min_metrics: dict | None = None) -> ModelVersion:
        """Gate promotion to production on metric thresholds."""
        mv = self.get(name, version)
        for key, threshold in (min_metrics or {}).items():
            value = mv.metrics.get(key)
            if value is None or value < threshold:
                raise ValueError(f"promotion blocked: {key}={value} < required {threshold}")
        for v in self._index[name]:
            if v["stage"] == "production":
                v["stage"] = "archived"
            if v["version"] == version:
                v["stage"] = "production"
        self._save()
        return self.get(name, version)

    def verify_artifact(self, name: str, version: int | None = None) -> bool:
        mv = self.get(name, version)
        return hashlib.sha256(Path(mv.artifact_path).read_bytes()).hexdigest() == mv.sha256


# ----------------------------------------------------------------- telemetry
@dataclass
class TelemetrySample:
    device_id: str
    ram_used_gb: float
    temperature_c: float
    latency_ms: float
    ts: float = 0.0


class TelemetryCollector:
    def __init__(self, ram_limit_gb: float = 6.0, temp_limit_c: float = 85.0,
                 latency_limit_ms: float = 500.0):
        self.samples: list[TelemetrySample] = []
        self.limits = {"ram": ram_limit_gb, "temp": temp_limit_c, "latency": latency_limit_ms}

    def record(self, sample: TelemetrySample) -> None:
        self.samples.append(sample)

    def device_health(self, device_id: str, window: int = 10) -> dict:
        recent = [s for s in self.samples if s.device_id == device_id][-window:]
        if not recent:
            return {"device_id": device_id, "healthy": True, "reason": "no data"}
        violations = []
        avg_ram = sum(s.ram_used_gb for s in recent) / len(recent)
        avg_temp = sum(s.temperature_c for s in recent) / len(recent)
        avg_lat = sum(s.latency_ms for s in recent) / len(recent)
        if avg_ram > self.limits["ram"]:
            violations.append(f"ram {avg_ram:.1f}GB > {self.limits['ram']}GB")
        if avg_temp > self.limits["temp"]:
            violations.append(f"temp {avg_temp:.0f}C > {self.limits['temp']}C")
        if avg_lat > self.limits["latency"]:
            violations.append(f"latency {avg_lat:.0f}ms > {self.limits['latency']}ms")
        return {"device_id": device_id, "healthy": not violations, "reason": "; ".join(violations) or "ok",
                "avg_ram_gb": avg_ram, "avg_temp_c": avg_temp, "avg_latency_ms": avg_lat}


# ------------------------------------------------------------------- rollout
@dataclass
class RolloutState:
    model_name: str
    version: int
    phase: str = "canary"          # canary -> staged -> full | rolled_back
    devices_updated: list[str] = field(default_factory=list)


class RolloutManager:
    """Canary -> staged -> full OTA rollout with health-gated progression."""

    PHASES = {"canary": 0.05, "staged": 0.5, "full": 1.0}

    def __init__(self, registry: ModelRegistry, telemetry: TelemetryCollector,
                 fleet: list[str]):
        self.registry = registry
        self.telemetry = telemetry
        self.fleet = list(fleet)
        self.state: RolloutState | None = None

    def start(self, model_name: str, version: int) -> RolloutState:
        if not self.registry.verify_artifact(model_name, version):
            raise ValueError("artifact hash mismatch — refusing to roll out")
        self.state = RolloutState(model_name=model_name, version=version)
        self._push_to_fraction(self.PHASES["canary"])
        return self.state

    def _push_to_fraction(self, fraction: float) -> None:
        assert self.state is not None
        target = max(1, int(len(self.fleet) * fraction))
        for device in self.fleet[:target]:
            if device not in self.state.devices_updated:
                self.state.devices_updated.append(device)

    def advance(self) -> RolloutState:
        """Check health of updated devices; advance a phase or roll back."""
        assert self.state is not None, "no rollout in progress"
        unhealthy = [
            h for d in self.state.devices_updated
            if not (h := self.telemetry.device_health(d))["healthy"]
        ]
        if unhealthy:
            self.state.phase = "rolled_back"
            self.state.devices_updated.clear()
            return self.state
        if self.state.phase == "canary":
            self.state.phase = "staged"
            self._push_to_fraction(self.PHASES["staged"])
        elif self.state.phase == "staged":
            self.state.phase = "full"
            self._push_to_fraction(self.PHASES["full"])
        return self.state
