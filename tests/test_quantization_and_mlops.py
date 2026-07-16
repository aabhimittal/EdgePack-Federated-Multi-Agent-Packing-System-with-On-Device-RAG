import json

import pytest

from edgepack.mlops import ModelRegistry, RolloutManager, TelemetryCollector, TelemetrySample
from edgepack.quantization import QuantizationConfig, QuantizationWorkflow, RAMBudget, ThermalBudget


# ------------------------------------------------------------------- budgets
def test_ram_estimate_4bit_7b_fits_6gb():
    fits, est = RAMBudget(6.0).check(7.2, bits=4, context_len=4096, n_layers=32, hidden=4096)
    assert fits
    assert est["weights_gb"] == pytest.approx(3.6, abs=0.1)


def test_ram_estimate_fp16_7b_does_not_fit_6gb():
    fits, est = RAMBudget(6.0).check(7.2, bits=16, context_len=4096, n_layers=32, hidden=4096)
    assert not fits
    assert est["total_gb"] > 14


def test_ram_estimate_70b_4bit_needs_more_than_6gb():
    fits, _ = RAMBudget(6.0).check(70.6, bits=4, context_len=4096, n_layers=80, hidden=8192)
    assert not fits


def test_thermal_check_never_hard_fails_without_sensor(monkeypatch):
    tb = ThermalBudget(max_celsius=85.0)
    monkeypatch.setattr(tb, "read_temperature", lambda: None)
    ok, detail = tb.check()
    assert ok and "unknown" in detail["status"]
    monkeypatch.setattr(tb, "read_temperature", lambda: 95.0)
    ok, _ = tb.check()
    assert not ok


# ------------------------------------------------------------------ workflow
def test_workflow_dry_run_mistral_7b(tmp_path):
    cfg = QuantizationConfig(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        out_dir=str(tmp_path), ram_limit_gb=6.0, export_coreml=True, dry_run=True,
    )
    result = QuantizationWorkflow(cfg).run()
    assert result.ok
    names = [s.name for s in result.stages]
    assert names == ["ram_budget", "thermal_budget", "mlx_quantize", "coreml_export", "verify"]
    report = json.loads((result.artifact_dir / "REPORT.json").read_text())
    assert report["ok"]
    plan = json.loads((result.artifact_dir / "mlx" / "PLAN.json").read_text())
    assert plan["args"]["q_bits"] == 4


def test_workflow_fails_fast_when_over_ram_budget(tmp_path):
    cfg = QuantizationConfig(
        model_id="llama-3-70b", out_dir=str(tmp_path), ram_limit_gb=6.0, dry_run=True,
    )
    result = QuantizationWorkflow(cfg).run()
    assert not result.ok
    assert result.artifact_dir is None  # never got past budget checks
    assert [s.name for s in result.stages] == ["ram_budget", "thermal_budget"]


# ------------------------------------------------------------------ registry
def test_registry_register_get_promote(tmp_path):
    artifact = tmp_path / "model.json"
    artifact.write_text('{"weights": "stub"}')
    reg = ModelRegistry(tmp_path / "registry")
    mv1 = reg.register("packer", artifact, metrics={"real_utilization": 0.3})
    assert mv1.version == 1 and mv1.stage == "staging"

    promoted = reg.promote("packer", 1, min_metrics={"real_utilization": 0.15})
    assert promoted.stage == "production"
    assert reg.get("packer", stage="production").version == 1
    assert reg.verify_artifact("packer", 1)


def test_registry_blocks_promotion_below_threshold(tmp_path):
    artifact = tmp_path / "model.json"
    artifact.write_text("{}")
    reg = ModelRegistry(tmp_path / "registry")
    reg.register("packer", artifact, metrics={"real_utilization": 0.05})
    with pytest.raises(ValueError, match="promotion blocked"):
        reg.promote("packer", 1, min_metrics={"real_utilization": 0.15})


def test_registry_detects_artifact_tampering(tmp_path):
    artifact = tmp_path / "model.json"
    artifact.write_text("{}")
    reg = ModelRegistry(tmp_path / "registry")
    reg.register("packer", artifact)
    artifact.write_text('{"tampered": true}')
    assert not reg.verify_artifact("packer", 1)


# ------------------------------------------------------------------- rollout
def _healthy(t, devices):
    for d in devices:
        t.record(TelemetrySample(d, ram_used_gb=3.0, temperature_c=60.0, latency_ms=100.0))


def test_rollout_canary_to_full(tmp_path):
    artifact = tmp_path / "m.json"
    artifact.write_text("{}")
    reg = ModelRegistry(tmp_path / "reg")
    mv = reg.register("packer", artifact)
    fleet = [f"d{i}" for i in range(20)]
    telemetry = TelemetryCollector()
    rm = RolloutManager(reg, telemetry, fleet)

    state = rm.start("packer", mv.version)
    assert state.phase == "canary" and len(state.devices_updated) == 1

    _healthy(telemetry, state.devices_updated)
    state = rm.advance()
    assert state.phase == "staged" and len(state.devices_updated) == 10

    _healthy(telemetry, state.devices_updated)
    state = rm.advance()
    assert state.phase == "full" and len(state.devices_updated) == 20


def test_rollout_rolls_back_on_unhealthy_canary(tmp_path):
    artifact = tmp_path / "m.json"
    artifact.write_text("{}")
    reg = ModelRegistry(tmp_path / "reg")
    mv = reg.register("packer", artifact)
    telemetry = TelemetryCollector(temp_limit_c=85.0)
    rm = RolloutManager(reg, telemetry, [f"d{i}" for i in range(10)])

    state = rm.start("packer", mv.version)
    for d in state.devices_updated:  # canary overheats
        telemetry.record(TelemetrySample(d, ram_used_gb=3.0, temperature_c=99.0, latency_ms=100.0))
    state = rm.advance()
    assert state.phase == "rolled_back"
    assert state.devices_updated == []
