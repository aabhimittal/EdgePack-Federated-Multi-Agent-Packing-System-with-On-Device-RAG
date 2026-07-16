"""RAM and thermal budget checks for on-device models.

The RAM model is analytic (works anywhere, including CI): weights at q bits +
KV cache + activation/runtime overhead.  The thermal monitor reads real
sensors where available (`/sys/class/thermal` on Linux; `powermetrics` hint on
macOS) and otherwise reports "unknown", which the workflow treats as a
soft-pass with a warning — budgets must never make CI flaky.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RAMBudget:
    limit_gb: float

    def estimate_model_gb(
        self,
        n_params_b: float,          # parameters, in billions
        bits: int = 4,
        context_len: int = 4096,
        n_layers: int = 32,
        hidden: int = 4096,
        kv_bits: int = 8,   # on-device runtimes quantize the KV cache too
        overhead_frac: float = 0.15,
    ) -> dict:
        weights_gb = n_params_b * 1e9 * bits / 8 / 1e9
        # KV cache: 2 (K+V) * layers * context * hidden * bytes
        kv_gb = 2 * n_layers * context_len * hidden * (kv_bits / 8) / 1e9
        overhead_gb = (weights_gb + kv_gb) * overhead_frac
        total = weights_gb + kv_gb + overhead_gb
        return {
            "weights_gb": round(weights_gb, 3),
            "kv_cache_gb": round(kv_gb, 3),
            "overhead_gb": round(overhead_gb, 3),
            "total_gb": round(total, 3),
            "limit_gb": self.limit_gb,
            "fits": total <= self.limit_gb,
        }

    def check(self, n_params_b: float, bits: int = 4, **kw) -> tuple[bool, dict]:
        est = self.estimate_model_gb(n_params_b, bits, **kw)
        return est["fits"], est


@dataclass
class ThermalBudget:
    max_celsius: float = 85.0

    def read_temperature(self) -> float | None:
        """Best-effort CPU/SoC temperature; None if no sensor is readable."""
        system = platform.system()
        if system == "Linux":
            for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
                try:
                    milli = int(zone.read_text().strip())
                    if milli > 1000:  # value in millidegrees
                        return milli / 1000.0
                    return float(milli)
                except (OSError, ValueError):
                    continue
        elif system == "Darwin" and shutil.which("powermetrics"):
            try:  # requires sudo on most machines; best effort only
                out = subprocess.run(
                    ["powermetrics", "-n", "1", "--samplers", "smc", "-i", "1"],
                    capture_output=True, text=True, timeout=10,
                ).stdout
                for line in out.splitlines():
                    if "CPU die temperature" in line:
                        return float(line.split(":")[1].strip().rstrip("C").strip())
            except (subprocess.SubprocessError, ValueError, OSError):
                return None
        return None

    def check(self) -> tuple[bool, dict]:
        temp = self.read_temperature()
        if temp is None:
            return True, {"temperature_c": None, "limit_c": self.max_celsius,
                          "status": "unknown-sensor (soft pass)"}
        return temp <= self.max_celsius, {
            "temperature_c": temp, "limit_c": self.max_celsius,
            "status": "ok" if temp <= self.max_celsius else "TOO HOT — defer quantization",
        }


# Known model geometries for the RAM estimator (params_b, layers, hidden)
MODEL_GEOMETRY = {
    "llama-3-8b": (8.0, 32, 4096),
    "llama-3-70b": (70.6, 80, 8192),
    "mistral-7b": (7.2, 32, 4096),
    "phi-3-mini": (3.8, 32, 3072),
}


def lookup_geometry(model_name: str) -> tuple[float, int, int]:
    key = model_name.lower()
    for name, geom in MODEL_GEOMETRY.items():
        if name in key:
            return geom
    # sensible default: assume 7B-class
    return MODEL_GEOMETRY["mistral-7b"]
