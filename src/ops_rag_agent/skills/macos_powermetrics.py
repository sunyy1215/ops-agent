from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from ops_rag_agent.observability import trace_skill_call
from ops_rag_agent.skills.base import SkillKind, SkillSpec


def _truncate(text: str, *, max_chars: int) -> str:
    value = text or ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n<truncated>\n"


def _first_float(patterns: list[str], text: str) -> float | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    return None


def _first_str(patterns: list[str], text: str) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return str(m.group(1))
    return None


def _all_floats(pattern: str, text: str) -> list[float]:
    out: list[float] = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
        try:
            out.append(float(m.group(1)))
        except Exception:
            continue
    return out


def parse_powermetrics(text: str) -> dict[str, Any]:
    raw = text or ""
    # Heuristic multi-candidate keys: names drift across macOS versions.
    cpu_temp_c = _first_float(
        [
            r"(?:CPU).*?(?:die|temp|temperature).*?([0-9]+(?:\.[0-9]+)?)\s*C",
            r"(?:eCPU|pCPU).*?(?:die|temp|temperature).*?([0-9]+(?:\.[0-9]+)?)\s*C",
        ],
        raw,
    )
    gpu_temp_c = _first_float(
        [
            r"(?:GPU).*?(?:die|temp|temperature).*?([0-9]+(?:\.[0-9]+)?)\s*C",
        ],
        raw,
    )
    # Some outputs use "Fan:" or "Fan 0:" etc.
    fan_rpms = _all_floats(r"Fan[^0-9]*([0-9]{3,6})\s*rpm", raw)
    thermal_pressure = _first_str(
        [
            r"Thermal.*Pressure.*?(Nominal|Moderate|Heavy|Critical)",
            r"thermal_pressure[:\s]+(nominal|moderate|heavy|critical)",
        ],
        raw,
    )
    cpu_power_w = _first_float(
        [
            r"(?:CPU).*?(?:Power|power).*?([0-9]+(?:\.[0-9]+)?)\s*W",
        ],
        raw,
    )
    gpu_power_w = _first_float(
        [
            r"(?:GPU).*?(?:Power|power).*?([0-9]+(?:\.[0-9]+)?)\s*W",
        ],
        raw,
    )
    package_power_w = _first_float(
        [
            r"(?:Package|SoC).*?(?:Power|power).*?([0-9]+(?:\.[0-9]+)?)\s*W",
        ],
        raw,
    )
    cpu_freq_mhz = _first_float(
        [
            r"(?:CPU).*?(?:frequency|freq).*?([0-9]+(?:\.[0-9]+)?)\s*MHz",
            r"Average\s+frequency.*?([0-9]+(?:\.[0-9]+)?)\s*MHz",
        ],
        raw,
    )
    return {
        "cpu_temp_c": cpu_temp_c,
        "gpu_temp_c": gpu_temp_c,
        "fan_rpm": fan_rpms or None,
        "thermal_pressure": thermal_pressure.lower() if thermal_pressure else None,
        "cpu_power_w": cpu_power_w,
        "gpu_power_w": gpu_power_w,
        "package_power_w": package_power_w,
        "cpu_freq_mhz": cpu_freq_mhz,
    }


@dataclass
class MacosPowermetricsSkill:
    spec: SkillSpec = SkillSpec(
        skill_id="ops.macos.powermetrics",
        name="macOS Powermetrics (SMC)",
        description=(
            "Collect local thermals/power/frequency/fans via `sudo -n /usr/bin/powermetrics` "
            "(read-only, fixed argument set)."
        ),
        version="1.0.0",
        business_domain="ops",
        kind=SkillKind.REGULAR,
        requires_approval=True,
        is_readonly=True,
        timeout_s=25,
        tags=("ops", "macos", "readonly", "sudo", "powermetrics"),
        when_to_use=(
            "需要读取 macOS 温度、风扇转速、功耗、CPU/GPU 频率等 SMC 指标时使用。"
            "调用 sudo -n /usr/bin/powermetrics --samplers smc -n 1，需显式审批，不要频繁调用。"
        ),
        argument_schema={
            "type": "object",
            "properties": {
                "approved": {
                    "type": "boolean",
                    "description": "是否已获得人工审批执行 sudo powermetrics。",
                    "default": False,
                },
                "show_process_energy": {
                    "type": "boolean",
                    "description": "是否附加 --show-process-energy 采集各进程能耗。",
                    "default": False,
                },
            },
            "required": ["approved"],
        },
        example_invocations=(
            {"approved": True},
            {"approved": True, "show_process_energy": True},
        ),
        risk_level="high",
    )

    @trace_skill_call("ops.macos.powermetrics")
    def invoke(self, arguments: dict[str, Any]) -> str:
        approved = bool(arguments.get("approved", False))
        if not approved:
            return "APPROVAL_REQUIRED: sudo powermetrics requires explicit collection authorization"

        show_process_energy = bool(arguments.get("show_process_energy", False))

        argv = [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/powermetrics",
            "--samplers",
            "smc",
            "-n",
            "1",
        ]
        if show_process_energy:
            argv.append("--show-process-energy")

        started = time.monotonic()
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.spec.timeout_s,
        )
        duration_ms = int((time.monotonic() - started) * 1000)

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        raw_text = (stdout + ("\n" + stderr if stderr else "")).strip()

        parsed: dict[str, Any] = {}
        parse_error: str | None = None
        try:
            parsed = parse_powermetrics(stdout)
        except Exception as exc:  # noqa: BLE001 - best-effort parsing
            parse_error = f"{type(exc).__name__}: {exc}"

        payload = {
            "timestamp": time.time(),
            "command": " ".join(argv),
            "duration_ms": duration_ms,
            "exit_code": completed.returncode,
            "stderr_excerpt": _truncate(stderr, max_chars=2000),
            "raw_excerpt": _truncate(raw_text, max_chars=6000),
            "parsed": parsed,
            "parse_error": parse_error,
        }
        return json.dumps(payload, ensure_ascii=True)

