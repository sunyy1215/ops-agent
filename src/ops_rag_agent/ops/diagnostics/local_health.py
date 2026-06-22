from __future__ import annotations

import json
from typing import Any, Literal, TypedDict


Priority = Literal["P0", "P1", "P2"]


class Hypothesis(TypedDict, total=False):
    id: str
    description: str
    confidence: float
    evidence_refs: list[str]
    next_checks: list[str]


class Recommendation(TypedDict, total=False):
    priority: Priority
    action: str
    rationale: str
    evidence_refs: list[str]
    suggested_commands: list[str]


def _safe_json_loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(str(value))
    except Exception:
        return {}


def diagnose_local_health(*, evidence: dict[str, Any]) -> dict[str, Any]:
    """
    Lightweight rule engine for macOS local health "deep source" diagnosis.

    Input evidence is expected to include:
    - metrics: output of ops.macos.metrics (JSON dict)
    - powermetrics: output of ops.macos.powermetrics (JSON dict), optional
    - gpu_profile: output of ops.macos.gpu_profile (JSON dict), optional
    - timeseries: output of ops.macos.timeseries_probe (JSON dict), optional
    """

    metrics = _safe_json_loads(evidence.get("metrics"))
    power = _safe_json_loads(evidence.get("powermetrics"))
    gpu_profile = _safe_json_loads(evidence.get("gpu_profile"))
    timeseries = _safe_json_loads(evidence.get("timeseries"))

    cpu_percent = float(metrics.get("cpu", {}).get("percent_total") or 0.0)
    load_1 = metrics.get("cpu", {}).get("loadavg", {}).get("1m")
    mem_percent = float(metrics.get("memory", {}).get("percent") or 0.0)
    swap_used = float(metrics.get("swap", {}).get("used_bytes") or 0.0)

    thermal_pressure = (
        (power.get("parsed", {}) or {}).get("thermal_pressure")
        if isinstance(power.get("parsed", {}), dict)
        else None
    )
    cpu_temp_c = (
        (power.get("parsed", {}) or {}).get("cpu_temp_c")
        if isinstance(power.get("parsed", {}), dict)
        else None
    )
    cpu_freq_mhz = (
        (power.get("parsed", {}) or {}).get("cpu_freq_mhz")
        if isinstance(power.get("parsed", {}), dict)
        else None
    )

    hypotheses: list[Hypothesis] = []
    recs: list[Recommendation] = []

    # Hypothesis family 1: sustained CPU hot process.
    top_procs = metrics.get("top_processes", []) if isinstance(metrics.get("top_processes"), list) else []
    top_proc = top_procs[0] if top_procs else {}
    top_proc_cpu = float(top_proc.get("cpu_percent") or 0.0) if isinstance(top_proc, dict) else 0.0
    if cpu_percent >= 75.0 and top_proc_cpu >= 20.0:
        hypotheses.append(
            {
                "id": "cpu_hot_process",
                "description": "CPU is dominated by one or a few hot processes (likely sustained load).",
                "confidence": 0.78,
                "evidence_refs": ["metrics.cpu.percent_total", "metrics.top_processes[0]"],
                "next_checks": [
                    "Collect a short timeseries to distinguish sustained load vs spikes.",
                    "Capture a sample/stack trace of the hot PID.",
                ],
            }
        )
        pid = int(top_proc.get("pid") or 0) if isinstance(top_proc, dict) else 0
        if pid:
            recs.append(
                {
                    "priority": "P0",
                    "action": f"Identify why PID {pid} is consuming CPU",
                    "rationale": "Top process CPU is high; root-cause is usually process-specific (indexing, builds, browser tabs, containers).",
                    "evidence_refs": ["metrics.top_processes[0]"],
                    "suggested_commands": [
                        f"ps -p {pid} -o pid,ppid,pcpu,pmem,etime,command",
                        f"sample {pid} 5 -file /tmp/sample_{pid}.txt",
                    ],
                }
            )

    # Hypothesis family 2: thermal throttling / thermal pressure.
    if thermal_pressure and thermal_pressure != "nominal":
        hypotheses.append(
            {
                "id": "thermal_pressure",
                "description": "System is under thermal pressure; throttling can cause slowdown even when CPU usage is not extreme.",
                "confidence": 0.74,
                "evidence_refs": ["powermetrics.parsed.thermal_pressure", "powermetrics.parsed.cpu_temp_c", "powermetrics.parsed.cpu_freq_mhz"],
                "next_checks": [
                    "Run a short timeseries probe to confirm thermal pressure persists.",
                    "Check which processes correlate with heat/power increase.",
                ],
            }
        )
        recs.append(
            {
                "priority": "P0",
                "action": "Confirm throttling trend and reduce heat sources temporarily",
                "rationale": "Non-nominal thermal pressure often explains perceived slowness via reduced CPU frequency.",
                "evidence_refs": ["powermetrics.parsed.thermal_pressure"],
                "suggested_commands": [
                    "sudo -n /usr/bin/powermetrics --samplers smc -n 1",
                    "pmset -g thermlog | tail -n 50",
                ],
            }
        )

    # Hypothesis family 3: memory pressure / swap thrashing.
    if mem_percent >= 85.0 or swap_used > 1_000_000_000:
        hypotheses.append(
            {
                "id": "memory_pressure",
                "description": "Memory pressure is high; swap activity can degrade responsiveness.",
                "confidence": 0.7,
                "evidence_refs": ["metrics.memory.percent", "metrics.swap.used_bytes"],
                "next_checks": [
                    "Check top RSS processes and whether swap grows over time.",
                    "Collect a short timeseries to confirm swap growth trend.",
                ],
            }
        )
        recs.append(
            {
                "priority": "P0",
                "action": "Find top memory consumers and confirm swap growth",
                "rationale": "High memory pressure and swap often point to a few heavyweight apps (IDE, browser, containers).",
                "evidence_refs": ["metrics.top_processes", "metrics.swap.used_bytes"],
                "suggested_commands": [
                    "ps -axo pid,comm,rss,vsz,%mem | sort -nrk3 | head",
                    "vm_stat 1 5",
                ],
            }
        )

    # GPU context hint (best-effort).
    if gpu_profile.get("parsed", {}).get("displays") and isinstance(gpu_profile.get("parsed", {}).get("displays"), list):
        recs.append(
            {
                "priority": "P2",
                "action": "If using external/high-refresh displays, consider reducing refresh rate during heavy workloads",
                "rationale": "External displays can increase GPU load and thermal output; this is context-dependent.",
                "evidence_refs": ["gpu_profile.parsed.displays"],
                "suggested_commands": [
                    "/usr/sbin/system_profiler SPDisplaysDataType",
                ],
            }
        )

    # Verify trigger suggestions based on uncertainty.
    should_verify = False
    verify_reason: list[str] = []
    if cpu_percent >= 60.0 and not timeseries:
        should_verify = True
        verify_reason.append("cpu_high_need_trend")
    if thermal_pressure and thermal_pressure != "nominal" and not timeseries:
        should_verify = True
        verify_reason.append("thermal_pressure_need_trend")

    summary = {
        "snapshot": {
            "cpu_percent": cpu_percent,
            "load_1m": load_1,
            "mem_percent": mem_percent,
            "swap_used_bytes": swap_used,
            "thermal_pressure": thermal_pressure,
            "cpu_temp_c": cpu_temp_c,
            "cpu_freq_mhz": cpu_freq_mhz,
        },
        "verify_suggested": should_verify,
        "verify_reason": verify_reason,
    }
    return {
        "summary": summary,
        "hypotheses": hypotheses[:3],
        "recommendations": _dedup_recommendations(recs),
    }


def _dedup_recommendations(items: list[Recommendation]) -> list[Recommendation]:
    seen: set[tuple[str, str]] = set()
    out: list[Recommendation] = []
    for item in items:
        key = (str(item.get("priority", "")), str(item.get("action", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    # Keep P0 first, then P1, then P2.
    prio_rank = {"P0": 0, "P1": 1, "P2": 2}
    out.sort(key=lambda x: prio_rank.get(str(x.get("priority", "P2")), 99))
    return out

