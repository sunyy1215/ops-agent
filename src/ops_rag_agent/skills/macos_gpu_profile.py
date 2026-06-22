from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

from ops_rag_agent.observability import trace_skill_call
from ops_rag_agent.skills.base import SkillKind, SkillSpec


def _truncate(text: str, *, max_chars: int) -> str:
    value = text or ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n<truncated>\n"


@dataclass
class MacosGpuProfileSkill:
    spec: SkillSpec = SkillSpec(
        skill_id="ops.macos.gpu_profile",
        name="macOS GPU Profile",
        description="Collect local GPU/display context via system_profiler (read-only).",
        version="1.0.0",
        business_domain="ops",
        kind=SkillKind.REGULAR,
        requires_approval=False,
        is_readonly=True,
        timeout_s=20,
        tags=("ops", "macos", "readonly", "gpu"),
        when_to_use=(
            "需要了解 macOS 本机 GPU 型号、显存、Metal 支持及显示器连接情况时使用。"
            "只读调用 system_profiler SPDisplaysDataType，返回结构化 GPU/显示器信息。"
        ),
        argument_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        example_invocations=(
            {},
        ),
        risk_level="low",
    )

    @trace_skill_call("ops.macos.gpu_profile")
    def invoke(self, arguments: dict[str, Any]) -> str:
        # Prefer JSON output when available for stable parsing.
        argv = ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"]
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.spec.timeout_s,
        )
        raw = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
        raw_excerpt = _truncate(raw, max_chars=6000)

        parsed: dict[str, Any] = {
            "exit_code": completed.returncode,
            "gpus": [],
            "displays": [],
        }
        try:
            data = json.loads(completed.stdout or "{}")
            items = data.get("SPDisplaysDataType", []) if isinstance(data, dict) else []
            for item in items:
                # GPU blocks usually contain keys like sppci_model / spdisplays_vendor / spdisplays_vram
                if not isinstance(item, dict):
                    continue
                model = item.get("sppci_model") or item.get("spdisplays_gpu_model") or ""
                vram = item.get("spdisplays_vram") or ""
                vendor = item.get("spdisplays_vendor") or item.get("sppci_vendor") or ""
                metal = item.get("spdisplays_metal") or ""
                parsed["gpus"].append(
                    {
                        "model": str(model),
                        "vendor": str(vendor),
                        "vram": str(vram),
                        "metal": str(metal),
                    }
                )
                # Displays may appear under a sub-key.
                displays = item.get("spdisplays_ndrvs", [])
                if isinstance(displays, list):
                    for d in displays:
                        if not isinstance(d, dict):
                            continue
                        parsed["displays"].append(
                            {
                                "name": str(d.get("_name", "") or d.get("spdisplays_display_type", "")),
                                "resolution": str(d.get("spdisplays_resolution", "")),
                                "pixel_depth": str(d.get("spdisplays_pixel_depth", "")),
                                "connection_type": str(d.get("spdisplays_connection_type", "")),
                                "main_display": bool(d.get("spdisplays_main") == "spdisplays_yes"),
                            }
                        )
        except Exception as exc:  # noqa: BLE001 - best-effort parsing
            parsed["parse_error"] = f"{type(exc).__name__}: {exc}"

        payload = {
            "timestamp": __import__("time").time(),
            "command": " ".join(argv),
            "raw_excerpt": raw_excerpt,
            "parsed": parsed,
        }
        return json.dumps(payload, ensure_ascii=True)

