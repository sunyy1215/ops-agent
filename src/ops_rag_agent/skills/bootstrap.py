from ops_rag_agent.skills.bash_readonly import BashReadonlySkill
from ops_rag_agent.skills.local_host import LocalHostSnapshotSkill
from ops_rag_agent.skills.macos_gpu_profile import MacosGpuProfileSkill
from ops_rag_agent.skills.macos_metrics import MacosMetricsSkill
from ops_rag_agent.skills.macos_powermetrics import MacosPowermetricsSkill
from ops_rag_agent.skills.macos_timeseries_probe import MacosTimeseriesProbeSkill
from ops_rag_agent.skills.prometheus import PrometheusQuerySkill
from ops_rag_agent.skills.rag_search import RagSearchSkill
from ops_rag_agent.skills.remote_host_snapshot import RemoteHostSnapshotSkill
from ops_rag_agent.skills.remote_service_inspect import RemoteServiceInspectSkill
from ops_rag_agent.skills.remote_ssh_exec import RemoteSshExecSkill
from ops_rag_agent.skills.registry import SkillRegistry
from ops_rag_agent.skills.terminal_exec import TerminalExecSkill
from ops_rag_agent.skills.web_search import WebSearchSkill


def build_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    # 知识库 / 联网检索类
    registry.register(RagSearchSkill())
    registry.register(WebSearchSkill())
    # 监控 / 指标
    registry.register(PrometheusQuerySkill())
    # 本机 / macOS 排查
    registry.register(LocalHostSnapshotSkill())
    registry.register(BashReadonlySkill())
    registry.register(TerminalExecSkill())
    registry.register(MacosMetricsSkill())
    registry.register(MacosPowermetricsSkill())
    registry.register(MacosGpuProfileSkill())
    registry.register(MacosTimeseriesProbeSkill())
    # 远程主机
    registry.register(RemoteSshExecSkill())
    registry.register(RemoteHostSnapshotSkill())
    registry.register(RemoteServiceInspectSkill())
    return registry
