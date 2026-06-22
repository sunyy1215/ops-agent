from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "ops-rag-agent"
    environment: str = "dev"

    llm_api_base: str = Field(default="https://api.openai.com/v1")
    llm_api_key: str = Field(default="change-me")
    llm_model_router: str = Field(default="gpt-4o-mini")
    llm_model_chat: str = Field(default="gpt-4o-mini")
    llm_model_reasoning: str = Field(default="gpt-4.1")

    embedding_api_base: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_model: str = Field(default="text-embedding-3-large")

    rerank_model: Optional[str] = Field(default=None)
    rerank_top_k: int = Field(default=6)
    rerank_backend: str = Field(default="dashscope")  # "dashscope" or "placeholder"
    rerank_api_base: Optional[str] = None
    rerank_api_key: Optional[str] = None

    milvus_uri: str = Field(default="http://localhost:19530")
    milvus_token: Optional[str] = None
    milvus_collection: str = Field(default="knowledge_chunks")
    memory_collection: str = Field(default="long_term_memory")
    milvus_dense_dim: int = Field(default=1024)
    milvus_text_field: str = Field(default="text")
    milvus_sparse_field: str = Field(default="text_sparse")

    # Backends: "milvus" or "memory"/"placeholder"
    rag_vector_backend: str = Field(default="milvus")
    long_term_memory_backend: str = Field(default="milvus")
    rag_bm25_backend: str = Field(default="milvus_fulltext")  # "milvus_fulltext" or "local_bm25"

    retrieval_cache_ttl_seconds: int = Field(default=120)
    retrieval_cache_namespace: str = Field(default="rag.hybrid")
    retrieval_cache_redis_url: Optional[str] = None

    # 联网搜索（web.search skill）：可选自定义 SERP 接口，例如:
    #   "https://example.com/search?q={q}&top_k={top_k}"
    # 留空则只用 DuckDuckGo HTML 兜底。
    web_search_endpoint: Optional[str] = None
    ingestion_manifest_path: str = Field(default=".data/ingestion/manifest.sqlite")
    ingestion_noise_filter_enabled: bool = Field(default=True)
    ingestion_noise_drop_prefixes: str = Field(
        default="Table of Contents,Contents,目录,导航,Navigation,Next:,Previous:,上一页,下一页"
    )
    ingestion_noise_drop_regexes: str = Field(
        default=(
            r"^(?:Last updated|Updated at|Generated on|Generated at|更新时间|生成时间)[:：].*$,"
            r"^(?:Copyright|版权所有).*$,"
            r"^(?:Home|Docs|Documentation)\s*(?:>\s*[\w\-/\u4e00-\u9fff ]+){1,}$,"
            r"^\[.*\]\(#.*\)$"
        )
    )
    ingestion_chunk_max_chars: int = Field(default=1200)
    ingestion_chunk_overlap_chars: int = Field(default=160)
    ingestion_chunk_max_tokens: int = Field(default=480)
    ingestion_chunk_overlap_tokens: int = Field(default=96)
    ingestion_chunk_min_tokens: int = Field(default=48)
    retrieval_bm25_weight: float = Field(default=0.45)
    retrieval_vector_weight: float = Field(default=0.55)
    retrieval_bm25_top_k: int = Field(default=20)
    retrieval_ann_top_k: int = Field(default=20)
    retrieval_fused_top_k: int = Field(default=24)
    retrieval_rrf_k: int = Field(default=60)

    long_term_memory_top_k: int = Field(default=3)
    max_compression_retries: int = Field(default=1)

    # LLM 上下文窗口（tokens）。默认按 200k 估算，可通过 .env 覆盖。
    llm_context_window_tokens: int = Field(default=200_000)
    # 每轮发给 LLM 的 prompt 允许占用的输入上限（tokens）。压缩触发阈值。
    llm_input_budget_tokens: int = Field(default=160_000)
    # 聊天历史里"最近原文"保留的条数；其余走压缩摘要。
    history_tail_messages: int = Field(default=6)
    # tiktoken 估算使用的编码名称（未安装 tiktoken 时回退到字符数估算）。
    tiktoken_encoding_name: str = Field(default="cl100k_base")

    prometheus_base_url: Optional[str] = None
    prometheus_auth_token: Optional[str] = None

    remote_ssh_default_host: str = Field(default="sunyuyang.121@10.37.194.221")
    remote_ssh_port: int = Field(default=22)
    remote_ssh_connect_timeout_s: int = Field(default=10)
    remote_ssh_command_timeout_s: int = Field(default=30)
    remote_ssh_allowed_hosts: str = Field(
        default="sunyuyang.121@10.37.194.221,10.37.194.221"
    )

    route_llm_router_enabled: bool = Field(default=True)
    router_native_tool_calling_enabled: bool = Field(default=False)
    route_ops_keywords: str = Field(
        default="cpu,memory,mem,disk,load,latency,error,alert,slow,timeout,service,restart,rollback,日志,故障,排障,异常,告警,温度,风扇,gpu,终端,命令,命令行,shell,bash,执行,运维,排查,诊断,进程,端口,磁盘,内存,本机,这台,我的电脑,这台电脑"
    )
    route_rag_keywords: str = Field(
        default="kb,knowledge,document,docs,manual,playbook,faq,search,检索,知识库,文档,手册"
    )

    max_ops_workers: int = 3
    compression_trigger_ratio: float = 0.5
    langgraph_checkpoint_path: str = Field(default=".data/langgraph/checkpoints.pkl")
    langgraph_checkpoint_namespace: str = Field(default="")
    langgraph_graph_name: str = Field(default="ops-rag-agent-supervisor")

    langsmith_tracing_enabled: bool = Field(default=False)
    langsmith_project: str = Field(default="ops-rag-agent")
    langsmith_api_key: Optional[str] = None
    langsmith_endpoint: Optional[str] = None

    # rag / web 显式列入白名单，防止以后改回原 business_domain 时被过滤掉
    skill_allowed_business_domains: str = Field(
        default="general,ops,knowledge,platform,rag,web"
    )
    guardrails_enabled: bool = Field(default=True)
    guardrails_model_review_enabled: bool = Field(default=True)

    @property
    def allowed_skill_business_domains(self) -> tuple[str, ...]:
        return tuple(
            item.strip()
            for item in self.skill_allowed_business_domains.split(",")
            if item.strip()
        )

    @property
    def allowed_remote_ssh_hosts(self) -> tuple[str, ...]:
        return tuple(
            item.strip()
            for item in self.remote_ssh_allowed_hosts.split(",")
            if item.strip()
        )

    @property
    def route_ops_keyword_list(self) -> tuple[str, ...]:
        return tuple(item.strip() for item in self.route_ops_keywords.split(",") if item.strip())

    @property
    def route_rag_keyword_list(self) -> tuple[str, ...]:
        return tuple(item.strip() for item in self.route_rag_keywords.split(",") if item.strip())


settings = Settings()
