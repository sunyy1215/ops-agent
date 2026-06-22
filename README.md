# ops-rag-agent

`ops-rag-agent` 是一个面向对话式运维排障与知识检索的 Agent 系统。

项目用 `LangGraph` 编排多阶段工作流，用 `LangChain` 负责模型接入与 tool-calling 适配，并在中间补了一层自定义 `runtime`，把工具调用的参数校验、策略裁决、审批拦截、执行归一化和审计事件统一收敛起来。

当前仓库同时包含：

- Python 后端 API
- `LangGraph` Supervisor 工作流
- `RAG` 检索与知识库入库链路
- 运维 skill/runtime 执行框架
- React + Vite 调试 dashboard

## 核心能力

- 多路由 Agent：支持 `dialog`、`ops`、`rag` 三类主路径
- `LangGraph` 状态机编排：包含输入准备、意图分析、计划生成、skill router、审批恢复、记忆压缩和收尾
- Runtime 驱动工具调用：模型只提动作，真正执行前统一经过 `resolve -> validate -> policy -> execute -> normalize`
- RAG 检索链路：支持 query rewrite、hybrid retrieval、fusion、rerank
- 长短期记忆：支持对话压缩、长期记忆写入与召回
- 可观测性：暴露 `runtime_events`、`runtime_summary`、会话状态和技能目录
- 可选 dashboard：便于调试聊天、知识库、配置、会话和系统状态

## 架构概览

请求主链路：

1. `prepare_input`
2. `route`
3. `analyze_intent`
4. `plan`
5. `skill_router`
6. `approval_gate`
7. `memory_compressor`
8. `finalize`

分层上可以理解为：

- `LangGraph`：编排整个工作流和状态流转
- `LangChain`：封装模型、消息对象和原生 tool calling
- `runtime`：统一负责工具调用的校验、裁决、执行、归一化和审计
- `skills`：封装 RAG、web、Prometheus、本机/远程命令等底层能力

## 技术栈

- Python 3.9+
- FastAPI
- LangChain
- LangGraph
- Pydantic v2
- Milvus
- React 19 + Vite + Ant Design
- pytest / ruff / nox

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

也可以直接使用：

```bash
make dev
```

### 2. 配置环境变量

复制模板并按需填写：

```bash
cp .env.example .env
```

最少建议确认这些变量：

- `LLM_API_KEY`
- `LLM_API_BASE`
- `LLM_MODEL_CHAT`
- `LLM_MODEL_ROUTER`
- `LLM_MODEL_REASONING`
- `MILVUS_URI`
- `RERANK_BACKEND`
- `RERANK_MODEL`

如果只想先本地把服务跑起来，也可以先使用占位或兼容配置，后续再逐步接入真实的 embedding、rerank、Milvus 和外部运维能力。

### 3. 启动服务

```bash
PYTHONPATH=src python -m ops_rag_agent
```

默认监听：

```text
http://127.0.0.1:8000
```

### 4. 运行测试

```bash
make test
```

或：

```bash
make ci
```

## Dashboard

前端位于 [`dashboard/`](dashboard/)。

启动方式：

```bash
cd dashboard
npm install
npm run dev
```

常用脚本：

- `npm run dev`
- `npm run build`
- `npm run preview`

## 常用 API

后端入口在 [`src/ops_rag_agent/api/app.py`](src/ops_rag_agent/api/app.py)。

主要接口包括：

- `GET /healthz`：健康检查
- `GET /config/public`：查看公开配置
- `PUT /config/runtime`：热更新部分运行时配置
- `POST /invoke`：执行一次主工作流调用，支持 `resume`
- `POST /kb/ingest`：导入知识库文档
- `POST /kb/search`：直接调试检索链路
- `POST /memory/write`：写入长期记忆
- `POST /memory/recall`：召回长期记忆
- `GET /sessions`：查看会话列表
- `GET /sessions/{thread_id}`：查看会话和 checkpoint 历史
- `GET /runs/{thread_id}/state`：查看指定运行时状态
- `GET /skills/catalog`：查看技能目录
- `GET /metrics/summary`：查看聚合指标
- `GET /runtime/summary`：查看 runtime 级摘要

`/invoke` 最小示例：

```bash
curl -X POST http://127.0.0.1:8000/invoke \
  -H 'Content-Type: application/json' \
  -d '{
    "user_query": "帮我排查这台机器 CPU 飙高的原因"
  }'
```

`/kb/search` 最小示例：

```bash
curl -X POST http://127.0.0.1:8000/kb/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "CrashLoopBackOff 排查手册",
    "fused_top_k": 10,
    "rerank_top_k": 5
  }'
```

## RAG 流程

当前 RAG 主路径更偏向作为 `rag.search` skill 接入总工作流，而不是单独的一条固定 QA 链。

典型流程是：

1. 先做意图分析和计划生成
2. `plan` 阶段强制把 `rag.search` 放在第一步
3. `runtime` 在执行层再次收紧 `rag-first` 策略
4. `rag.search` 内部执行：
   - query rewrite
   - BM25 / vector hybrid retrieve
   - fusion
   - rerank
5. 检索结果作为证据继续喂给 `skill_router`，决定是否继续调用更多 skill 或直接收敛答案

## Runtime 设计

这个仓库的一个重点是把工具调用从 prompt 约束升级为 runtime 约束。

单次 skill 调用会统一走：

1. `resolve_skill`
2. `validate_arguments`
3. `apply_policy`
4. `execute`
5. `normalize_output`
6. `emit_runtime_events`

这样做的目的是：

- 参数错误尽早暴露
- 高风险技能统一走审批
- 避免重复 skill 死循环
- 统一不同 skill 的输出结构
- 为 API / dashboard / trace 提供一致的运行时数据

核心实现见 [`src/ops_rag_agent/skills/runtime.py`](src/ops_rag_agent/skills/runtime.py)。

## 仓库结构

```text
.
├── dashboard/                  # React + Vite 调试面板
├── evals/                      # 评测数据与检索实验脚本
├── scripts/                    # 入库、评测、Milvus 初始化等脚本
├── src/ops_rag_agent/
│   ├── agents/                 # Intent / Planner / Router / Agent 节点
│   ├── api/                    # FastAPI 服务
│   ├── graph/                  # LangGraph 工作流定义
│   ├── kb/                     # 知识库入库与检索服务
│   ├── memory/                 # 上下文压缩与长期记忆
│   ├── models/                 # LLM / embedding / provider 工厂
│   ├── rag/                    # 检索器与 reranker
│   ├── schemas/                # 状态与接口 schema
│   └── skills/                 # skill 定义、注册和 runtime
└── tests/                      # 单测与集成测试
```

## 开发说明

常用命令：

```bash
make compile
make lint
make test
make ci
python -m nox -s tests
```

知识库相关脚本示例：

```bash
python scripts/ingest_dir.py --help
python scripts/milvus_health.py
python scripts/milvus_smoke.py
```

## 当前状态

当前仓库已经具备：

- 可运行的 FastAPI 服务
- LangGraph Supervisor 工作流
- skill runtime 与审批恢复链路
- RAG 检索/入库接口
- dashboard 调试页面

一些检索与重排能力仍保留了可替换后端或 placeholder 接口，方便在不同环境下逐步接入真实服务。

## License

仓库当前未显式声明开源许可证。如需公开分发，建议补充 `LICENSE` 文件并明确使用条款。
