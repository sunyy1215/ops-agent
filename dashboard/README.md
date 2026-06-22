# Dashboard 联调说明

`dashboard/` 是独立的 `Vite + React + Ant Design` 前端工程，默认通过 `/api` 访问主服务，开发态由 `vite` 代理到本地 `FastAPI`。

## 目录说明

- `src/pages/ChatPage.tsx`：对接 `POST /invoke`
- `src/pages/KnowledgePage.tsx`：对接 `POST /kb/ingest`、`POST /kb/search`
- `src/pages/MemoryPage.tsx`：对接 `POST /memory/write`、`POST /memory/recall`
- `src/pages/ConfigPage.tsx`：对接 `GET /config/public`、`PUT /config/runtime`、`GET /healthz`
- `src/pages/SystemPage.tsx`：对接 `GET /healthz`、`GET /sessions`、`GET /runs/{thread_id}/state`、`GET /skills/catalog`、`GET /metrics/summary`

## 启动方式

1. 在仓库根目录启动后端服务，确保默认监听 `http://127.0.0.1:8000`
2. 进入前端目录并安装依赖

```bash
cd dashboard
npm install
```

3. 启动前端开发服务器

```bash
npm run dev
```

4. 打开终端输出中的本地地址，页面请求会自动按 `vite.config.ts` 转发：

```ts
server: {
  proxy: {
    '/api': {
      target: 'http://127.0.0.1:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ''),
    },
  },
}
```

## 联调要点

- 默认 API 前缀是 `/api`，可通过 `VITE_API_PREFIX` 覆盖
- 若后端不是 `127.0.0.1:8000`，请同步修改 `vite.config.ts` 代理目标
- `Config` 页的“测试连接”按钮会调用 `GET /healthz`，用于校验前端代理与后端可达性
- `System` 页会并行拉取健康状态、会话列表、技能目录和指标摘要，并在选中会话后继续请求对应 `run state`
- `PUT /config/runtime` 只提交后端返回的 `editable_fields`，敏感字段不会在页面明文展示或提交

## 推荐联调流程

1. 先打开 `System` 页确认 `GET /healthz`、`GET /sessions`、`GET /skills/catalog`、`GET /metrics/summary` 均返回成功
2. 在 `Sessions` 表中选择一个 `thread_id`，确认右侧 `Run State` 能正常返回
3. 打开 `Config` 页，确认公开配置可加载、白名单字段可编辑、保存反馈正常
4. 使用 `Config` 页的“测试连接”按钮复核代理配置是否正确

## 构建与校验

在 `dashboard/` 目录执行：

```bash
npm run lint
npm run build
```

如需接入生产环境，保持页面请求仍走同域 `/api` 即可，由网关或反向代理将其转发到实际 FastAPI 服务。
