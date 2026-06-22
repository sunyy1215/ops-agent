# Debug Session: service-unavailable

- Status: OPEN
- Started At: 2026-04-28
- Symptom: 用户访问时提示“服务不可用”
- Scope: 本地前端开发服务器、后端 FastAPI、前端到后端代理链路

## Hypotheses
- H1: 前端开发服务器已经退出，导致浏览器无法访问 `http://127.0.0.1:3000`
- H2: 后端 `uvicorn` 已退出或启动失败，导致前端代理 `/api` 返回不可用
- H3: 前端仍在运行，但请求 `/api/invoke` 时代理到后端失败
- H4: 最近代码改动引入了运行时异常，只在请求时触发而不是启动时触发
- H5: 端口仍被旧进程占用或进程状态异常，导致看似启动成功但实际不可服务

## Evidence Log
- 运行时检查：`http://127.0.0.1:8000/healthz` 返回 `200 OK`
- 运行时检查：`http://127.0.0.1:3000/` 连接被拒绝
- 端口检查：`127.0.0.1:8000 OPEN`
- 端口检查：`127.0.0.1:3000 ConnectionRefusedError`
- 后端运行日志显示近期仍在成功处理 `/healthz`、`/sessions`、`/metrics/summary`、`POST /invoke`

## Next Step
- 重启前端开发服务器，并验证首页与聊天页可访问
