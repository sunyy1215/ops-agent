# Debug Session: dashboard-500-error [OPEN]

## Problem
- Dashboard chat still shows `500 Internal Server Error` after refresh.
- Backend logs observed `POST /invoke` returning `200 OK`, so symptom may be in frontend routing/proxy/runtime handling.

## Hypotheses
- H1: Browser is not using the current Vite dev server and is loading an old static bundle.
- H2: Frontend proxy/API path resolution fails in browser runtime and surfaces as a generic 500.
- H3: Backend returns 200, but frontend response parsing or state update throws and gets shown as a 500-like error.
- H4: Chat submission flow sends an unexpected payload or duplicate request that causes inconsistent UI state.

## Evidence Plan
- Add frontend instrumentation around request URL, response status, and thrown errors.
- Add chat submit instrumentation for payload/thread id and keydown submit path.
- Reproduce in browser and inspect emitted runtime evidence before changing logic.

## Status
- Session opened.
- No business logic changes yet.
