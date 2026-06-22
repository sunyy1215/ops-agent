import type {
  HealthzResponse,
  InvokeRequest,
  InvokeResponse,
  KbIngestRequest,
  KbIngestResponse,
  KbSearchRequest,
  KbSearchResponse,
  MemoryRecallRequest,
  MemoryRecallResponse,
  MemoryWriteRequest,
  MemoryWriteResponse,
  MetricsSummaryResponse,
  PublicConfigResponse,
  RunStateResponse,
  RuntimeConfigUpdateRequest,
  RuntimeConfigUpdateResponse,
  SessionsResponse,
  SkillsCatalogResponse,
} from '../types'

const API_PREFIX = import.meta.env.VITE_API_PREFIX ?? '/api'

class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_PREFIX}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    try {
      const data = (await response.json()) as { detail?: string | { msg?: string }[] }
      if (typeof data.detail === 'string') {
        message = data.detail
      } else if (Array.isArray(data.detail)) {
        message = data.detail.map((item) => item.msg).filter(Boolean).join('; ') || message
      }
    } catch {
      try {
        const text = await response.text()
        if (text.trim()) {
          message = text.trim().slice(0, 500)
        }
      } catch {
        // Ignore body read failures and fall back to status text.
      }
    }
    throw new ApiError(message, response.status)
  }

  return (await response.json()) as T
}

function postJson<TResponse, TRequest>(path: string, body: TRequest): Promise<TResponse> {
  return request<TResponse>(path, {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

function putJson<TResponse, TRequest>(path: string, body: TRequest): Promise<TResponse> {
  return request<TResponse>(path, {
    method: 'PUT',
    body: JSON.stringify(body),
  })
}

export const apiClient = {
  ApiError,
  invoke(payload: InvokeRequest) {
    return postJson<InvokeResponse, InvokeRequest>('/invoke', payload)
  },
  ingestKnowledge(payload: KbIngestRequest) {
    return postJson<KbIngestResponse, KbIngestRequest>('/kb/ingest', payload)
  },
  searchKnowledge(payload: KbSearchRequest) {
    return postJson<KbSearchResponse, KbSearchRequest>('/kb/search', payload)
  },
  writeMemory(payload: MemoryWriteRequest) {
    return postJson<MemoryWriteResponse, MemoryWriteRequest>('/memory/write', payload)
  },
  recallMemory(payload: MemoryRecallRequest) {
    return postJson<MemoryRecallResponse, MemoryRecallRequest>('/memory/recall', payload)
  },
  getPublicConfig() {
    return request<PublicConfigResponse>('/config/public')
  },
  updateRuntimeConfig(payload: RuntimeConfigUpdateRequest) {
    return putJson<RuntimeConfigUpdateResponse, RuntimeConfigUpdateRequest>(
      '/config/runtime',
      payload,
    )
  },
  getHealthz() {
    return request<HealthzResponse>('/healthz')
  },
  getSessions(limit = 20) {
    return request<SessionsResponse>(`/sessions?limit=${encodeURIComponent(String(limit))}`)
  },
  getRunState(threadId: string, checkpointId?: string) {
    const query = checkpointId
      ? `?checkpoint_id=${encodeURIComponent(checkpointId)}`
      : ''
    return request<RunStateResponse>(`/runs/${encodeURIComponent(threadId)}/state${query}`)
  },
  getSkillsCatalog() {
    return request<SkillsCatalogResponse>('/skills/catalog')
  },
  getMetricsSummary() {
    return request<MetricsSummaryResponse>('/metrics/summary')
  },
}

export { API_PREFIX }
