import type {
  AgentResponse,
  AgentRunDetail,
  AgentRunListResponse,
  OptimizationMode,
} from '../types/agent'

const API_BASE = import.meta.env.VITE_API_URL || ''

export class ApiError extends Error {
  status: number
  detail?: string

  constructor(status: number, message: string, detail?: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail: string
    try {
      const errorJson = await response.json()
      errorDetail =
        typeof errorJson.detail === 'string'
          ? errorJson.detail
          : JSON.stringify(errorJson.detail || errorJson)
    } catch {
      errorDetail = response.statusText
    }
    throw new ApiError(response.status, `Request failed: ${errorDetail}`, errorDetail)
  }
  return response.json() as Promise<T>
}

export async function runAgent(
  prompt: string,
  mode: OptimizationMode = 'balanced',
  maxIterations: number = 3
): Promise<AgentResponse> {
  const response = await fetch(`${API_BASE}/api/v1/agent/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt: prompt.trim(),
      mode,
      max_iterations: maxIterations,
    }),
  })
  return handleResponse<AgentResponse>(response)
}

export async function listRuns(skip = 0, limit = 50): Promise<AgentRunListResponse> {
  const response = await fetch(`${API_BASE}/api/v1/runs?skip=${skip}&limit=${limit}`)
  return handleResponse<AgentRunListResponse>(response)
}

export async function getRun(runId: string): Promise<AgentRunDetail> {
  const response = await fetch(`${API_BASE}/api/v1/runs/${runId}`)
  return handleResponse<AgentRunDetail>(response)
}

export async function deleteRun(runId: string): Promise<{ success: boolean; deleted_id: string }> {
  const response = await fetch(`${API_BASE}/api/v1/runs/${runId}`, {
    method: 'DELETE',
  })
  return handleResponse<{ success: boolean; deleted_id: string }>(response)
}

export async function checkHealth(): Promise<{ status: string; service: string }> {
  const response = await fetch(`${API_BASE}/health`)
  return handleResponse<{ status: string; service: string }>(response)
}
