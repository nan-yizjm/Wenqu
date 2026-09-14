export type ProductSettings = {
  provider: 'ollama' | 'deepseek'
  ollama_base_url: string
  ollama_model: string
  onboarding_complete: boolean
  display_name: string
  retrieval_mode: 'bm25' | 'hybrid'
}

export type SetupState = {
  settings: ProductSettings
  deepseek_key_configured: boolean
  data_root: string
  steps: Record<string, boolean>
  retrieval_model: RetrievalModelState
}

export type RetrievalModelState = {
  status: 'not_downloaded' | 'downloading' | 'loading' | 'verifying' | 'ready' | 'failed'
  model: string
  detail: string | null
  dimension?: number
  device?: string
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: init?.body ? { 'Content-Type': 'application/json', ...init.headers } : init?.headers,
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.message || body.error || `请求失败：${response.status}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  setup: () => request<SetupState>('/api/v1/setup'),
  saveSettings: (values: Partial<ProductSettings>) => request<{ settings: ProductSettings }>(
    '/api/v1/settings', { method: 'PATCH', body: JSON.stringify(values) }),
  saveDeepSeek: (apiKey: string) => request<{ configured: boolean }>(
    '/api/v1/credentials/deepseek', { method: 'PUT', body: JSON.stringify({ api_key: apiKey }) }),
  diagnostics: () => request<Record<string, unknown>>('/api/v1/system/diagnostics'),
  prepareRetrievalModel: () => request<RetrievalModelState>(
    '/api/v1/setup/retrieval-model', { method: 'POST' }),
  retrievalModelStatus: () => request<RetrievalModelState>('/api/v1/setup/retrieval-model'),
}
