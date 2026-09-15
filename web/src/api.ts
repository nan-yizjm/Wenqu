export type ProductSettings = {
  provider: 'ollama' | 'deepseek'
  ollama_base_url: string
  ollama_model: string
  onboarding_complete: boolean
  display_name: string
  retrieval_mode: 'bm25' | 'hybrid'
  deepseek_model: string
}

export type SetupState = {
  settings: ProductSettings
  deepseek_key_configured: boolean
  data_root: string
  steps: Record<string, boolean>
  retrieval_model: RetrievalModelState
  materials: { total_documents: number; ready_documents: number; chunk_count: number; index_version: string | null }
  recovery_required?: boolean
  migration_error?: string
  recovery_backups?: { name: string; size: number; modified_at: string }[]
}

export type Library = { id: string; name: string; kind: 'folder' | 'uploads'; root_path?: string; document_count: number; ready_count: number }
export type DocumentItem = { id: string; library_id: string; relative_path: string; display_name: string; media_type: 'markdown' | 'pdf'; status: string; error: string | null; current_version_id: string | null; updated_at: string; library_name: string }
export type ImportJob = { id: string; job_type: string; status: string; total: number; completed: number; failed: number; message: string | null }
export type SearchHit = { chunk_id: string; document_id: string; version_id: string; title: string; media_type: 'markdown' | 'pdf'; heading_path: string; locator: { kind: 'markdown'; start_line: number; end_line: number } | { kind: 'pdf'; page: number }; preview: string; score: number; matched_tokens: string[] }
export type SourceContent = { document_id: string; version_id: string; title: string; media_type: 'markdown' | 'pdf'; text: string }
export type MessageSource = Omit<SearchHit, 'score' | 'matched_tokens'> & { label: string; position?: number }
export type ChatMessage = { id: string; conversation_id: string; role: 'user' | 'assistant'; content: string; status: 'complete' | 'streaming' | 'stopped' | 'failed'; provider: string | null; model: string | null; index_version: string | null; error_code: string | null; reply_to_message_id: string | null; sources: MessageSource[] }
export type Conversation = { id: string; title: string; created_at: string; updated_at: string; message_count?: number; messages?: ChatMessage[] }
export type FavoriteSummary = { id: string; message_id: string; title: string; note: string; question: string; answer: string; provider: string | null; model: string | null; index_version: string | null; generated_at: string | null; updated_at: string; source_count: number }
export type Favorite = FavoriteSummary & { sources: MessageSource[] }
export type FeedbackKind = 'helpful' | 'missing' | 'citation_wrong' | 'answer_wrong'
export type StreamEvent = { type: 'retrieval' | 'generation' | 'token' | 'final' | 'stopped' | 'error'; message_id: string; text?: string; content?: string; status?: ChatMessage['status']; sources?: MessageSource[]; provider?: string; model?: string; message?: string; citation_warning?: boolean; rejected?: boolean }

export type BundledResource = { name: string; size: number; modified_at: string }
export type ResourcesIndex = { docs: BundledResource[]; examples: BundledResource[] }
export type Diagnostics = {
  product_version: string
  python: string
  frozen: boolean
  database_schema: number
  data_directories: { root_exists: boolean; database_exists: boolean; model_cache_exists: boolean }
  credentials: { deepseek_configured: boolean }
  runtime: { status: string; detail: string | null }
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
  const isForm = init?.body instanceof FormData
  const response = await fetch(path, {
    ...init,
    headers: init?.body && !isForm ? { 'Content-Type': 'application/json', ...init.headers } : init?.headers,
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
  diagnostics: () => request<Diagnostics>('/api/v1/system/diagnostics'),
  resources: () => request<ResourcesIndex>('/api/v1/resources'),
  importBundledExample: (name: string) => request<{ document_id: string; job_id: string | null; already_imported: boolean }>(
    '/api/v1/resources/import', { method: 'POST', body: JSON.stringify({ name }) }),
  bundledDocUrl: (name: string) => `/api/v1/resources/docs/${encodeURIComponent(name)}`,
  prepareRetrievalModel: () => request<RetrievalModelState>(
    '/api/v1/setup/retrieval-model', { method: 'POST' }),
  retrievalModelStatus: () => request<RetrievalModelState>('/api/v1/setup/retrieval-model'),
  libraries: () => request<{ libraries: Library[] }>('/api/v1/libraries'),
  documents: () => request<{ documents: DocumentItem[] }>('/api/v1/documents'),
  jobs: () => request<{ jobs: ImportJob[] }>('/api/v1/import-jobs'),
  pickFolder: () => request<{ path: string | null }>('/api/v1/system/pick-folder', { method: 'POST' }),
  connectFolder: (path: string) => request<{ library_id: string; job_id: string }>(
    '/api/v1/libraries/folders', { method: 'POST', body: JSON.stringify({ path }) }),
  refreshLibrary: (id: string) => request<{ job_id: string }>(
    `/api/v1/libraries/${id}/refresh`, { method: 'POST' }),
  upload: (file: File) => { const body = new FormData(); body.append('file', file); return request<{ document_id: string; job_id: string }>(
    '/api/v1/documents/upload', { method: 'POST', body }) },
  removeDocument: (id: string) => request<{ removed: boolean }>(`/api/v1/documents/${id}`, { method: 'DELETE' }),
  retryDocument: (id: string) => request<{ job_id: string }>(`/api/v1/documents/${id}/retry`, { method: 'POST' }),
  search: (query: string) => request<{ query: string; index_version: string | null; results: SearchHit[] }>(
    `/api/v1/search?q=${encodeURIComponent(query)}`),
  source: (documentId: string, versionId: string) => request<SourceContent>(
    `/api/v1/documents/${documentId}/versions/${versionId}/source`),
  sourceFileUrl: (documentId: string, versionId: string) =>
    `/api/v1/documents/${documentId}/versions/${versionId}/file`,
  conversations: () => request<{ conversations: Conversation[] }>('/api/v1/conversations'),
  createConversation: (title = '新会话') => request<Conversation>(
    '/api/v1/conversations', { method: 'POST', body: JSON.stringify({ title }) }),
  conversation: (id: string) => request<Conversation>(`/api/v1/conversations/${id}`),
  streamMessage: async (
    conversationId: string,
    body: { question?: string; retry_message_id?: string; skip_guard?: boolean },
    onEvent: (event: StreamEvent) => void,
    signal: AbortSignal,
  ) => {
    const response = await fetch(`/api/v1/conversations/${conversationId}/messages/stream`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body), signal,
    })
    if (!response.ok || !response.body) {
      const detail = await response.json().catch(() => ({}))
      throw new Error(detail.message || `请求失败：${response.status}`)
    }
    const reader = response.body.getReader(); const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      const lines = buffer.split('\n'); buffer = lines.pop() || ''
      for (const line of lines) if (line.trim()) onEvent(JSON.parse(line) as StreamEvent)
      if (done) break
    }
    if (buffer.trim()) onEvent(JSON.parse(buffer) as StreamEvent)
  },
  stopMessage: (conversationId: string, messageId: string) => request<{ stopping: boolean }>(
    `/api/v1/conversations/${conversationId}/messages/${messageId}/stop`, { method: 'POST' }),
  favorites: () => request<{ favorites: FavoriteSummary[] }>('/api/v1/favorites'),
  favorite: (id: string) => request<Favorite>(`/api/v1/favorites/${id}`),
  createFavorite: (messageId: string) => request<Favorite>(
    '/api/v1/favorites', { method: 'POST', body: JSON.stringify({ message_id: messageId }) }),
  updateFavorite: (id: string, values: { title?: string; note?: string }) => request<Favorite>(
    `/api/v1/favorites/${id}`, { method: 'PATCH', body: JSON.stringify(values) }),
  deleteFavorite: (id: string) => request<{ deleted: boolean }>(
    `/api/v1/favorites/${id}`, { method: 'DELETE' }),
  favoriteExportUrl: (id: string) => `/api/v1/favorites/${id}/export`,
  feedback: (messageId: string, kind: FeedbackKind, note = '') => request<{ stored_locally: boolean }>(
    '/api/v1/feedback', { method: 'POST', body: JSON.stringify({ message_id: messageId, kind, note }) }),
  backup: async () => {
    const response = await fetch('/api/v1/system/backup', { method: 'POST' })
    if (!response.ok) throw new Error('创建备份失败')
    return { blob: await response.blob(), disposition: response.headers.get('content-disposition') }
  },
  restore: (file: File) => { const body = new FormData(); body.append('file', file); return request<{ restored: boolean; restart_required: boolean }>(
    '/api/v1/system/restore', { method: 'POST', body }) },
  diagnosticExportUrl: () => '/api/v1/system/diagnostics/export',
  shutdown: () => request<{ status: string }>('/api/v1/system/shutdown', { method: 'POST' }),
  restoreMigration: (backupName: string) => request<{ restored: boolean; restart_required: boolean }>(
    '/api/v1/system/recovery/restore', { method: 'POST', body: JSON.stringify({ backup_name: backupName }) }),
}
