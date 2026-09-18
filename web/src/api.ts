import { describeFailure, OFFLINE_HINT } from './lib/errors'

export type ProductSettings = {
  provider: 'ollama' | 'deepseek'
  ollama_base_url: string
  ollama_model: string
  onboarding_complete: boolean
  display_name: string
  retrieval_mode: 'bm25' | 'hybrid'
  deepseek_model: string
  theme: 'system' | 'light' | 'dark'
  /** 记忆接缝的开关。默认关；关着时后端**根本不会调用**记忆提供者。 */
  memory_enabled: boolean
  /** 联网开关。默认关；关着时后端一个字节都不发。 */
  web_enabled: boolean
  /**
   * "用户已经被告知什么会离开这台机器"的凭据。与开关分开：后端**拒绝**在没有这个
   * 凭据时打开联网，所以界面必须先把告知展示出来（见 App.tsx 的设置页）。
   */
  web_disclosure_acknowledged: boolean
}

export type VectorIndexState = {
  status: 'ready' | 'building' | 'off'
  version_id: string | null
  // 索引里超出检索模型长度上限的片段数；只要索引没就绪就不上报。
  truncated_chunks?: number
}
export type SetupState = {
  settings: ProductSettings
  deepseek_key_configured: boolean
  data_root: string
  steps: Record<string, boolean>
  retrieval_model: RetrievalModelState
  materials: { total_documents: number; ready_documents: number; chunk_count: number; index_version: string | null; vector_index: VectorIndexState }
  recovery_required?: boolean
  migration_error?: string
  recovery_backups?: { name: string; size: number; modified_at: string }[]
}

export type MediaType = 'markdown' | 'pdf' | 'notebook'
/**
 * 一条依据来自哪一层。`note` 是检索到的笔记片段，也是唯一出厂就存在的层；
 * `memory` 与 `web` 分别由记忆提供者和联网补充追加，默认都关着。
 */
export type Origin = 'note' | 'memory' | 'web'
/** 来源列表与证据里出现的媒体类型：三层来源共用一个字段，所以比文档媒体类型宽。 */
export type SourceMediaType = MediaType | 'memory' | 'web'
export type Locator =
  | { kind: 'markdown'; start_line: number; end_line: number }
  | { kind: 'pdf'; page: number }
  | { kind: 'notebook'; cell: number; cell_type: string; start_line: number; end_line: number }
  /** 记忆条目没有行号可定位，只能回到它派生自哪次对话/哪份笔记。 */
  | { kind: 'memory'; id: string; derived_from: string }
  /**
   * 网络来源定位到 URL。`published_at` 是这条事实何时成立、`retrieved_at` 是何时
   * 看到的——两者可能差很远，时效判断要的正是这个差。
   */
  | { kind: 'web'; url: string; published_at: string | null; retrieved_at: string | null }

export type Library = { id: string; name: string; kind: 'folder' | 'uploads'; root_path?: string; document_count: number; ready_count: number }
/** 应用内文件夹选择器看到的一层目录。`parent` 为空说明已经是盘符/根，不能再往上。 */
export type FolderListing = {
  path: string
  name: string
  parent: string | null
  entries: { name: string; path: string }[]
  /** 子目录超过后端上限时为 true，界面要说明"没有列全"，不能假装这就是全部。 */
  truncated: boolean
  roots: { name: string; path: string }[]
}
export type DocumentItem = { id: string; library_id: string; relative_path: string; display_name: string; media_type: MediaType; status: string; error: string | null; current_version_id: string | null; updated_at: string; library_name: string }
export type ImportJob = { id: string; job_type: string; status: string; total: number; completed: number; failed: number; message: string | null }
export type SearchHit = { origin?: Origin; chunk_id: string; document_id: string; version_id: string; title: string; media_type: SourceMediaType; heading_path: string; locator: Locator; preview: string; score: number; matched_tokens: string[]; channels?: Record<string, number>; channel_scores?: Record<string, number>; quality_reason?: string | null }
export type SearchResult = { query: string; index_version: string | null; retrieval_mode: 'bm25' | 'hybrid'; effective_mode: 'bm25' | 'hybrid'; results: SearchHit[] }
export type SourceContent = { document_id: string; version_id: string; title: string; media_type: MediaType; text: string }
/**
 * 来源既可能来自本轮搜索（分数齐全），也可能来自引入分数之前存下的记录。
 * 后者没有 `score`，界面据此隐藏相关度条，而不是拿 0 冒充。
 */
export type SourceRef = Omit<SearchHit, 'score' | 'matched_tokens'> & { score?: number | null; matched_tokens?: string[] }
export type MessageSource = SourceRef & { label: string }
/**
 * `web_state` 是这一轮联网的**持久记录**，`null` 表示没有记录（旧版本写的回答，
 * 或这条回答没过联网这一层，例如被守卫拦下）。它不从事件里推，也不在界面重新
 * 计算——服务端当时怎么记的，翻出来就是什么。
 */
export type ChatMessage = { id: string; conversation_id: string; role: 'user' | 'assistant'; content: string; status: 'complete' | 'streaming' | 'stopped' | 'failed'; provider: string | null; model: string | null; index_version: string | null; error_code: string | null; reply_to_message_id: string | null; sources: MessageSource[]; web_state: WebState | null }
export type Conversation = { id: string; title: string; created_at: string; updated_at: string; message_count?: number; messages?: ChatMessage[] }
export type FavoriteSummary = { id: string; message_id: string; title: string; note: string; question: string; answer: string; provider: string | null; model: string | null; index_version: string | null; generated_at: string | null; updated_at: string; source_count: number; tags: string[]; libraries: { id: string; name: string }[]; feedback_kind: FeedbackKind | null }
export type Favorite = FavoriteSummary & { sources: MessageSource[] }
/** 筛选器选项取自全部收藏（不是筛完的那一批），否则选中之后切不回去。 */
export type FavoriteFilters = { library?: string; tag?: string; feedback?: FeedbackKind; days?: '7d' | '30d' }
export type FavoritesView = { favorites: FavoriteSummary[]; libraries: { id: string; name: string }[]; tags: string[]; total: number }
export type FeedbackKind = 'helpful' | 'missing' | 'citation_wrong' | 'answer_wrong'
export type StreamEvent = { type: 'retrieval' | 'generation' | 'token' | 'final' | 'stopped' | 'error'; message_id: string; text?: string; content?: string; status?: ChatMessage['status']; sources?: MessageSource[]; web?: WebState; provider?: string; model?: string; message?: string; citation_warning?: boolean; rejected?: boolean }

export type BundledResource = { name: string; size: number; modified_at: string }
export type ResourcesIndex = { docs: BundledResource[]; examples: BundledResource[] }

// ---------------------------------------------------------------------------
// Studio 产出
// ---------------------------------------------------------------------------

export type ArtifactKind = 'guide' | 'mindmap'
export type ArtifactStatus = 'running' | 'complete' | 'stopped' | 'failed'
/**
 * 回链报告：**产出功能唯一的诚实分数**。
 *
 * `hit_rate` 在"一句断言都没有"时是 `null` 而不是 1——空产出没有"全部带来源"
 * 这回事，显示成满分就是数字说谎。引用了不存在编号的行同时计入 `missing_count`
 * 与 `invalid_labels`：它比缺来源更隐蔽，因为它看起来像有来源。
 */
export type BacklinkReport = {
  assertions: number
  with_source: number
  hit_rate: number | null
  cited_labels: string[]
  invalid_labels: string[]
  missing_count: number
  missing: { line: number; text: string }[]
}
/** 思维导图节点。`sources` 是片段编号：每个节点都能点回原文。 */
export type MindmapNode = {
  id: string
  label: string
  level: number
  sources: string[]
  children: MindmapNode[]
}
/**
 * 思维导图。**刻意没有 `hit_rate`**：节点是拿片段自身的标题层级建的，覆盖率必然
 * 接近 1，把它和指南的命中率并排显示会让人以为两者可比。
 */
export type Mindmap = {
  topic: string
  tree: MindmapNode
  node_count: number
  linked_chunks: number
  coverage_note: string
  /** 仅供复制到 Obsidian 用；产品自身不渲染 Mermaid（前端没有渲染器）。 */
  mermaid: string
}
export type ArtifactSummary = {
  id: string
  kind: ArtifactKind
  title: string
  topic: string
  status: ArtifactStatus
  index_version: string | null
  error_code: string | null
  created_at: string
  completed_at: string | null
  content_length: number
}
/** `backlink` 只在指南上有，`mindmap` 只在导图上有——两者不会同时出现。 */
export type Artifact = ArtifactSummary & {
  content: string
  sources: MessageSource[]
  backlink?: BacklinkReport
  mindmap?: Mindmap | null
}
/**
 * 信息图的渲染记录（P4）。
 *
 * `status` 只有两种：
 * - `complete`：本机浏览器渲染成功，`pixels` / `milliseconds` 都是实测值；
 * - `unavailable`：本机没有可用浏览器。**这不是失败**——同一目录下那份 HTML 照样
 *   导出了，用户可以用自己的浏览器打开它。界面按"降级"显示，不按"出错"显示。
 */
export type InfographicRender = {
  status: 'complete' | 'unavailable'
  reason: string | null
  message: string | null
  browser: string | null
  browser_path: string | null
  milliseconds: number | null
  bytes: number | null
  pixels: { width: number; height: number } | null
  /** 浏览器有没有按请求的尺寸出图。`false` 说明图能看但尺寸不对，得让用户看见。 */
  pixels_match: boolean | null
  created_at: string
}
/**
 * 信息图导出记录。**它不是一种产出类型**，而是把一份产出的来源画成 PNG：
 * 不调模型、不需要新表，只把已经存在的来源分布、章节结构、命中率摊到一张图上。
 * 所以图里没有一个字来自模型——每个编号都指向一份资料片段。
 */
export type InfographicExport = {
  artifact_id: string
  title: string
  kind: ArtifactKind
  kind_label: string
  layout: { width: number; height: number; scale: number; pixels: { width: number; height: number } }
  stats: { chunks: number; documents: number; sections: number; origin_text: string }
  /** 只有指南有：图里印的就是这个真分数。导图不给（它的覆盖率是构造结果）。 */
  backlink: BacklinkReport | null
  files: { png: string; html: string; record: string; png_path: string; html_path: string }
  render: InfographicRender
  degraded: boolean
}
export type ArtifactStreamEvent = {
  type: 'retrieval' | 'token' | 'final' | 'stopped' | 'error'
  artifact_id: string
  text?: string
  content?: string
  status?: ArtifactStatus
  sources?: MessageSource[]
  backlink?: BacklinkReport
  mindmap?: Mindmap
  error?: string
  message?: string
  query?: string
  index_version?: string | null
  provider?: string
  model?: string
}
/**
 * 记忆接缝的运行状态。`items` 单独可为 null：条目数要问提供者才拿得到，
 * 第三方实现抛异常时后端只把类型名放进 `error`，不让整个诊断接口失败。
 */
export type MemoryDiagnostics = {
  available: boolean
  provider?: string
  items?: number | null
  error?: string
  enabled: boolean
}
/**
 * 联网这一层的结果状态。每一轮问答都有一个确定值——不给"说不清到底联没联上"留
 * 中间态，否则用户没法判断眼前的回答里有没有外部信息。
 *
 * - `off`：开关关着（或没开过告知）；
 * - `unconfigured`：开关开着但没配任何后端，**一个请求都没发**；
 * - `ok` / `empty`：问到了，有结果 / 没结果；
 * - `failed`：后端报错，`detail` 是异常类型名，界面要显示"本次未能联网"。
 */
export type WebStatus = 'off' | 'unconfigured' | 'ok' | 'empty' | 'failed'
export type WebState = { status: WebStatus; provider: string; detail?: string }
export type WebDiagnostics = {
  available: boolean
  provider?: string
  configured?: boolean
  enabled?: boolean
  disclosure_acknowledged?: boolean
}
/** 产出的运行状态：按状态数个数，好区分"还没生成完"和"生成失败了"。 */
export type StudioDiagnostics = { available: boolean; by_status?: Record<string, number> }
export type Diagnostics = {
  product_version: string
  python: string
  frozen: boolean
  database_schema: number
  data_directories: { root_exists: boolean; database_exists: boolean; model_cache_exists: boolean }
  credentials: { deepseek_configured: boolean }
  memory: MemoryDiagnostics
  web: WebDiagnostics
  studio: StudioDiagnostics
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
  let response: Response
  try {
    response = await fetch(path, {
      ...init,
      headers: init?.body && !isForm ? { 'Content-Type': 'application/json', ...init.headers } : init?.headers,
    })
  } catch (reason) {
    if (reason instanceof DOMException && reason.name === 'AbortError') throw reason
    throw new Error(OFFLINE_HINT)
  }
  if (!response.ok) {
    throw new Error(describeFailure(response.status, await response.json().catch(() => null)))
  }
  return response.json() as Promise<T>
}

/**
 * 逐行读 NDJSON 流。
 *
 * 抽出来是因为问答与产出用的是同一种流格式。两处各写一份解析的话，将来改缓冲或
 * 分帧只会改到一处，症状是其中一边偶发地丢事件、或者拿半个 JSON 去 parse。
 * 收尾那段（`buffer.trim()`）不是多余的：服务端最后一行可能没有换行符。
 */
async function readNdjson<T>(response: Response, onEvent: (event: T) => void): Promise<void> {
  if (!response.ok || !response.body) {
    throw new Error(describeFailure(response.status, await response.json().catch(() => null)))
  }
  const reader = response.body.getReader(); const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const lines = buffer.split('\n'); buffer = lines.pop() || ''
    for (const line of lines) if (line.trim()) onEvent(JSON.parse(line) as T)
    if (done) break
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer) as T)
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
  /** 应用内选择文件夹：只列子目录，不弹系统对话框。
   *
   * 安装版是无窗口进程，系统文件夹对话框会被创建在浏览器窗口后面（实测在 z 序上
   * 紧排在前台窗口之后），用户看不到就等于按钮坏了。 */
  folders: (path?: string) => request<FolderListing>(
    `/api/v1/system/folders${path ? `?path=${encodeURIComponent(path)}` : ''}`),
  connectFolder: (path: string) => request<{ library_id: string; job_id: string }>(
    '/api/v1/libraries/folders', { method: 'POST', body: JSON.stringify({ path }) }),
  refreshLibrary: (id: string) => request<{ job_id: string }>(
    `/api/v1/libraries/${id}/refresh`, { method: 'POST' }),
  upload: (file: File) => { const body = new FormData(); body.append('file', file); return request<{ document_id: string; job_id: string }>(
    '/api/v1/documents/upload', { method: 'POST', body }) },
  removeDocument: (id: string) => request<{ removed: boolean }>(`/api/v1/documents/${id}`, { method: 'DELETE' }),
  retryDocument: (id: string) => request<{ job_id: string }>(`/api/v1/documents/${id}/retry`, { method: 'POST' }),
  search: (query: string) => request<SearchResult>(
    `/api/v1/search?q=${encodeURIComponent(query)}`),
  source: (documentId: string, versionId: string) => request<SourceContent>(
    `/api/v1/documents/${documentId}/versions/${versionId}/source`),
  sourceFileUrl: (documentId: string, versionId: string) =>
    `/api/v1/documents/${documentId}/versions/${versionId}/file`,
  conversations: () => request<{ conversations: Conversation[] }>('/api/v1/conversations'),
  createConversation: (title = '新会话') => request<Conversation>(
    '/api/v1/conversations', { method: 'POST', body: JSON.stringify({ title }) }),
  conversation: (id: string) => request<Conversation>(`/api/v1/conversations/${id}`),
  /** 删除会话会连带删掉它的消息；收藏是独立保存的结论，不会被一起删掉。 */
  deleteConversation: (id: string) => request<{ deleted: boolean; title: string; messages: number; kept_favorites: number }>(
    `/api/v1/conversations/${id}`, { method: 'DELETE' }),
  streamMessage: async (
    conversationId: string,
    body: { question?: string; retry_message_id?: string; skip_guard?: boolean },
    onEvent: (event: StreamEvent) => void,
    signal: AbortSignal,
  ) => {
    let response: Response
    try {
      response = await fetch(`/api/v1/conversations/${conversationId}/messages/stream`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body), signal,
      })
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === 'AbortError') throw reason
      throw new Error(OFFLINE_HINT)
    }
    await readNdjson<StreamEvent>(response, onEvent)
  },
  artifacts: () => request<{ artifacts: ArtifactSummary[] }>('/api/v1/artifacts'),
  artifact: (id: string) => request<Artifact>(`/api/v1/artifacts/${id}`),
  deleteArtifact: (id: string) => request<{ deleted: boolean }>(
    `/api/v1/artifacts/${id}`, { method: 'DELETE' }),
  /** 停止生成。与问答一样，停止后已写出的部分会保留为 `stopped`，不丢内容。 */
  stopArtifact: (id: string) => request<{ stopping: boolean }>(
    `/api/v1/artifacts/${id}/stop`, { method: 'POST' }),
  streamArtifact: async (
    topic: string,
    kind: ArtifactKind,
    onEvent: (event: ArtifactStreamEvent) => void,
    signal: AbortSignal,
  ) => {
    let response: Response
    try {
      response = await fetch('/api/v1/artifacts/stream', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, kind }), signal,
      })
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === 'AbortError') throw reason
      throw new Error(OFFLINE_HINT)
    }
    await readNdjson<ArtifactStreamEvent>(response, onEvent)
  },
  /**
   * 信息图的三条路径集中放在这里：界面只该知道"有这么一个接口"，不该自己拼 URL。
   * `stamp` 传上一次的渲染时间——重新导出后不加它，浏览器会把旧图继续拿来显示。
   */
  infographicImage: (id: string, stamp?: string) =>
    `/api/v1/artifacts/${id}/infographic.png${stamp ? `?v=${encodeURIComponent(stamp)}` : ''}`,
  infographicDownload: (id: string) => `/api/v1/artifacts/${id}/infographic.png?download=1`,
  /** 导出（渲染）信息图。浏览器不可用时也返回 200，看 `degraded` / `render.status`。 */
  exportInfographic: (id: string) => request<InfographicExport>(
    `/api/v1/artifacts/${id}/infographic`, { method: 'POST' }),
  /** 读回上一次的导出记录；从没导出过就是 `export: null`（不是错误）。 */
  infographicExport: (id: string) => request<{ artifact_id: string; export: InfographicExport | null }>(
    `/api/v1/artifacts/${id}/infographic`),
  stopMessage: (conversationId: string, messageId: string) => request<{ stopping: boolean }>(
    `/api/v1/conversations/${conversationId}/messages/${messageId}/stop`, { method: 'POST' }),
  favorites: (filters: FavoriteFilters = {}) => {
    const query = new URLSearchParams()
    for (const [key, value] of Object.entries(filters)) if (value) query.set(key, value)
    const suffix = query.size ? `?${query}` : ''
    return request<FavoritesView>(`/api/v1/favorites${suffix}`)
  },
  favorite: (id: string) => request<Favorite>(`/api/v1/favorites/${id}`),
  createFavorite: (messageId: string) => request<Favorite>(
    '/api/v1/favorites', { method: 'POST', body: JSON.stringify({ message_id: messageId }) }),
  updateFavorite: (id: string, values: { title?: string; note?: string; tags?: string[] }) => request<Favorite>(
    `/api/v1/favorites/${id}`, { method: 'PATCH', body: JSON.stringify(values) }),
  deleteFavorite: (id: string) => request<{ deleted: boolean }>(
    `/api/v1/favorites/${id}`, { method: 'DELETE' }),
  favoriteExportUrl: (id: string) => `/api/v1/favorites/${id}/export`,
  /** 收哪些由前端点名，所以"导出的就是眼前这些"。
   *
   * 路径写在 `/{favorite_id}/export` 之前才不会被它当成一条收藏的 id。 */
  collectionExportUrl: (ids: string[], title: string) => {
    const query = new URLSearchParams({ ids: ids.join(',') })
    if (title) query.set('title', title)
    return `/api/v1/favorites/collection/export?${query}`
  },
  feedback: (messageId: string, kind: FeedbackKind, note = '') => request<{ stored_locally: boolean }>(
    '/api/v1/feedback', { method: 'POST', body: JSON.stringify({ message_id: messageId, kind, note }) }),
  backup: async () => {
    const response = await fetch('/api/v1/system/backup', { method: 'POST' })
    if (!response.ok) throw new Error(describeFailure(response.status, await response.json().catch(() => null)))
    return { blob: await response.blob(), disposition: response.headers.get('content-disposition') }
  },
  restore: (file: File) => { const body = new FormData(); body.append('file', file); return request<{ restored: boolean; restart_required: boolean }>(
    '/api/v1/system/restore', { method: 'POST', body }) },
  diagnosticExportUrl: () => '/api/v1/system/diagnostics/export',
  shutdown: () => request<{ status: string; message: string }>(
    '/api/v1/system/shutdown', { method: 'POST' }),
  restoreMigration: (backupName: string) => request<{ restored: boolean; restart_required: boolean }>(
    '/api/v1/system/recovery/restore', { method: 'POST', body: JSON.stringify({ backup_name: backupName }) }),
}
