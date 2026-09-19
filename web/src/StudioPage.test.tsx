import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type {
  Artifact, ArtifactStreamEvent, ArtifactSummary, InfographicExport, Mindmap,
} from './api'
import { StudioPage } from './StudioPage'

type Event = ArtifactStreamEvent

const report = (over: Partial<Record<string, unknown>> = {}) => ({
  assertions: 3, with_source: 2, hit_rate: 2 / 3,
  cited_labels: ['S1'], invalid_labels: [], missing_count: 1,
  missing: [{ line: 4, text: '这句没有来源。' }], ...over,
})

const summary: ArtifactSummary = {
  id: 'art_1', kind: 'guide', title: 'KV Cache', topic: 'KV Cache',
  status: 'complete', index_version: 'idx_1', error_code: null,
  created_at: '2026-09-18T10:00:00', completed_at: '2026-09-18T10:00:05',
  content_length: 120,
}

const mindmap: Mindmap = {
  topic: 'KV Cache',
  tree: { id: 'root', label: 'KV Cache', level: 0, sources: [], children: [
    { id: 'n0', label: '推理.md', level: 1, sources: [], children: [
      { id: 'n1', label: 'PagedAttention', level: 2, sources: ['S1', 'S2'], children: [] },
    ] },
  ] },
  node_count: 3, linked_chunks: 2,
  coverage_note: '思维导图不经过模型：节点来自片段自身的标题层级，所以“每个节点都带编号”是构造结果而不是质量得分。',
  mermaid: 'mindmap\n  root((KV Cache))',
}

const source = {
  label: 'S1', chunk_id: 'c1', document_id: 'doc_a', version_id: 'ver_a',
  title: '推理.md', media_type: 'markdown' as const, heading_path: '推理 > PagedAttention',
  locator: { kind: 'markdown' as const, start_line: 1, end_line: 2 },
  preview: '分页管理 KV Cache。', score: 1,
}

/**
 * 信息图导出记录。`degraded()` 模拟"本机没装 / 找不到浏览器"——那种情况**不是失败**：
 * HTML 已经导出了，用户还能自己打开它。
 */
const exported = (over: Partial<InfographicExport> = {}): InfographicExport => ({
  artifact_id: 'art_1', title: 'KV Cache', kind: 'guide', kind_label: '学习指南',
  layout: { width: 1080, height: 948, scale: 2, pixels: { width: 2160, height: 1896 } },
  stats: { chunks: 3, documents: 2, sections: 2, origin_text: '笔记 3' },
  backlink: report(),
  files: {
    png: 'KV Cache-01234567.png', html: 'KV Cache-01234567.html',
    record: 'KV Cache-01234567.render.json',
    png_path: 'C:\\数据\\exports\\KV Cache-01234567.png',
    html_path: 'C:\\数据\\exports\\KV Cache-01234567.html',
  },
  render: {
    status: 'complete', reason: null, message: null, browser: 'Microsoft Edge',
    browser_path: 'C:\\Edge\\msedge.exe', milliseconds: 812, bytes: 243387,
    pixels: { width: 2160, height: 1896 }, pixels_match: true,
    created_at: '2026-09-18T10:02:00+00:00',
  },
  degraded: false,
  ...over,
})

const degraded = (): InfographicExport => exported({
  degraded: true,
  render: {
    status: 'unavailable', reason: 'browser_unavailable',
    message: '没有找到可用的浏览器（Edge 或 Chrome）。可以先设置环境变量',
    browser: null, browser_path: null, milliseconds: null, bytes: null,
    pixels: null, pixels_match: null, created_at: '2026-09-18T10:02:00+00:00',
  },
})

const mocks = vi.hoisted(() => ({
  list: vi.fn(),
  detail: vi.fn(),
  remove: vi.fn(),
  removeBatch: vi.fn(),
  stop: vi.fn(),
  stream: vi.fn(),
  source: vi.fn(),
  exportInfographic: vi.fn(),
  infographicExport: vi.fn(),
}))

vi.mock('./api', () => ({
  api: {
    artifacts: () => mocks.list(),
    artifact: (id: string) => mocks.detail(id),
    deleteArtifact: (id: string) => mocks.remove(id),
    deleteArtifacts: (ids: string[]) => mocks.removeBatch(ids),
    stopArtifact: (id: string) => mocks.stop(id),
    streamArtifact: (topic: string, kind: string, onEvent: unknown, signal: unknown) =>
      mocks.stream(topic, kind, onEvent, signal),
    source: () => mocks.source(),
    exportInfographic: (id: string) => mocks.exportInfographic(id),
    infographicExport: (id: string) => mocks.infographicExport(id),
    infographicImage: (id: string, stamp?: string) =>
      `/api/v1/artifacts/${id}/infographic.png${stamp ? `?v=${stamp}` : ''}`,
    infographicDownload: (id: string) => `/api/v1/artifacts/${id}/infographic.png?download=1`,
  },
}))

let listed: ArtifactSummary[]

beforeEach(() => {
  for (const mock of Object.values(mocks)) mock.mockReset()
  listed = [{ ...summary }]
  mocks.list.mockImplementation(async () => ({ artifacts: listed }))
  mocks.detail.mockImplementation(async (id: string) => ({
    ...summary, id, content: '分页管理 KV Cache [S1]。', sources: [source], backlink: report(),
  } satisfies Artifact))
  mocks.remove.mockResolvedValue({ deleted: true })
  mocks.removeBatch.mockResolvedValue({ deleted: 0, skipped: [] })
  mocks.stop.mockResolvedValue({ stopping: true })
  mocks.stream.mockResolvedValue(undefined)
  mocks.source.mockResolvedValue({ document_id: 'doc_a', version_id: 'ver_a', title: '推理.md',
    media_type: 'markdown', text: '分页管理 KV Cache。' })
  mocks.infographicExport.mockResolvedValue({ artifact_id: 'art_1', export: null })
  mocks.exportInfographic.mockResolvedValue(exported())
})

/** 走一次"填主题 → 生成"，把事件按顺序喂给页面。 */
async function generate(events: Event[], topic = 'KV Cache', kind = 'guide') {
  mocks.stream.mockImplementation(async (_topic: string, _kind: string,
    onEvent: (event: Event) => void) => { for (const event of events) onEvent(event) })
  render(<StudioPage />)
  fireEvent.change(screen.getByLabelText('主题'), { target: { value: topic } })
  if (kind === 'mindmap') fireEvent.click(screen.getByRole('button', { name: '思维导图' }))
  fireEvent.click(screen.getByRole('button', { name: '开始生成' }))
  await waitFor(() => expect(mocks.stream).toHaveBeenCalled())
}

describe('StudioPage backlink reporting', () => {
  it('shows the hit rate the backend computed for the guide', async () => {
    await generate([
      { type: 'retrieval', artifact_id: 'art_1', sources: [source] },
      { type: 'token', artifact_id: 'art_1', text: '分页管理 KV Cache [S1]。' },
      { type: 'final', artifact_id: 'art_1', status: 'complete', content: '分页管理 KV Cache [S1]。',
        sources: [source], backlink: report() },
    ])

    const rate = await screen.findByTestId('backlink-rate')
    expect(rate).toHaveTextContent('67%')
    expect(screen.getByText('3 句陈述中，2 句带了来源。')).toBeInTheDocument()
  })

  it('does not present an unmeasurable rate as a perfect one', async () => {
    // 空文档报 100% 是最容易发生的一种"数字说谎"，界面必须原样显示"无法统计"。
    await generate([
      { type: 'retrieval', artifact_id: 'art_1', sources: [source] },
      { type: 'final', artifact_id: 'art_1', status: 'complete', content: '# 只有标题',
        sources: [source],
        backlink: report({ assertions: 0, with_source: 0, hit_rate: null, missing_count: 0, missing: [] }) },
    ])

    const rate = await screen.findByTestId('backlink-rate')
    expect(rate).toHaveTextContent('无可统计的句子')
    expect(rate).not.toHaveTextContent('100%')
  })

  it('lists the sentences that carried no source', async () => {
    await generate([
      { type: 'retrieval', artifact_id: 'art_1', sources: [source] },
      { type: 'final', artifact_id: 'art_1', status: 'complete', content: 'x', sources: [source],
        backlink: report() },
    ])

    expect(await screen.findByText('这句没有来源。')).toBeInTheDocument()
    expect(screen.getByText('1 句没有来源（列出前 1 句）')).toBeInTheDocument()
  })

  it('calls out citation numbers that do not exist', async () => {
    // 引用了不存在的编号比"漏引"更隐蔽，因为它看起来像有来源，所以要单独说。
    await generate([
      { type: 'retrieval', artifact_id: 'art_1', sources: [source] },
      { type: 'final', artifact_id: 'art_1', status: 'complete', content: 'x', sources: [source],
        backlink: report({ invalid_labels: ['S7', 'S9'], with_source: 1 }) },
    ])

    expect(await screen.findByText(/引用了不存在的编号：S7、S9/)).toBeInTheDocument()
  })

  it('reports a failure instead of showing an empty guide as a success', async () => {
    await generate([
      { type: 'retrieval', artifact_id: 'art_1', sources: [] },
      { type: 'final', artifact_id: 'art_1', status: 'failed', content: '', error: 'no_evidence',
        message: '当前资料里没有找到与这个主题相关的片段，无法产出。可以先添加资料或换个主题。', sources: [] },
    ])

    expect(await screen.findByText(/没有找到与这个主题相关的片段/)).toBeInTheDocument()
  })
})

describe('StudioPage mindmap', () => {
  it('renders the tree with a link back to the fragment on every node', async () => {
    await generate([
      { type: 'retrieval', artifact_id: 'art_2', sources: [source] },
      { type: 'final', artifact_id: 'art_2', status: 'complete', content: mindmap.mermaid,
        sources: [source], mindmap },
    ], 'KV Cache', 'mindmap')

    expect(await screen.findByTestId('mindmap-view')).toBeInTheDocument()
    expect(screen.getByText('PagedAttention')).toBeInTheDocument()
    expect(screen.getByText('推理.md')).toBeInTheDocument()
    // 节点的编号是可点的，点了要能打开原文片段
    fireEvent.click(screen.getByRole('button', { name: 'S1' }))
    await waitFor(() => expect(mocks.source).toHaveBeenCalled())
  })

  it('says the coverage is constructional rather than showing a score', async () => {
    await generate([
      { type: 'retrieval', artifact_id: 'art_2', sources: [source] },
      { type: 'final', artifact_id: 'art_2', status: 'complete', content: mindmap.mermaid,
        sources: [source], mindmap },
    ], 'KV Cache', 'mindmap')

    expect(await screen.findByTestId('mindmap-coverage')).toHaveTextContent('构造结果')
    // 导图不该出现命中率卡片：一个必然接近 1 的数字和指南的命中率并列会误导
    expect(screen.queryByTestId('backlink-report')).not.toBeInTheDocument()
    expect(screen.queryByTestId('backlink-rate')).not.toBeInTheDocument()
  })

  it('offers the mermaid text for copying without claiming to render it', async () => {
    await generate([
      { type: 'retrieval', artifact_id: 'art_2', sources: [source] },
      { type: 'final', artifact_id: 'art_2', status: 'complete', content: mindmap.mermaid,
        sources: [source], mindmap },
    ], 'KV Cache', 'mindmap')

    expect(await screen.findByText('本工作台不渲染 Mermaid，只提供可复制的文本。'))
      .toBeInTheDocument()
  })
})

describe('StudioPage lifecycle', () => {
  it('refuses to start without a topic instead of calling the backend', async () => {
    render(<StudioPage />)

    fireEvent.click(screen.getByRole('button', { name: '开始生成' }))

    expect(await screen.findByText('请先填一个主题。')).toBeInTheDocument()
    expect(mocks.stream).not.toHaveBeenCalled()
  })

  it('stops a running generation on the backend and then locally', async () => {
    // 流永不结束：模拟"生成很久"，此时用户点停止。
    mocks.stream.mockImplementation((_topic: string, _kind: string,
      onEvent: (event: Event) => void, signal: AbortSignal) =>
      new Promise<void>((_resolve, reject) => {
        onEvent({ type: 'retrieval', artifact_id: 'art_9', sources: [source] })
        signal.addEventListener('abort', () =>
          reject(new DOMException('aborted', 'AbortError')))
      }))
    render(<StudioPage />)
    fireEvent.change(screen.getByLabelText('主题'), { target: { value: 'KV Cache' } })
    fireEvent.click(screen.getByRole('button', { name: '开始生成' }))
    const stop = await screen.findByRole('button', { name: '停止' })

    fireEvent.click(stop)

    // 先让服务端停（它才知道流跑在哪），所以这个调用必须在
    await waitFor(() => expect(mocks.stop).toHaveBeenCalledWith('art_9'))
    expect(await screen.findByText(/已停止。已经写出的部分保留在这份产出里。/)).toBeInTheDocument()
  })

  it('opens a stored artifact and shows the rate recomputed from its text', async () => {
    await generate([])
    mocks.detail.mockResolvedValue({
      ...summary, content: '分页管理 KV Cache [S1]。', sources: [source],
      backlink: report({ assertions: 1, with_source: 1, hit_rate: 1, missing_count: 0, missing: [] }),
    })

    fireEvent.click(await screen.findByRole('button', { name: /KV Cache/ }))

    await waitFor(() => expect(mocks.detail).toHaveBeenCalledWith('art_1'))
    expect(await screen.findByTestId('backlink-rate')).toHaveTextContent('100%')
  })

  it('lets you open a past artifact after a failed generation', async () => {
    // 真实缺陷：失败说明挂在屏幕上时，`showingLive` 一直为真，右侧永远显示那次的
    // 残留，点历史里任何一份都打不开——删除按钮也就永远点不到。无头验收脚本抓到
    // 的就是这一条，这里固定住。
    await generate([
      { type: 'retrieval', artifact_id: 'art_1', sources: [] },
      { type: 'final', artifact_id: 'art_1', status: 'failed', content: '', error: 'no_evidence',
        message: '当前资料里没有找到与这个主题相关的片段，无法产出。', sources: [] },
    ])
    expect(await screen.findByText(/没有找到与这个主题相关的片段/)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /KV Cache/ }))

    expect(await screen.findByRole('button', { name: '删除' })).toBeInTheDocument()
  })

  it('removes an artifact only after the user confirms', async () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true)
    await generate([])
    fireEvent.click(await screen.findByRole('button', { name: /KV Cache/ }))
    await screen.findByTestId('backlink-report')

    fireEvent.click(screen.getByRole('button', { name: '删除' }))

    await waitFor(() => expect(mocks.remove).toHaveBeenCalledWith('art_1'))
    confirm.mockRestore()
  })
})

describe('StudioPage infographic export', () => {
  /** 打开左侧历史里的第一份产出。 */
  async function openFirst(name = /KV Cache/) {
    render(<StudioPage />)
    fireEvent.click(await screen.findByRole('button', { name }))
    return screen.findByTestId('infographic-card')
  }

  it('offers the export before anything has been rendered', async () => {
    await openFirst()

    expect(screen.queryByTestId('infographic-image')).not.toBeInTheDocument()
    expect(screen.getByTestId('infographic-export')).toHaveTextContent('导出信息图')
  })

  it('shows the image, what it cost to render, and a download link', async () => {
    await openFirst()

    fireEvent.click(screen.getByTestId('infographic-export'))

    await waitFor(() => expect(mocks.exportInfographic).toHaveBeenCalledWith('art_1'))
    const image = await screen.findByTestId('infographic-image')
    // 带一串渲染时间当缓存键：重新导出后不能再显示上一次那张图。
    expect(image).toHaveAttribute(
      'src', '/api/v1/artifacts/art_1/infographic.png?v=2026-09-18T10:02:00+00:00')
    expect(screen.getByTestId('infographic-milliseconds')).toHaveTextContent('812 毫秒')
    expect(screen.getByText('2160×1896 像素')).toBeInTheDocument()
    expect(screen.getByText('Microsoft Edge')).toBeInTheDocument()
    expect(screen.getByTestId('infographic-download')).toHaveAttribute(
      'href', '/api/v1/artifacts/art_1/infographic.png?download=1')
  })

  it('keeps the exported HTML as a way out when no browser is available', async () => {
    // 降级要能自救：说清为什么、并且给出 HTML 这条退路，且**不显示一张不存在的图**。
    mocks.exportInfographic.mockResolvedValue(degraded())
    await openFirst()

    fireEvent.click(screen.getByTestId('infographic-export'))

    const block = await screen.findByTestId('infographic-degraded')
    expect(block).toHaveTextContent('没有找到可用的浏览器')
    expect(block).toHaveTextContent('OBSIDIAN_RAG_BROWSER')
    expect(block).toHaveTextContent('KV Cache-01234567.html')
    expect(screen.queryByTestId('infographic-image')).not.toBeInTheDocument()
    // 它不该被当成故障显示（红色错误块）：用户手里有 HTML，这不是坏掉。
    expect(block.querySelector('.error')).toBeNull()
  })

  it('reports a browser that ignored the requested size', async () => {
    mocks.exportInfographic.mockResolvedValue(exported({
      render: {
        ...exported().render, pixels: { width: 800, height: 600 }, pixels_match: false,
      },
    }))
    await openFirst()

    fireEvent.click(screen.getByTestId('infographic-export'))

    expect(await screen.findByText(/浏览器没有按请求的尺寸出图/)).toBeInTheDocument()
  })

  it('shows the record already on disk when you open a past artifact', async () => {
    mocks.infographicExport.mockResolvedValue({ artifact_id: 'art_1', export: exported() })

    await openFirst()

    expect(await screen.findByTestId('infographic-image')).toBeInTheDocument()
    expect(screen.getByTestId('infographic-export')).toHaveTextContent('重新导出')
  })

  it('does not carry the previous image over to the next artifact', async () => {
    // 这个页面踩过的坑正是"上一次的状态跟了过来"（见 clearLive 的注释）。这里固定的是
    // **行为**：换一份产出就不能再看到上一份的图。它由两处共同保证——卡片在换 `key`
    // 时重挂、以及 `artifactId` 变化时清空本地记录；任一处单独去掉，这条仍会通过。
    listed = [{ ...summary },
      { ...summary, id: 'art_2', title: '缓存策略', kind: 'mindmap', content_length: 40 }]
    mocks.infographicExport.mockImplementation(async (id: string) =>
      ({ artifact_id: id, export: id === 'art_1' ? exported() : null }))
    mocks.detail.mockImplementation(async (id: string) => ({
      ...summary, id, title: id === 'art_1' ? 'KV Cache' : '缓存策略',
      kind: id === 'art_1' ? 'guide' : 'mindmap', content: '', sources: [source],
    } satisfies Artifact))
    render(<StudioPage />)

    fireEvent.click(await screen.findByRole('button', { name: /KV Cache/ }))
    expect(await screen.findByTestId('infographic-image')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /缓存策略/ }))

    await waitFor(() =>
      expect(screen.queryByTestId('infographic-image')).not.toBeInTheDocument())
    expect(screen.getByTestId('infographic-export')).toHaveTextContent('导出信息图')
  })

  it('reports a failed export instead of silently doing nothing', async () => {
    mocks.exportInfographic.mockRejectedValue(new Error('无法连接到工作台。'))
    await openFirst()

    fireEvent.click(screen.getByTestId('infographic-export'))

    expect(await screen.findByText('无法连接到工作台。')).toBeInTheDocument()
    expect(screen.getByTestId('infographic-export')).not.toBeDisabled()
  })
})

describe('StudioPage 的证据裁剪说明', () => {
  it('final 事件带 evidence_note 时显示在产出区', async () => {
    await generate([
      { type: 'retrieval', artifact_id: 'art_1', sources: [source],
        evidence_note: '模型上下文窗口为 8192 token，证据从 12 条减到 6 条；被减掉的条目没有参与生成。' },
      { type: 'token', artifact_id: 'art_1', text: '分页管理 KV Cache [S1]。' },
      { type: 'final', artifact_id: 'art_1', status: 'complete', content: '分页管理 KV Cache [S1]。',
        sources: [source],
        evidence_note: '模型上下文窗口为 8192 token，证据从 12 条减到 6 条；被减掉的条目没有参与生成。',
        backlink: report() },
    ])

    expect(await screen.findByTestId('live-evidence-note')).toHaveTextContent('没有参与生成')
  })
})

describe('StudioPage 的逐句 JSON 过程显示', () => {
  it('生成中显示进度提示，不把原始 JSON 碎片当正文渲染', async () => {
    // token 是原始 JSON 碎片：过程直接渲染会闪出半截 JSON，过程只报收到的字数，
    // 正文等 final 渲染完成再显示。
    const fragment = '{"sections": [{"s": "第一句。", "src": [1]}'
    let release: () => void = () => {}
    const gate = new Promise<void>(resolve => { release = resolve })
    mocks.stream.mockImplementation(async (_topic: string, _kind: string,
      onEvent: (event: Event) => void) => {
      onEvent({ type: 'retrieval', artifact_id: 'art_1', sources: [source] })
      onEvent({ type: 'token', artifact_id: 'art_1', text: fragment })
      await gate
      onEvent({ type: 'final', artifact_id: 'art_1', status: 'complete',
        content: '第一句。 [S1]', sources: [source],
        backlink: report({ assertions: 1, with_source: 1, hit_rate: 1,
          missing_count: 0, missing: [] }) })
    })
    render(<StudioPage />)
    fireEvent.change(screen.getByLabelText('主题'), { target: { value: 'KV Cache' } })
    fireEvent.click(screen.getByRole('button', { name: '开始生成' }))

    expect(await screen.findByTestId('live-progress')).toHaveTextContent(
      `已收到 ${fragment.length} 字`)
    expect(screen.queryByText(/第一句/)).not.toBeInTheDocument()

    release()
    expect(await screen.findByText(/第一句/)).toBeInTheDocument()
    expect(screen.queryByTestId('live-progress')).not.toBeInTheDocument()
  })

  it('停止后正文以服务端回传为准，不再残留原始 JSON 碎片', async () => {
    await generate([
      { type: 'retrieval', artifact_id: 'art_1', sources: [source] },
      { type: 'token', artifact_id: 'art_1', text: '{"sections": [{"s": "第一句。", "src": [1]}' },
      { type: 'stopped', artifact_id: 'art_1', status: 'stopped',
        content: '第一句。 [S1]', sources: [source],
        backlink: report({ assertions: 1, with_source: 1, hit_rate: 1,
          missing_count: 0, missing: [] }) },
    ])

    expect(await screen.findByText(/第一句/)).toBeInTheDocument()
    expect(screen.queryByText(/"sections"/)).not.toBeInTheDocument()
  })
})

describe('StudioPage 的批量删除', () => {
  it('选择模式里勾选历史产出，确认文案说明「正在生成会跳过」', async () => {
    mocks.removeBatch.mockResolvedValue({ deleted: 1, skipped: [], files_removed: 1 })
    render(<StudioPage />)

    fireEvent.click(await screen.findByRole('button', { name: '批量选择' }))
    fireEvent.click(await screen.findByRole('button', { name: /KV Cache/ }))
    expect(await screen.findByText('已选 1 项')).toBeInTheDocument()

    fireEvent.click(screen.getByTestId('batch-confirm'))
    expect(await screen.findByText('删除 1 份产出？')).toBeInTheDocument()
    expect(screen.getByText('随产出导出的图片也会一并删除。')).toBeInTheDocument()
    expect(screen.getByText('正在生成的产出会被跳过，不会被删除。')).toBeInTheDocument()

    fireEvent.click(screen.getByTestId('batch-confirm'))
    await waitFor(() => expect(mocks.removeBatch).toHaveBeenCalledWith(['art_1']))
    expect(await screen.findByText(/已删除 1 项/)).toBeInTheDocument()
    expect(screen.getByText(/一并清理导出文件 1 个/)).toBeInTheDocument()
  })

  it('结果条里的 skipped 项带标题与原因', async () => {
    mocks.removeBatch.mockResolvedValue({ deleted: 0, skipped: [
      { id: 'art_1', label: 'KV Cache', code: 'busy', reason: '这份产出还在生成，请先停止再删除。' },
    ] })
    render(<StudioPage />)

    fireEvent.click(await screen.findByRole('button', { name: '批量选择' }))
    fireEvent.click(await screen.findByRole('button', { name: /KV Cache/ }))
    await screen.findByText('已选 1 项')
    fireEvent.click(screen.getByTestId('batch-confirm'))
    fireEvent.click(await screen.findByTestId('batch-confirm'))

    expect(await screen.findByText('已删除 0 项')).toBeInTheDocument()
    expect(screen.getByText(/KV Cache：这份产出还在生成/)).toBeInTheDocument()
    // 产出没有被删，列表还在
    expect(screen.getByRole('button', { name: /KV Cache/ })).toBeInTheDocument()
  })
})
