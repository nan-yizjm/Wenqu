import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { Artifact, ArtifactStreamEvent, ArtifactSummary, Mindmap } from './api'
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

const mocks = vi.hoisted(() => ({
  list: vi.fn(),
  detail: vi.fn(),
  remove: vi.fn(),
  stop: vi.fn(),
  stream: vi.fn(),
  source: vi.fn(),
}))

vi.mock('./api', () => ({
  api: {
    artifacts: () => mocks.list(),
    artifact: (id: string) => mocks.detail(id),
    deleteArtifact: (id: string) => mocks.remove(id),
    stopArtifact: (id: string) => mocks.stop(id),
    streamArtifact: (topic: string, kind: string, onEvent: unknown, signal: unknown) =>
      mocks.stream(topic, kind, onEvent, signal),
    source: () => mocks.source(),
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
  mocks.stop.mockResolvedValue({ stopping: true })
  mocks.stream.mockResolvedValue(undefined)
  mocks.source.mockResolvedValue({ document_id: 'doc_a', version_id: 'ver_a', title: '推理.md',
    media_type: 'markdown', text: '分页管理 KV Cache。' })
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
