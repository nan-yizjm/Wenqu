import { render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { SourceRef } from './api'
import { SourcePanel } from './SourcePanel'

const mocks = vi.hoisted(() => ({ source: vi.fn() }))

vi.mock('./api', () => ({
  api: {
    source: (documentId: string, versionId: string) => mocks.source(documentId, versionId),
    sourceFileUrl: (documentId: string, versionId: string) =>
      `/api/v1/documents/${documentId}/versions/${versionId}/file`,
  },
}))

const MEMORY_TEXT = '用户偏好用对比表格看结论。'

const memoryHit: SourceRef = {
  origin: 'memory', chunk_id: 'memory:m1', document_id: 'memory', version_id: 'memory',
  title: '记忆', media_type: 'memory', heading_path: '记忆',
  locator: { kind: 'memory', id: 'm1', derived_from: '关于排版的旧对话' },
  preview: MEMORY_TEXT,
}

const noteHit: SourceRef = {
  origin: 'note', chunk_id: 'chk_a', document_id: 'doc_a', version_id: 'ver_a',
  title: '推理.md', media_type: 'markdown', heading_path: 'PagedAttention',
  locator: { kind: 'markdown', start_line: 3, end_line: 4 },
  preview: '分页管理 KV Cache。', score: 1,
}

const WEB_URL = 'https://arxiv.org/abs/2309.06180'
const WEB_SNIPPET = 'PagedAttention 把 KV Cache 分成固定大小的页来管理。'

const webHit: SourceRef = {
  origin: 'web', chunk_id: 'web:w1', document_id: 'web', version_id: 'web',
  title: 'PagedAttention 原论文', media_type: 'web', heading_path: 'arxiv.org',
  locator: {
    kind: 'web', url: WEB_URL,
    published_at: '2023-09-12T00:00:00Z', retrieved_at: '2026-09-18T10:00:00Z',
  },
  preview: WEB_SNIPPET,
}

beforeEach(() => {
  mocks.source.mockReset()
})

describe('SourcePanel source layers', () => {
  it('renders a memory item from its own text, without asking for a document snapshot', async () => {
    render(<SourcePanel hit={memoryHit} close={() => {}} />)

    // 正文来自 preview 本身：记忆条目的 document_id 是占位串，按它去请求
    // /versions/memory/source 只会拿到 404，界面上表现为一个报错面板。
    expect(await screen.findByText(MEMORY_TEXT)).toBeInTheDocument()
    expect(screen.getByText(/来自记忆 · 关于排版的旧对话/)).toBeInTheDocument()
    expect(mocks.source).not.toHaveBeenCalled()
  })

  it('does not promise a historical snapshot for a memory item', () => {
    render(<SourcePanel hit={memoryHit} close={() => {}} />)

    // "历史快照"只对真正的笔记片段成立：记忆条目没有版本可言。
    expect(screen.queryByText(/历史快照/)).not.toBeInTheDocument()
    expect(screen.getByText('SOURCE LAYER')).toBeInTheDocument()
  })

  it('still fetches the snapshot for a note source', async () => {
    mocks.source.mockResolvedValue({
      document_id: 'doc_a', version_id: 'ver_a', title: '推理.md',
      media_type: 'markdown', text: '第一行\n分页管理 KV Cache。',
    })

    render(<SourcePanel hit={noteHit} close={() => {}} />)

    await waitFor(() => expect(mocks.source).toHaveBeenCalledWith('doc_a', 'ver_a'))
    expect(await screen.findByText(/分页管理 KV Cache/)).toBeInTheDocument()
    expect(screen.getByText(/历史快照/)).toBeInTheDocument()
  })

  it('treats a source stored before the origin column as a note', async () => {
    // v8 之前的记录没有 origin 字段。它们本来就是笔记片段，不能因为缺字段
    // 就被当成"非笔记"而跳过取原文。
    mocks.source.mockResolvedValue({
      document_id: 'doc_a', version_id: 'ver_a', title: '推理.md',
      media_type: 'markdown', text: '分页管理 KV Cache。',
    })

    render(<SourcePanel hit={{ ...noteHit, origin: undefined }} close={() => {}} />)

    await waitFor(() => expect(mocks.source).toHaveBeenCalledWith('doc_a', 'ver_a'))
  })

  it('opens a web source at its url instead of asking for a local snapshot', async () => {
    render(<SourcePanel hit={webHit} close={() => {}} />)

    const link = await screen.findByRole('link', { name: WEB_URL })
    // 网络来源的 document_id 是占位串，按它去请求 /versions/web/source 只会 404。
    expect(mocks.source).not.toHaveBeenCalled()
    expect(link).toHaveAttribute('href', WEB_URL)
    // 外链必须带 noopener noreferrer：新开的页面不该拿到本工作台的 window 引用。
    expect(link).toHaveAttribute('rel', 'noopener noreferrer')
    expect(screen.getByText(WEB_SNIPPET)).toBeInTheDocument()
  })

  it('keeps publication time and fetch time apart for a web source', async () => {
    render(<SourcePanel hit={webHit} close={() => {}} />)

    // 这两个时间差很远时才是关键（2023 年的论文，今天才被搜到）。混成一个
    // "时间" 会让用户以为这条事实是今天成立的。
    expect(await screen.findByText('2023-09-12 00:00')).toBeInTheDocument()
    expect(screen.getByText('2026-09-18 10:00')).toBeInTheDocument()
    expect(screen.getByText(/没有正文快照/)).toBeInTheDocument()
  })

  it('says so instead of pretending a web source has a link', async () => {
    render(<SourcePanel hit={{
      ...webHit,
      locator: { kind: 'web', url: '', published_at: null, retrieved_at: null },
    }} close={() => {}} />)

    expect(await screen.findByText('这条来源没有带链接')).toBeInTheDocument()
    expect(screen.getByText('未知')).toBeInTheDocument()
    expect(screen.getByText('未记录')).toBeInTheDocument()
  })
})
