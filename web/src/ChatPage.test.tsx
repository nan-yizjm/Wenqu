import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { Conversation, MessageSource } from './api'
import { ChatPage } from './ChatPage'

const seed: Conversation[] = [
  { id: 'c1', title: '第一个会话', created_at: '2026-01-01', updated_at: '2026-01-01', message_count: 2 },
  { id: 'c2', title: '第二个会话', created_at: '2026-01-02', updated_at: '2026-01-02', message_count: 4 },
]

// vi.mock 会被提升到文件顶部执行，所以共享桩必须放进 vi.hoisted。
const mocks = vi.hoisted(() => ({
  list: vi.fn(),
  remove: vi.fn(),
  create: vi.fn(),
  detail: vi.fn(),
  source: vi.fn(),
}))

vi.mock('./api', () => ({
  api: {
    conversations: () => mocks.list(),
    conversation: (id: string) => mocks.detail(id),
    createConversation: () => mocks.create(),
    deleteConversation: (id: string) => mocks.remove(id),
    streamMessage: vi.fn(),
    stopMessage: vi.fn(),
    createFavorite: vi.fn(),
    feedback: vi.fn(),
    source: (documentId: string, versionId: string) => mocks.source(documentId, versionId),
  },
}))

let current: Conversation[]

beforeEach(() => {
  mocks.list.mockReset(); mocks.remove.mockReset()
  mocks.create.mockReset(); mocks.detail.mockReset(); mocks.source.mockReset()
  mocks.source.mockResolvedValue({
    document_id: 'doc_a', version_id: 'ver_a', title: '推理.md',
    media_type: 'markdown', text: '分页管理 KV Cache。',
  })
  current = [...seed]
  mocks.list.mockImplementation(async () => ({ conversations: current }))
  // 打开会话要带上真实标题，否则顶栏标题会退化成占位文案，测不到"切到哪个会话"。
  mocks.detail.mockImplementation(async (id: string) => ({
    ...(current.find(item => item.id === id)
      ?? { id, title: '新会话', created_at: '', updated_at: '' }),
    messages: [],
  }))
  mocks.remove.mockImplementation(async (id: string) => {
    const target = current.find(item => item.id === id)!
    current = current.filter(item => item.id !== id)
    return { deleted: true, title: target.title, messages: target.message_count, kept_favorites: 0 }
  })
  // jsdom 没有布局，scrollIntoView 不存在。
  Element.prototype.scrollIntoView = vi.fn()
})

describe('ChatPage conversation deletion', () => {
  it('offers a delete control on every conversation', async () => {
    render(<ChatPage />)

    expect(await screen.findByLabelText('删除会话 第一个会话')).toBeInTheDocument()
    expect(screen.getByLabelText('删除会话 第二个会话')).toBeInTheDocument()
  })

  it('deletes the conversation once the user confirms', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    render(<ChatPage />)

    fireEvent.click(await screen.findByLabelText('删除会话 第二个会话'))

    await waitFor(() => expect(mocks.remove).toHaveBeenCalledWith('c2'))
    await waitFor(() =>
      expect(screen.queryByLabelText('删除会话 第二个会话')).not.toBeInTheDocument())
    expect(screen.getByLabelText('删除会话 第一个会话')).toBeInTheDocument()
  })

  it('deletes nothing when the user cancels', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(false)
    render(<ChatPage />)

    fireEvent.click(await screen.findByLabelText('删除会话 第二个会话'))

    expect(mocks.remove).not.toHaveBeenCalled()
    expect(screen.getByLabelText('删除会话 第二个会话')).toBeInTheDocument()
  })

  it('moves to another conversation when the open one is deleted', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    render(<ChatPage />)
    // 打开的是列表里的第一个。
    await waitFor(() => expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('第一个会话'))

    fireEvent.click(screen.getByLabelText('删除会话 第一个会话'))

    // 删掉的正是当前会话，界面不能停在一个已经不存在的会话上。
    await waitFor(() => expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('第二个会话'))
  })

  it('starts a fresh conversation when the last one is deleted', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    current = [seed[0]]
    mocks.create.mockResolvedValue({ id: 'c-new', title: '新会话', created_at: '', updated_at: '', message_count: 0 })
    render(<ChatPage />)
    await waitFor(() => expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('第一个会话'))

    fireEvent.click(screen.getByLabelText('删除会话 第一个会话'))

    await waitFor(() => expect(mocks.create).toHaveBeenCalled())
  })
})

const NOTE_SOURCE: Omit<MessageSource, 'label'> = {
  origin: 'note', chunk_id: 'chk_a', document_id: 'doc_a', version_id: 'ver_a',
  title: '推理.md', media_type: 'markdown', heading_path: 'PagedAttention',
  locator: { kind: 'markdown', start_line: 1, end_line: 2 },
  preview: '分页管理 KV Cache。', score: 1, matched_tokens: [], channels: {},
}

// 真实后端把记忆条目的标题固定写成"记忆"，那会和层名撞成同一个字符串，
// 查询就分不出"这是层名还是条目名"。这里换个名字只为了让断言能区分。
const MEMORY_SOURCE: Omit<MessageSource, 'label'> = {
  origin: 'memory', chunk_id: 'memory:m1', document_id: 'memory', version_id: 'memory',
  title: '偏好', media_type: 'memory', heading_path: '记忆',
  locator: { kind: 'memory', id: 'm1', derived_from: '关于排版的旧对话' },
  preview: '用户偏好用对比表格看结论。', score: null, matched_tokens: [], channels: {},
}

function conversationWith(sources: Omit<MessageSource, 'label'>[]): Conversation {
  return {
    id: 'c1', title: '一个会话', created_at: '2026-01-01', updated_at: '2026-01-01',
    messages: [{
      id: 'a1', conversation_id: 'c1', role: 'assistant', content: '答案 [S1]。',
      status: 'complete', provider: null, model: null, index_version: null,
      error_code: null, reply_to_message_id: null,
      sources: sources.map((source, index) => ({ ...source, label: `S${index + 1}` })),
    }],
  }
}

describe('ChatPage source layering', () => {
  it('shows no layer name while every source comes from the notes', async () => {
    mocks.detail.mockImplementation(async () => conversationWith([NOTE_SOURCE]))

    render(<ChatPage />)
    await screen.findByText('推理.md')

    // 出厂状态只有笔记层：界面上不该冒出"笔记"这种标签，多一层视觉噪音。
    expect(screen.queryByText('笔记')).not.toBeInTheDocument()
    expect(screen.queryByText('记忆')).not.toBeInTheDocument()
  })

  it('names each layer once a second one appears', async () => {
    mocks.detail.mockImplementation(async () => conversationWith([NOTE_SOURCE, MEMORY_SOURCE]))

    render(<ChatPage />)
    await screen.findByText('推理.md')

    expect(screen.getByText('笔记')).toBeInTheDocument()
    expect(screen.getByText('记忆')).toBeInTheDocument()
    expect(screen.getByText('偏好')).toBeInTheDocument()
  })

  it('opens a memory source from its own text, without fetching a document snapshot', async () => {
    mocks.detail.mockImplementation(async () => conversationWith([NOTE_SOURCE, MEMORY_SOURCE]))
    render(<ChatPage />)
    await screen.findByText('推理.md')

    fireEvent.click(screen.getByText('偏好'))

    expect(await screen.findByText('用户偏好用对比表格看结论。')).toBeInTheDocument()
    // 记忆来源的 document_id 是占位串：照着它去取原文只会拿到 404。
    expect(mocks.source).not.toHaveBeenCalled()
  })

  it('highlights only the clicked answer when two answers reuse the S1 label', async () => {
    // S1/S2 是**每轮回答各自从 1 开始**的编号，两轮之间必然重号。按 label 比较
    // 会让另一轮的同号卡片跟着亮，按 chunk_id 比较才只亮一个。
    const other: Conversation = conversationWith([NOTE_SOURCE])
    other.messages!.push({
      ...other.messages![0], id: 'a2',
      sources: [{ ...NOTE_SOURCE, chunk_id: 'chk_b', title: '另一篇.md', label: 'S1' }],
    })
    mocks.detail.mockImplementation(async () => other)

    render(<ChatPage />)
    const labels = await screen.findAllByText('推理.md')
    expect(labels).toHaveLength(1)

    fireEvent.click(labels[0])

    const pressed = await screen.findAllByRole('button', { pressed: true })
    expect(pressed).toHaveLength(1)
  })
})
