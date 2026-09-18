import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { Conversation, MessageSource, WebState } from './api'
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
  stream: vi.fn(),
}))

vi.mock('./api', () => ({
  api: {
    conversations: () => mocks.list(),
    conversation: (id: string) => mocks.detail(id),
    createConversation: () => mocks.create(),
    deleteConversation: (id: string) => mocks.remove(id),
    streamMessage: (id: string, body: unknown, onEvent: unknown, signal: unknown) =>
      mocks.stream(id, body, onEvent, signal),
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
  mocks.stream.mockReset()
  mocks.stream.mockResolvedValue(undefined)
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
      error_code: null, reply_to_message_id: null, web_state: null,
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

const WEB_SOURCE: Omit<MessageSource, 'label'> = {
  origin: 'web', chunk_id: 'web:w1', document_id: 'web', version_id: 'web',
  title: 'PagedAttention 原论文', media_type: 'web', heading_path: 'arxiv.org',
  locator: {
    kind: 'web', url: 'https://arxiv.org/abs/2309.06180',
    published_at: '2023-09-12T00:00:00Z', retrieved_at: '2026-09-18T10:00:00Z',
  },
  preview: 'PagedAttention 把 KV Cache 分成固定大小的页来管理。',
  score: null, matched_tokens: [], channels: {},
}

describe('ChatPage web notice', () => {
  /**
   * 问一轮，让后端按给定的联网状态回一次流。
   *
   * 重取会话的桩也要返回同一条消息**连同同一个 `web_state`**：答完之后组件会在
   * 350ms 后重新拉一次会话，服务端持久化过的东西就该原样回来。桩里漏掉它，测的
   * 就是"组件记住了"而不是"库里记住了"——那正是这次要修的东西。
   */
  async function ask(web: WebState, sources: Omit<MessageSource, 'label'>[] = [WEB_SOURCE]) {
    const labelled = sources.map((source, index) => ({ ...source, label: `S${index + 1}` }))
    const answer: Conversation = {
      id: 'c1', title: '一个会话', created_at: '2026-01-01', updated_at: '2026-01-01',
      messages: [{
        id: 'a-new', conversation_id: 'c1', role: 'assistant', content: '答案 [S1]。',
        status: 'complete', provider: null, model: null, index_version: null,
        error_code: null, reply_to_message_id: null, sources: labelled, web_state: web,
      }],
    }
    mocks.detail.mockImplementation(async () => answer)
    mocks.stream.mockImplementation(async (_id: string, _body: unknown, onEvent: (event: unknown) => void) => {
      onEvent({ type: 'retrieval', message_id: 'a-new', sources: labelled, web })
      onEvent({ type: 'final', message_id: 'a-new', content: '答案 [S1]。', status: 'complete', sources: labelled })
    })
    render(<ChatPage />)
    const box = await screen.findByPlaceholderText('询问你的资料；Shift + Enter 换行')
    fireEvent.change(box, { target: { value: 'PagedAttention 是什么？' } })
    fireEvent.click(screen.getByText('发送'))
  }

  it('says the answer is offline when the web backend failed', async () => {
    await ask({ status: 'failed', provider: 'recording', detail: 'RuntimeError' })

    // 联网失败不许静默降级成"看起来就像没开联网"。异常类型名也带出来，
    // 否则用户没法判断是自己断网了还是后端坏了。
    expect(await screen.findByText(/本次未能联网（RuntimeError）/)).toBeInTheDocument()
  })

  it('says no request went out when the switch is on but no backend exists', async () => {
    await ask({ status: 'unconfigured', provider: 'none' })

    expect(await screen.findByText(/还没有配置搜索后端/)).toBeInTheDocument()
  })

  it('stays quiet for the states that need no explanation', async () => {
    // off / ok / empty 都不该冒提示：前两个是正常，第三个是真实结果不是故障。
    await ask({ status: 'empty', provider: 'recording' })

    // 用来源卡片等消息落地（答案正文里的 [S1] 会被渲染成引用元素，文本是断开的）。
    await screen.findByText('PagedAttention 原论文')
    expect(screen.queryByText(/本次未能联网/)).not.toBeInTheDocument()
    expect(screen.queryByText(/还没有配置搜索后端/)).not.toBeInTheDocument()
  })

  it('marks the web source as the third layer only once a second layer appears', async () => {
    // 只有网络一层时不显示层名（P1 的规矩：多一层标签是视觉噪音）。真的与笔记
    // 混在一起时才需要提醒"这条不是来自你的笔记"。
    await ask({ status: 'ok', provider: 'recording' }, [NOTE_SOURCE, WEB_SOURCE])

    expect(await screen.findByText('网络')).toBeInTheDocument()
    expect(screen.getByText('笔记')).toBeInTheDocument()
    expect(screen.getByText('PagedAttention 原论文')).toBeInTheDocument()
  })

  it('still says the answer was offline when the conversation is reopened', async () => {
    // 这条不经过任何流式事件：只把服务端返回的一条历史回答渲染出来。联网状态
    // 过去只活在事件里，重开一次会话提示就没了——`failed` 于是变成"看起来今天
    // 没什么可网的"。现在它随消息从库里读回，翻旧的也该看得见。
    mocks.detail.mockImplementation(async () => ({
      id: 'c1', title: '一个会话', created_at: '2026-01-01', updated_at: '2026-01-01',
      messages: [{
        id: 'a-old', conversation_id: 'c1', role: 'assistant', content: '答案 [S1]。',
        status: 'complete', provider: null, model: null, index_version: null,
        error_code: null, reply_to_message_id: null,
        sources: [{ ...WEB_SOURCE, label: 'S1' }],
        web_state: { status: 'failed', provider: 'recording', detail: 'TimeoutError' },
      }],
    }))

    render(<ChatPage />)

    expect(await screen.findByText(/本次未能联网（TimeoutError）/)).toBeInTheDocument()
    // 不能拿流式事件凑——这条回答根本没有流。
    expect(mocks.stream).not.toHaveBeenCalled()
  })

  it('warns as soon as the answer lands, without waiting for the refetch', async () => {
    // 答完之后组件还要在 350ms 处重取一次会话。提示不能依赖那一次：重取慢、重取
    // 失败、用户抢在那之前就滑到这条回答——"本次未能联网"都该已经在屏幕上。
    // 这里让重取永远不返回，把时序钉死。
    const web = { status: 'failed', provider: 'recording', detail: 'RuntimeError' }
    mocks.stream.mockImplementation(
      async (_id: string, _body: unknown, onEvent: (event: unknown) => void) => {
        onEvent({ type: 'retrieval', message_id: 'a-new', sources: [], web })
        onEvent({ type: 'final', message_id: 'a-new', content: '答案。', status: 'complete', sources: [] })
      })
    mocks.detail.mockImplementationOnce(async (id: string) => ({
      id, title: '新会话', created_at: '', updated_at: '', messages: [] }))
    mocks.detail.mockImplementation(() => new Promise(() => {}))

    render(<ChatPage />)
    const box = await screen.findByPlaceholderText('询问你的资料；Shift + Enter 换行')
    fireEvent.change(box, { target: { value: 'PagedAttention 是什么？' } })
    fireEvent.click(screen.getByText('发送'))

    expect(await screen.findByText(/本次未能联网（RuntimeError）/)).toBeInTheDocument()
  })

  it('does not invent a web verdict for answers that have no record', async () => {
    // `web_state: null` 是"没有记录"（v10 之前的回答，或被守卫拦下的回答），
    // 不是"当时没联网"。界面一句话都不该说：说出来就是替历史编结论。
    mocks.detail.mockImplementation(async () => ({
      id: 'c1', title: '一个会话', created_at: '2026-01-01', updated_at: '2026-01-01',
      messages: [{
        id: 'a-old', conversation_id: 'c1', role: 'assistant', content: '答案 [S1]。',
        status: 'complete', provider: null, model: null, index_version: null,
        error_code: null, reply_to_message_id: null,
        sources: [{ ...NOTE_SOURCE, label: 'S1' }], web_state: null,
      }],
    }))

    render(<ChatPage />)
    await screen.findByText('推理.md')

    expect(screen.queryByText(/本次未能联网/)).not.toBeInTheDocument()
    expect(screen.queryByText(/还没有配置搜索后端/)).not.toBeInTheDocument()
  })
})
