import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { Conversation } from './api'
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
  },
}))

let current: Conversation[]

beforeEach(() => {
  mocks.list.mockReset(); mocks.remove.mockReset()
  mocks.create.mockReset(); mocks.detail.mockReset()
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
