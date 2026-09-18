import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { Favorite, FavoriteSummary, FavoritesView } from './api'
import { FavoritesPage } from './FavoritesPage'

// 收藏页此前只有手工路径（进详情后单条删除）。批量删除走的是左侧列表的
// 选择模式，这里的重点是：入口在、确认文案如实、请求真的带上了选中的 id。
const mocks = vi.hoisted(() => ({
  favorites: vi.fn(),
  favorite: vi.fn(),
  deleteFavorites: vi.fn(),
}))

vi.mock('./api', () => ({
  api: {
    favorites: (filters: unknown) => mocks.favorites(filters),
    favorite: (id: string) => mocks.favorite(id),
    deleteFavorites: (ids: string[]) => mocks.deleteFavorites(ids),
    deleteFavorite: vi.fn(),
    updateFavorite: vi.fn(),
    source: vi.fn(),
    collectionExportUrl: vi.fn(() => '/export'),
    favoriteExportUrl: vi.fn(() => '/export'),
  },
}))

const summary = (over: Partial<FavoriteSummary>): FavoriteSummary => ({
  id: 'fav-a', message_id: 'msg-a', title: 'RAG 在线流程', note: '', question: '什么是 RAG？',
  answer: '先检索再生成。', provider: null, model: null, index_version: null,
  generated_at: null, updated_at: '2026-01-01', source_count: 1, tags: [],
  libraries: [], feedback_kind: null, ...over,
})

const listed: FavoriteSummary[] = [
  summary({}),
  summary({ id: 'fav-b', message_id: 'msg-b', title: 'PagedAttention 要点' }),
]

const detail = (id: string): Favorite => ({
  ...summary({ id }), sources: [],
})

const view = (favorites: FavoriteSummary[]): FavoritesView => ({
  favorites, libraries: [], tags: [], total: favorites.length,
})

beforeEach(() => {
  for (const mock of Object.values(mocks)) mock.mockReset()
  mocks.favorites.mockResolvedValue(view(listed))
  mocks.favorite.mockImplementation(async (id: string) => detail(id))
  mocks.deleteFavorites.mockResolvedValue({ deleted: 1, skipped: [] })
})

describe('FavoritesPage 的批量删除', () => {
  it('选择模式里点列表项是切换选中，确认后按选中 id 发请求', async () => {
    render(<FavoritesPage />)

    fireEvent.click(await screen.findByRole('button', { name: '批量选择' }))
    fireEvent.click(await screen.findByRole('button', { name: /PagedAttention 要点/ }))
    expect(await screen.findByText('已选 1 项')).toBeInTheDocument()

    fireEvent.click(screen.getByTestId('batch-confirm'))
    expect(await screen.findByText('删除 1 条收藏？')).toBeInTheDocument()
    expect(screen.getByText('收藏是副本，删除后不影响原会话与原回答。')).toBeInTheDocument()

    fireEvent.click(screen.getByTestId('batch-confirm'))
    await waitFor(() => expect(mocks.deleteFavorites).toHaveBeenCalledWith(['fav-b']))
    expect(await screen.findByText('已删除 1 项')).toBeInTheDocument()
  })

  it('完成之后回到普通列表，选择状态清空', async () => {
    render(<FavoritesPage />)

    fireEvent.click(await screen.findByRole('button', { name: '批量选择' }))
    fireEvent.click(await screen.findByRole('button', { name: /PagedAttention 要点/ }))
    await screen.findByText('已选 1 项')
    fireEvent.click(screen.getByTestId('batch-confirm'))
    fireEvent.click(await screen.findByTestId('batch-confirm'))
    fireEvent.click(await screen.findByRole('button', { name: '完成' }))

    expect(await screen.findByRole('button', { name: '批量选择' })).toBeInTheDocument()
    expect(screen.queryByText(/已选/)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /PagedAttention 要点/ })).toBeInTheDocument()
  })

  it('skipped 的收藏单独列出原因', async () => {
    mocks.deleteFavorites.mockResolvedValue({ deleted: 0, skipped: [
      { id: 'fav-ghost', label: null, code: 'not_found', reason: '收藏不存在。' },
    ] })
    render(<FavoritesPage />)

    fireEvent.click(await screen.findByRole('button', { name: '批量选择' }))
    fireEvent.click(screen.getByLabelText('全选（2）'))
    fireEvent.click(screen.getByTestId('batch-confirm'))
    fireEvent.click(await screen.findByTestId('batch-confirm'))

    expect(await screen.findByText('已删除 0 项')).toBeInTheDocument()
    expect(screen.getByText(/fav-ghost：收藏不存在/)).toBeInTheDocument()
  })
})
