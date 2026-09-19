import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { LibraryPage } from './LibraryPage'

// 这一组测试的重点不是 FolderPicker 自己（那有单独的测试），而是
// LibraryPage 有没有真的把它挂上去。曾经出过的真实缺陷：组件写好了、
// 状态也接了，JSX 里却漏了挂载，于是"选择"按钮点了没反应。
const mocks = vi.hoisted(() => ({
  folders: vi.fn(),
  libraries: vi.fn(),
  documents: vi.fn(),
  removeDocuments: vi.fn(),
  removeLibrary: vi.fn(),
}))

vi.mock('./api', () => ({
  api: {
    libraries: () => mocks.libraries(),
    documents: () => mocks.documents(),
    jobs: async () => ({ jobs: [] }),
    resources: async () => ({ examples: [] }),
    folders: (path?: string) => mocks.folders(path),
    connectFolder: vi.fn(),
    refreshLibrary: vi.fn(),
    upload: vi.fn(),
    search: vi.fn(),
    removeDocument: vi.fn(),
    removeDocuments: (ids: string[]) => mocks.removeDocuments(ids),
    removeLibrary: (id: string) => mocks.removeLibrary(id),
    retryDocument: vi.fn(),
    importBundledExample: vi.fn(),
  },
}))

const home = {
  path: 'C:\\Users\\zjm', name: 'zjm', parent: 'C:\\Users',
  entries: [{ name: 'Desktop', path: 'C:\\Users\\zjm\\Desktop' },
            { name: 'Documents', path: 'C:\\Users\\zjm\\Documents' }],
  truncated: false, roots: [{ name: 'C:', path: 'C:\\' }],
}

const desktop = {
  path: 'C:\\Users\\zjm\\Desktop', name: 'Desktop', parent: 'C:\\Users\\zjm',
  entries: [{ name: '项目', path: 'C:\\Users\\zjm\\Desktop\\项目' }],
  truncated: false, roots: [{ name: 'C:', path: 'C:\\' }],
}

const setup = () => render(<LibraryPage setupReload={async () => {}} />)

const document = (over: Record<string, unknown>) => ({
  id: 'doc-a', library_id: 'lib-up', relative_path: 'RAG.md', display_name: 'RAG.md',
  media_type: 'markdown', status: 'ready', error: null, current_version_id: 'ver-1',
  updated_at: '2026-01-01', library_name: '上传', ...over,
})

const folderLibrary = {
  id: 'lib-folder', name: '我的笔记本', kind: 'folder', document_count: 1, ready_count: 1,
}

beforeEach(() => {
  mocks.folders.mockReset()
  mocks.folders.mockImplementation(async (path?: string) =>
    path === desktop.path ? desktop : home)
  mocks.libraries.mockResolvedValue({ libraries: [] })
  mocks.documents.mockResolvedValue({ documents: [] })
  mocks.removeDocuments.mockResolvedValue({ deleted: 0, skipped: [] })
  mocks.removeLibrary.mockResolvedValue({ library_id: 'lib-folder', name: '我的笔记本', deleted: 0, skipped: [] })
})

describe('LibraryPage 的文件夹选择入口', () => {
  it('点击「选择」会打开应用内文件夹选择器', async () => {
    setup()
    // 按钮连上之后才可点，测试从等待按钮开始，避免误判成"没反应"。
    fireEvent.click(await screen.findByRole('button', { name: '选择' }))

    expect(await screen.findByRole('dialog', { name: '选择资料文件夹' })).toBeInTheDocument()
  })

  it('选择器里能逐级进入并把路径回填到输入框', async () => {
    setup()
    fireEvent.click(await screen.findByRole('button', { name: '选择' }))
    await screen.findByRole('dialog', { name: '选择资料文件夹' })

    fireEvent.click(await screen.findByRole('button', { name: /Desktop/ }))
    await waitFor(() => expect(mocks.folders).toHaveBeenCalledWith('C:\\Users\\zjm\\Desktop'))

    fireEvent.click(await screen.findByRole('button', { name: '选中此文件夹' }))

    await waitFor(() =>
      expect(screen.getByPlaceholderText('选择或粘贴文件夹路径')).toHaveValue('C:\\Users\\zjm\\Desktop'))
    expect(screen.queryByRole('dialog', { name: '选择资料文件夹' })).not.toBeInTheDocument()
  })

  it('取消后输入框保持原样、选择器关闭', async () => {
    setup()
    const input = await screen.findByPlaceholderText('选择或粘贴文件夹路径')
    fireEvent.change(input, { target: { value: 'C:\\已填的路径' } })

    fireEvent.click(screen.getByRole('button', { name: '选择' }))
    await screen.findByRole('dialog', { name: '选择资料文件夹' })
    fireEvent.click(screen.getByRole('button', { name: '取消' }))

    await waitFor(() =>
      expect(screen.queryByRole('dialog', { name: '选择资料文件夹' })).not.toBeInTheDocument())
    expect(screen.getByPlaceholderText('选择或粘贴文件夹路径')).toHaveValue('C:\\已填的路径')
  })
})

describe('LibraryPage 的批量移除', () => {
  it('确认框把文件夹来源的「暂时移除」说清楚，确认后才发请求', async () => {
    mocks.libraries.mockResolvedValue({ libraries: [folderLibrary] })
    mocks.documents.mockResolvedValue({ documents: [
      document({}),
      document({ id: 'doc-b', library_id: 'lib-folder', display_name: '笔记.md',
        relative_path: '笔记.md', library_name: '我的笔记本' }),
    ] })
    mocks.removeDocuments.mockResolvedValue({ deleted: 1, skipped: [] })
    setup()

    fireEvent.click(await screen.findByRole('button', { name: '批量选择' }))
    // 选择模式里的行 checkbox：第一条是上传资料，第二条来自文件夹
    const boxes = await screen.findAllByRole('checkbox')
    fireEvent.click(boxes[1])

    expect(await screen.findByText('已选 1 项')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '删除' }))
    expect(await screen.findByText('移除 1 篇资料？')).toBeInTheDocument()
    expect(screen.getByText(/只是暂时移除——刷新那个文件夹时它们会被重新导入/)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: '删除' }))
    await waitFor(() => expect(mocks.removeDocuments).toHaveBeenCalledWith(['doc-b']))
    expect(await screen.findByText('已删除 1 项')).toBeInTheDocument()
  })

  it('勾选上传来源的资料时，确认框不提「暂时移除」这句', async () => {
    mocks.documents.mockResolvedValue({ documents: [document({})] })
    setup()

    fireEvent.click(await screen.findByRole('button', { name: '批量选择' }))
    fireEvent.click((await screen.findAllByRole('checkbox'))[0])
    expect(await screen.findByText('已选 1 项')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '删除' }))

    expect(await screen.findByText('移除 1 篇资料？')).toBeInTheDocument()
    expect(screen.queryByText(/暂时移除/)).not.toBeInTheDocument()
  })

  it('取消选择模式不需要删除任何东西', async () => {
    mocks.documents.mockResolvedValue({ documents: [document({})] })
    setup()

    fireEvent.click(await screen.findByRole('button', { name: '批量选择' }))
    fireEvent.click((await screen.findAllByRole('checkbox'))[0])
    fireEvent.click(screen.getByRole('button', { name: '退出选择' }))

    expect(screen.queryByText('移除 1 篇资料？')).not.toBeInTheDocument()
    expect(mocks.removeDocuments).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: '批量选择' })).toBeInTheDocument()
  })
})

describe('LibraryPage 的断开归类', () => {
  it('断开前确认并把后果说清，确认后调接口并报告移除篇数', async () => {
    mocks.libraries.mockResolvedValue({ libraries: [folderLibrary] })
    mocks.removeLibrary.mockResolvedValue(
      { library_id: 'lib-folder', name: '我的笔记本', deleted: 1, skipped: [] })
    setup()

    fireEvent.click(await screen.findByRole('button', { name: '断开' }))

    expect(await screen.findByText('断开「我的笔记本」？')).toBeInTheDocument()
    // 文案纪律：断开不是"永久删除"（软删口径，行都留着），原文件必须说不动。
    expect(screen.getByText(/1 篇资料将从工作台移除，不再可搜索/)).toBeInTheDocument()
    expect(screen.getByText(/原文件夹与文件不会被删除；重新连接同一文件夹即可再次导入/)).toBeInTheDocument()
    expect(screen.queryByText(/永久删除/)).not.toBeInTheDocument()

    fireEvent.click(screen.getByTestId('library-disconnect-confirm'))
    await waitFor(() => expect(mocks.removeLibrary).toHaveBeenCalledWith('lib-folder'))
    expect(await screen.findByText('已断开，移除 1 篇资料')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: '完成' }))
    expect(await screen.findByText(/已断开「我的笔记本」，移除 1 篇资料/)).toBeInTheDocument()
    expect(screen.queryByTestId('library-disconnect-confirm')).not.toBeInTheDocument()
  })

  it('「再想想」收起确认条，不发任何请求', async () => {
    mocks.libraries.mockResolvedValue({ libraries: [folderLibrary] })
    setup()

    fireEvent.click(await screen.findByRole('button', { name: '断开' }))
    fireEvent.click(await screen.findByRole('button', { name: '再想想' }))

    expect(screen.queryByTestId('library-disconnect-confirm')).not.toBeInTheDocument()
    expect(mocks.removeLibrary).not.toHaveBeenCalled()
  })
})
