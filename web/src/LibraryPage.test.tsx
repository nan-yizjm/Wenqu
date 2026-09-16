import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { LibraryPage } from './LibraryPage'

// 这一组测试的重点不是 FolderPicker 自己（那有单独的测试），而是
// LibraryPage 有没有真的把它挂上去。曾经出过的真实缺陷：组件写好了、
// 状态也接了，JSX 里却漏了挂载，于是"选择"按钮点了没反应。
const mocks = vi.hoisted(() => ({ folders: vi.fn() }))

vi.mock('./api', () => ({
  api: {
    libraries: async () => ({ libraries: [] }),
    documents: async () => ({ documents: [] }),
    jobs: async () => ({ jobs: [] }),
    resources: async () => ({ examples: [] }),
    folders: (path?: string) => mocks.folders(path),
    connectFolder: vi.fn(),
    refreshLibrary: vi.fn(),
    upload: vi.fn(),
    search: vi.fn(),
    removeDocument: vi.fn(),
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

beforeEach(() => {
  mocks.folders.mockReset()
  mocks.folders.mockImplementation(async (path?: string) =>
    path === desktop.path ? desktop : home)
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
