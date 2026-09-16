import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { FolderPicker } from './FolderPicker'

// 用 vi.hoisted：vi.mock 会被提升到文件顶部执行，直接引用下面的 const 会撞上
// 暂时性死区。
const mocks = vi.hoisted(() => ({ folders: vi.fn() }))
vi.mock('../api', () => ({ api: { folders: (path?: string) => mocks.folders(path) } }))

function listing(path: string, names: string[], parent: string | null = 'C:\\Users') {
  return {
    path,
    name: path.split('\\').filter(Boolean).pop() || path,
    parent,
    truncated: false,
    roots: [{ name: 'C:', path: 'C:\\' }],
    entries: names.map(name => ({ name, path: `${path}\\${name}` })),
  }
}

const setup = (initial?: string) => {
  const onPick = vi.fn()
  const onCancel = vi.fn()
  render(<FolderPicker initial={initial} onCancel={onCancel} onPick={onPick} />)
  return { onPick, onCancel }
}

beforeEach(() => { mocks.folders.mockReset() })

describe('FolderPicker', () => {
  it('lists the subdirectories of the starting folder', async () => {
    mocks.folders.mockResolvedValue(listing('C:\\Users\\zjm', ['Desktop', 'Documents']))
    setup('C:\\Users\\zjm')

    expect(await screen.findByText('Desktop')).toBeInTheDocument()
    expect(screen.getByText('Documents')).toBeInTheDocument()
    expect(mocks.folders).toHaveBeenCalledWith('C:\\Users\\zjm')
  })

  it('descends into a subdirectory when it is clicked', async () => {
    mocks.folders.mockImplementation(async (path?: string) => path === 'C:\\Users\\zjm\\Desktop'
      ? listing('C:\\Users\\zjm\\Desktop', ['项目'])
      : listing('C:\\Users\\zjm', ['Desktop']))
    setup('C:\\Users\\zjm')

    fireEvent.click(await screen.findByText('Desktop'))

    expect(await screen.findByText('项目')).toBeInTheDocument()
    expect(mocks.folders).toHaveBeenLastCalledWith('C:\\Users\\zjm\\Desktop')
  })

  it('hands back the folder currently on screen', async () => {
    mocks.folders.mockResolvedValue(listing('C:\\Users\\zjm', ['Desktop']))
    const { onPick } = setup('C:\\Users\\zjm')

    fireEvent.click(await screen.findByText('选中此文件夹'))

    expect(onPick).toHaveBeenCalledWith('C:\\Users\\zjm')
  })

  it('falls back to the home directory when the typed path cannot be read', async () => {
    // 输入框里的路径可能是手打或粘贴来的。选择器不该一打开就是一屏错误。
    mocks.folders.mockImplementation(async (path?: string) => {
      if (path) throw new Error('这个位置不存在或读不到，请换一个文件夹。')
      return listing('C:\\Users\\zjm', ['Desktop'])
    })
    setup('C:\\没有这个目录')

    expect(await screen.findByText('Desktop')).toBeInTheDocument()
    expect(mocks.folders).toHaveBeenLastCalledWith(undefined)
  })

  it('closes on Escape', async () => {
    mocks.folders.mockResolvedValue(listing('C:\\Users\\zjm', []))
    const { onCancel } = setup()

    await screen.findByText('这个文件夹下面没有子文件夹，可以直接选它。')
    fireEvent.keyDown(window, { key: 'Escape' })

    expect(onCancel).toHaveBeenCalled()
  })

  it('will not confirm a folder it could not read', async () => {
    mocks.folders.mockRejectedValue(new Error('没有权限读取这个文件夹，请换一个。'))
    setup('C:\\Windows\\System32\\config')

    expect(await screen.findByText('没有权限读取这个文件夹，请换一个。')).toBeInTheDocument()
    expect(screen.getByText('选中此文件夹')).toBeDisabled()
  })
})
