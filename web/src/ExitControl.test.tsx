import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ExitControl, SignedOff } from './ExitControl'

const shutdown = vi.fn()

vi.mock('./api', () => ({
  api: { shutdown: () => shutdown() },
}))

beforeEach(() => {
  shutdown.mockReset()
  shutdown.mockResolvedValue({ status: 'shutting_down', message: '本地服务正在退出。' })
})

describe('ExitControl', () => {
  it('says up front that closing the tab is not the same as quitting', () => {
    render(<ExitControl onExit={vi.fn()} />)

    expect(screen.getByTestId('exit-open')).toBeTruthy()
    // 这条文案是这个入口存在的理由：多数人以为关掉标签就等于退出。
    expect(screen.getByText(/关掉浏览器标签不会停止本地服务/)).toBeTruthy()
  })

  it('asks for confirmation once before quitting, and does not quit on the first click', () => {
    const onExit = vi.fn()
    render(<ExitControl onExit={onExit} />)

    fireEvent.click(screen.getByTestId('exit-open'))

    expect(screen.getByTestId('exit-question')).toBeTruthy()
    expect(screen.getByText(/资料、索引和会话都留在原处/)).toBeTruthy()
    // 第一下只是展开确认：不许已经发出退出请求。
    expect(shutdown).not.toHaveBeenCalled()
    expect(onExit).not.toHaveBeenCalled()
  })

  it('cancelling puts everything back and sends nothing', () => {
    render(<ExitControl onExit={vi.fn()} />)

    fireEvent.click(screen.getByTestId('exit-open'))
    fireEvent.click(screen.getByTestId('exit-cancel-button'))

    expect(screen.queryByTestId('exit-question')).toBeNull()
    expect(screen.getByTestId('exit-open')).toBeTruthy()
    expect(shutdown).not.toHaveBeenCalled()
  })

  it('quitting asks the service to stop and only then reports the way out', async () => {
    const onExit = vi.fn()
    render(<ExitControl onExit={onExit} />)

    fireEvent.click(screen.getByTestId('exit-open'))
    fireEvent.click(screen.getByTestId('exit-confirm-button'))

    await waitFor(() => expect(onExit).toHaveBeenCalledTimes(1))
    expect(shutdown).toHaveBeenCalledTimes(1)
  })

  it('does not claim to have quit when the service refused', async () => {
    shutdown.mockRejectedValue(new Error('这个实例没有连接启动器，无法自行退出；请结束它的进程。'))
    const onExit = vi.fn()
    render(<ExitControl onExit={onExit} />)

    fireEvent.click(screen.getByTestId('exit-open'))
    fireEvent.click(screen.getByTestId('exit-confirm-button'))

    // 停不了就不能显示"已退出"——那会变成一句假话。
    await waitFor(() => expect(screen.getByTestId('exit-failure')).toBeTruthy())
    expect(screen.getByTestId('exit-failure').textContent).toContain('无法自行退出')
    expect(onExit).not.toHaveBeenCalled()
  })

  it('can be tried again after a failure', async () => {
    shutdown.mockRejectedValueOnce(new Error('没能让本地服务退出。'))
    const onExit = vi.fn()
    render(<ExitControl onExit={onExit} />)

    fireEvent.click(screen.getByTestId('exit-open'))
    fireEvent.click(screen.getByTestId('exit-confirm-button'))
    await waitFor(() => expect(screen.getByTestId('exit-failure')).toBeTruthy())

    fireEvent.click(screen.getByTestId('exit-confirm-button'))
    await waitFor(() => expect(onExit).toHaveBeenCalledTimes(1))
  })
})

describe('SignedOff', () => {
  it('states that the data survived and how to come back', () => {
    render(<SignedOff />)

    expect(screen.getByTestId('signed-off')).toBeTruthy()
    expect(screen.getByText(/都还在原处/)).toBeTruthy()
    expect(screen.getByText(/不会删除任何东西/)).toBeTruthy()
    expect(screen.getByText(/双击桌面上的工作台图标/)).toBeTruthy()
  })
})
