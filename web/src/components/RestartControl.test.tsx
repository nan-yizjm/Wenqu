import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { RestartControl } from './RestartControl'

const restart = vi.fn()
const health = vi.fn()

vi.mock('../api', () => ({ api: { restart: () => restart(), health: () => health() } }))

beforeEach(() => {
  restart.mockReset()
  health.mockReset()
  restart.mockResolvedValue({ status: 'restarting', message: '本地服务正在重启。' })
})

describe('RestartControl', () => {
  it('reloads once the instance reports a new start time, even without a visible gap', async () => {
    // 实测重启窗口只有一秒上下：轮询撞不撞上断连全看运气，所以判据是
    // **启动时刻变了**。这条用例里 health 从没失败过——旧判据会一直等到超时。
    health.mockResolvedValueOnce({ status: 'ready', started_at: 'A' })
    health.mockResolvedValue({ status: 'ready', started_at: 'B' })
    const onReload = vi.fn()
    render(<RestartControl onReload={onReload} timeoutMs={4000} />)

    fireEvent.click(screen.getByTestId('restart-button'))
    await waitFor(() => expect(restart).toHaveBeenCalledTimes(1))
    expect((screen.getByTestId('restart-button') as HTMLButtonElement).disabled).toBe(true)

    await waitFor(() => expect(onReload).toHaveBeenCalledTimes(1), { timeout: 4000 })
  })

  it('reports a service that went down and never came back', async () => {
    health.mockRejectedValue(new Error('连接拒绝'))
    const onReload = vi.fn()
    render(<RestartControl onReload={onReload} timeoutMs={250} />)

    fireEvent.click(screen.getByTestId('restart-button'))

    await waitFor(() => expect(screen.getByTestId('restart-failed')).toBeTruthy(), { timeout: 4000 })
    expect(screen.getByTestId('restart-note').textContent).toContain('一直没有回来')
    expect(onReload).not.toHaveBeenCalled()
  })

  it('reports a restart that changed nothing', async () => {
    // 服务一直活着、启动时刻也没变：重启请求没有生效，不能报成功。
    health.mockResolvedValue({ status: 'ready', started_at: 'A' })
    const onReload = vi.fn()
    render(<RestartControl onReload={onReload} timeoutMs={250} />)

    fireEvent.click(screen.getByTestId('restart-button'))

    await waitFor(() => expect(screen.getByTestId('restart-failed')).toBeTruthy(), { timeout: 4000 })
    expect(screen.getByTestId('restart-note').textContent).toContain('没有变化')
    expect(onReload).not.toHaveBeenCalled()
  })

  it('reports a refused restart instead of pretending', async () => {
    restart.mockRejectedValue(
      new Error('这个实例没有连接启动器，无法自行重启；请结束它的进程后重新打开。'))
    const onReload = vi.fn()
    render(<RestartControl onReload={onReload} />)

    fireEvent.click(screen.getByTestId('restart-button'))

    await waitFor(() => expect(screen.getByTestId('restart-note')).toBeTruthy())
    expect(screen.getByTestId('restart-note').textContent).toContain('无法自行重启')
    expect(onReload).not.toHaveBeenCalled()
  })
})
