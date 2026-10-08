import { useEffect, useRef, useState } from 'react'
import { api } from '../api'

/**
 * 恢复完成后的「立即重启」。
 *
 * 为什么需要它：恢复备份与迁移恢复之后，应用处在"待重启"状态，用户能做的下一件事
 * 就是重启——此前只提示"请退出后重新打开"，得自己回桌面找图标。这里点一下把这件事
 * 做完，并且**等服务真的回来**才刷新页面。
 *
 * 判据是**启动时刻变了**（`/api/v1/health` 的 `started_at`），不是"撞见一次断连"：
 * 实测重启窗口只有一秒上下（Windows 上 `os.execv` 是"起新进程 + 退旧进程"，停机与
 * 新实例监听之间非常短），轮询撞不撞得上全看运气。启动时刻变了就一定是新实例，
 * 这与窗口长短无关。老版本服务不返回这个字段时，退回"断过又回来"的旧判据。
 */
export function RestartControl({ onReload, timeoutMs = 90000 }: {
  onReload?: () => void
  /** 等待上限。测试用短值；正常重启要重建 app（索引与模型状态），给足余量。 */
  timeoutMs?: number
}) {
  const [phase, setPhase] = useState<'idle' | 'waiting' | 'failed'>('idle')
  const [note, setNote] = useState('')
  const timer = useRef<number | undefined>(undefined)
  useEffect(() => () => window.clearTimeout(timer.current), [])

  const restart = async () => {
    setPhase('waiting')
    setNote('正在重启本地服务…')
    // 先记下当前实例的启动时刻：重启后它必然不同。
    let previous: string | null = null
    try {
      previous = (await api.health()).started_at ?? null
    } catch {
      previous = null
    }
    try {
      await api.restart()
    } catch (reason) {
      setPhase('failed')
      setNote(reason instanceof Error ? reason.message : '没能让本地服务重启。')
      return
    }
    const deadline = Date.now() + timeoutMs
    let sawDown = false
    const poll = async () => {
      try {
        const reported = (await api.health()).started_at ?? null
        const replaced = previous !== null && reported !== null && reported !== previous
        if (replaced || (reported === null && sawDown)) {
          ;(onReload ?? (() => window.location.reload()))()
          return
        }
        if (previous === null && reported !== null) previous = reported
      } catch {
        sawDown = true
      }
      if (Date.now() > deadline) {
        setPhase('failed')
        setNote(sawDown
          ? '服务已经停下，但一直没有回来。请重新打开 Wenqu。'
          : '重启请求已发出，但服务没有变化。请重新打开 Wenqu。')
        return
      }
      timer.current = window.setTimeout(() => void poll(), 400)
    }
    timer.current = window.setTimeout(() => void poll(), 300)
  }

  if (phase === 'failed') {
    return <div className="restart-control" data-testid="restart-failed">
      <p className="error" data-testid="restart-note">{note}</p>
    </div>
  }
  return <div className="restart-control">
    <button className="primary" disabled={phase === 'waiting'}
      data-testid="restart-button"
      onClick={() => void restart()}>{phase === 'waiting' ? '正在重启…' : '立即重启'}</button>
    {note && <p className="hint" data-testid="restart-note">{note}</p>}
  </div>
}
