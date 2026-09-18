import { useState } from 'react'
import { api } from './api'

/**
 * 退出入口。
 *
 * 为什么需要一个专门的东西：产品是个常驻的本地服务（`console=False`、没有托盘），
 * 而"关掉浏览器标签"**不会**让它退出——此前正常使用下没有任何退出入口，唯一的办法
 * 是去任务管理器结束进程。这个组件的存在就是为了补上那个缺口，所以它的文案必须
 * 先把这个事实说清楚，而不是只放一个"退出"。
 *
 * 两段式：第一下点开确认，第二下才真的退出。**不用 `window.confirm`**——它在无头
 * 浏览器里会阻塞页面求值，验收脚本点不动它（这个坑在验收脚本里已经记过一次）。
 *
 * 退出失败时**不显示"已退出"**：后端在没有接启动器时会回 503，此时界面要如实说
 * 停不了，而不是让用户以为服务没了。
 */
export function ExitControl({ onExit, compact = false }: { onExit: () => void; compact?: boolean }) {
  const [confirming, setConfirming] = useState(false)
  const [busy, setBusy] = useState(false)
  const [failure, setFailure] = useState('')

  const quit = async () => {
    setBusy(true); setFailure('')
    try {
      await api.shutdown()
      onExit()
    } catch (reason) {
      setFailure(reason instanceof Error ? reason.message : '没能让本地服务退出。')
      setBusy(false)
    }
  }

  if (confirming) {
    return <div className={`exit-control confirming${compact ? ' compact' : ''}`} data-testid="exit-confirm">
      <p className="exit-question" data-testid="exit-question">
        确认退出？本地服务会停止，这个标签页可以关掉；资料、索引和会话都留在原处。
      </p>
      <div className="exit-actions">
        <button className="secondary" disabled={busy} onClick={() => void quit()}
          data-testid="exit-confirm-button">{busy ? '正在退出…' : '确认退出'}</button>
        <button className="ghost" disabled={busy} onClick={() => setConfirming(false)}
          data-testid="exit-cancel-button">取消</button>
      </div>
      {failure && <p className="error" data-testid="exit-failure">{failure}</p>}
    </div>
  }

  return <div className={`exit-control${compact ? ' compact' : ''}`}>
    {/* "工作台" 用一个 span 包着，窄屏折成图标条时由 CSS 收掉，只剩"退出"两个字——
        和 nav 的 `.nav-label` 同一个做法，`title` 里保留完整说明。 */}
    <button className="ghost exit-open" onClick={() => setConfirming(true)}
      title="停止本地服务；资料不会被删除" data-testid="exit-open">
      退出<span className="exit-label">工作台</span></button>
    {/* 这句话是这个入口存在的理由：多数人以为关掉标签就等于退出。 */}
    {!compact && <small className="exit-note">关掉浏览器标签不会停止本地服务。</small>}
  </div>
}

/** 退出之后的收尾页。
 *
 * 服务已经停了，这一页**不能再发任何请求**——所以它不显示实时状态，只交代两件事：
 * 东西都还在，以及下次怎么回来。继续留在原来的界面上只会让每个请求都失败，
 * 看起来像产品坏了。
 */
export function SignedOff() {
  return <main className="signed-off" data-testid="signed-off">
    <div className="setup-card">
      <span className="step">已退出</span>
      <h1>工作台已经停止</h1>
      <p>本地服务已关闭，浏览器标签可以关掉了。<strong>资料、索引、会话和收藏都还在原处</strong>，
        这次退出不会删除任何东西。</p>
      <p className="hint">下次双击桌面上的工作台图标就会回到这里。数据保存在你自己的用户目录里，
        卸载也不会动它。</p>
    </div>
  </main>
}
