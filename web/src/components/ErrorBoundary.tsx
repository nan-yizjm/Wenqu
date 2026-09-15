import { Component, type ErrorInfo, type ReactNode } from 'react'

type State = { error: Error | null }

// 渲染期抛错会让 React 卸载整棵树，用户看到的是一片空白。
// 这里留住一个能读懂的页面，并给出下一步动作。
export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('界面渲染失败', error, info.componentStack)
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    return <main className="fatal">
      <h1>这个页面没能显示出来</h1>
      <p>已保存的资料和会话都还在本机。刷新通常就能恢复；若反复出现，请下载脱敏诊断后反馈。</p>
      <pre className="error-detail">{error.message}</pre>
      <div className="empty-actions">
        <button className="primary" onClick={() => location.reload()}>刷新工作台</button>
        <a className="secondary" href="/api/v1/system/diagnostics/export">下载脱敏诊断</a>
      </div>
    </main>
  }
}