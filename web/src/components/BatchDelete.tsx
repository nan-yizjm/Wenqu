import { useState } from 'react'
import type { BatchResult } from '../api'

/**
 * 多选状态。三页（资料库 / 收藏 / 产出）共用一份行为：
 * 选中顺序不保证——界面只关心"选了哪些"，不关心先点谁。
 */
export function useSelection() {
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const toggle = (id: string) => setSelected(current => {
    const next = new Set(current)
    if (next.has(id)) next.delete(id); else next.add(id)
    return next
  })
  const clear = () => setSelected(new Set())
  const setAll = (ids: string[]) => setSelected(new Set(ids))
  const isSelected = (id: string) => selected.has(id)
  return { selected, toggle, clear, setAll, isSelected }
}

export type BatchDeleteBarProps = {
  /** 当前选中的 id。 */
  ids: string[]
  /** 当前列表的全部 id，供"全选"使用。 */
  allIds: string[]
  /** 确认框标题，例如"移除 3 篇资料？"。 */
  heading: string
  /** 后果说明，一行一条。这里的话必须与后端行为一致——文案先于行为就是撒谎。 */
  lines: string[]
  confirmLabel?: string
  onToggleAll: () => void
  allSelected: boolean
  onCancel: () => void
  onConfirm: (ids: string[]) => Promise<BatchResult>
  /** 删除请求成功后调用（含部分 skipped）——父页面在这里刷新并退出选择模式。 */
  onDone: () => void
}

type Phase = 'idle' | 'confirming' | 'working' | 'result'

/**
 * 选择模式底部条：默认显示"已选几项"，点删除后进入页面内两段式确认（不用
 * `window.confirm`——无头验收和部分嵌入环境里它会卡死），完成后如实展示结果，
 * 包括没删掉的那几条：只报"已删除 N 项"会把一次不完整的删除说成成功的。
 */
export function BatchDeleteBar({ ids, allIds, heading, lines, confirmLabel = '删除',
  onToggleAll, allSelected, onCancel, onConfirm, onDone }: BatchDeleteBarProps) {
  const [phase, setPhase] = useState<Phase>('idle')
  const [result, setResult] = useState<BatchResult | null>(null)
  const [error, setError] = useState('')

  const run = async () => {
    setPhase('working')
    try {
      const outcome = await onConfirm(ids)
      setResult(outcome); setPhase('result')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '删除失败'); setPhase('confirming')
    }
  }

  if (phase === 'result' && result) {
    return <div className="batch-bar batch-result" data-testid="batch-result" aria-live="polite">
      <div>
        <strong>已删除 {result.deleted} 项{result.files_removed
          ? ` · 一并清理导出文件 ${result.files_removed} 个` : ''}</strong>
        {result.skipped.length > 0 && <>
          <span>以下 {result.skipped.length} 项未删除：</span>
          <ul>{result.skipped.map(item =>
            <li key={item.id}>{item.label ?? item.id}：{item.reason}</li>)}</ul>
        </>}
      </div>
      <button className="ghost" onClick={onDone}>完成</button>
    </div>
  }

  if (phase === 'confirming' || phase === 'working') {
    return <div className="batch-bar batch-confirm" aria-live="polite">
      <div>
        <strong>{heading}</strong>
        {lines.map(line => <p key={line}>{line}</p>)}
        {error && <p className="batch-error">{error}</p>}
      </div>
      <div className="batch-actions">
        <button className="ghost" disabled={phase === 'working'}
          onClick={() => { setError(''); setPhase('idle') }}>再想想</button>
        {/* data-testid：详情区/列表行的单条「删除」按钮与本条同名，定位时靠它区分。 */}
        <button className="danger" data-testid="batch-confirm" disabled={phase === 'working'}
          onClick={() => void run()}>
          {phase === 'working' ? '正在删除…' : confirmLabel}</button>
      </div>
    </div>
  }

  return <div className="batch-bar">
    <label className="batch-select-all">
      <input type="checkbox" checked={allSelected} onChange={onToggleAll} />
      {allSelected ? '取消全选' : '全选'}（{allIds.length}）
    </label>
    <span className="batch-count">已选 {ids.length} 项</span>
    <div className="batch-actions">
      <button className="ghost" onClick={onCancel}>退出选择</button>
      <button className="danger" data-testid="batch-confirm" disabled={ids.length === 0}
        onClick={() => setPhase('confirming')}>
        {confirmLabel}</button>
    </div>
  </div>
}
