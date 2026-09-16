import { useEffect, useRef, useState } from 'react'
import { api, type FolderListing } from '../api'
import { SkeletonLines } from './Placeholders'
import { crumbs } from '../lib/paths'

/** 应用内的文件夹选择器。
 *
 * 不用系统文件夹对话框：安装版是无控制台、无主窗口的进程，Windows 会把系统
 * 对话框创建在浏览器窗口**后面**（z 序实测确实如此），用户看不到就等于按钮
 * 没反应。改成后端列子目录、前端画选择器之后，行为只由 HTTP 决定，离线测试
 * 也能覆盖；顺带不再需要 powershell.exe。
 */
export function FolderPicker({ initial, onCancel, onPick }: {
  initial?: string
  onCancel: () => void
  onPick: (path: string) => void
}) {
  const [listing, setListing] = useState<FolderListing | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const dialog = useRef<HTMLDivElement>(null)

  const load = async (path?: string) => {
    setLoading(true)
    try {
      setListing(await api.folders(path)); setError('')
      return true
    } catch (e) {
      setError(e instanceof Error ? e.message : '无法读取这个文件夹')
      return false
    } finally { setLoading(false) }
  }
  useEffect(() => { void (async () => {
    // 输入框里可能是手打或粘贴来的路径，不一定真的存在。那就退回主目录，
    // 而不是让选择器一打开就是一屏错误。
    if (initial && await load(initial)) return
    await load()
  })() }, [])
  useEffect(() => { dialog.current?.focus() }, [])
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onCancel() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onCancel])

  const trail = listing ? crumbs(listing.path) : []
  const roots = listing?.roots || []
  return <div className="picker-backdrop" onClick={onCancel}>
    <div className="folder-picker" role="dialog" aria-modal="true" aria-label="选择资料文件夹"
      tabIndex={-1} ref={dialog} onClick={event => event.stopPropagation()}>
      <header>
        <div><span className="eyebrow">CHOOSE A FOLDER</span><h2>选择资料文件夹</h2></div>
        <button className="ghost" onClick={onCancel} aria-label="关闭">✕</button>
      </header>
      {roots.length > 1 && <div className="picker-roots">{roots.map(root => <button
        key={root.path} className={listing?.path.startsWith(root.path) ? 'active' : ''}
        onClick={() => void load(root.path)}>{root.name}</button>)}</div>}
      <div className="picker-crumbs">
        <button className="ghost" disabled={!listing?.parent}
          onClick={() => listing?.parent && void load(listing.parent)}>↑ 上一级</button>
        {trail.map((crumb, index) => <button key={crumb.path}
          className={index === trail.length - 1 ? 'active' : ''}
          onClick={() => void load(crumb.path)}>{crumb.name}</button>)}
      </div>
      <div className="picker-list">
        {error ? <p className="error">{error}</p>
          : loading && !listing ? <SkeletonLines count={5} />
          : !listing?.entries.length ? <p className="hint">这个文件夹下面没有子文件夹，可以直接选它。</p>
          : listing.entries.map(entry => <button key={entry.path} className="picker-entry"
              onClick={() => void load(entry.path)}><span aria-hidden="true">▸</span>{entry.name}</button>)}
        {listing?.truncated && <p className="hint">子文件夹太多，只列出了前 {listing.entries.length} 个。可以直接把完整路径粘贴到输入框。</p>}
      </div>
      <footer>
        <code title={listing?.path}>{listing?.path || '…'}</code>
        <div className="picker-actions">
          <button className="ghost" onClick={onCancel}>取消</button>
          <button className="primary" disabled={!listing || !!error}
            onClick={() => listing && onPick(listing.path)}>选中此文件夹</button>
        </div>
      </footer>
    </div>
  </div>
}
