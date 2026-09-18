import { useEffect, useState } from 'react'
import { api, type SourceContent, type SourceRef } from './api'
import { locatorLabel, locatorLines, mediaLabel } from './lib/locator'
import { PdfViewer } from './components/PdfViewer'
import { HitMeta } from './components/HitMeta'
import { SkeletonLines } from './components/Placeholders'

/** 非笔记层在面板上的名字。`note` 不走这里——它就是"原文快照"。 */
const LAYER_LABEL: Record<string, string> = { memory: '记忆', web: '网络' }

export function SourcePanel({ hit, close }: { hit: SourceRef; close: () => void }) {
  const [source, setSource] = useState<SourceContent | null>(null)
  const [error, setError] = useState('')
  // 老记录（v8 之前存的来源）没有 origin，它们本来就是笔记片段。
  const origin = hit.origin || 'note'
  const isNote = origin === 'note'
  useEffect(() => {
    setSource(null); setError('')
    // 只有笔记层的来源取得到原文快照。记忆与网络来源的 document_id 是占位串，
    // 照 document_id 去请求 /versions/{id}/source 只会拿到 404——它们的正文
    // 就是 preview（一条可引用的事实/摘要），不需要也不该再去取"原文"。
    if (!isNote) return
    // PDF 直接渲染原文件，不需要抽取出来的全文。
    if (hit.media_type === 'pdf') return
    void api.source(hit.document_id, hit.version_id).then(setSource).catch(e => setError(e.message))
  }, [hit, isNote])
  const lines = source?.text.split(/\r?\n/) ?? []
  const selected = locatorLines(hit.locator)
  return <aside className="source-panel">
    <header><div><span className="eyebrow">{isNote ? 'SOURCE SNAPSHOT' : 'SOURCE LAYER'}</span>
      <h2>{hit.title}</h2><p>{hit.heading_path}</p></div>
      <button className="icon-button" onClick={close} aria-label="关闭来源">×</button></header>
    <div className="source-location">{isNote
      ? `${locatorLabel(hit.locator)} · 历史快照`
      : `${LAYER_LABEL[origin] || origin} · ${mediaLabel(hit.media_type)} · ${locatorLabel(hit.locator)}`}</div>
    <HitMeta hit={hit} />
    {!isNote ? <div className="markdown-source">
      <div className="source-line"><span />
        <code>{hit.preview}</code></div>
      </div>
      : hit.media_type === 'pdf' ? <PdfViewer key={hit.chunk_id} hit={hit} /> : error ?
      <p className="panel-error">{error}</p> : !source ?
      <SkeletonLines count={8} className="panel-skeleton" /> : <div className="markdown-source">{lines.map((line, index) => {
        const number = index + 1
        const on = selected !== null && number >= selected[0] && number <= selected[1]
        return <div key={number} className={on ? 'source-line selected' : 'source-line'}>
          <span>{number}</span><code>{line || ' '}</code></div>
      })}</div>}
  </aside>
}
