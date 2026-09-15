import { useEffect, useState } from 'react'
import { api, type SourceContent, type SourceRef } from './api'
import { locatorLabel, locatorLines } from './lib/locator'
import { PdfViewer } from './components/PdfViewer'
import { HitMeta } from './components/HitMeta'
import { SkeletonLines } from './components/Placeholders'

export function SourcePanel({ hit, close }: { hit: SourceRef; close: () => void }) {
  const [source, setSource] = useState<SourceContent | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    setSource(null); setError('')
    // PDF 直接渲染原文件，不需要抽取出来的全文。
    if (hit.media_type === 'pdf') return
    void api.source(hit.document_id, hit.version_id).then(setSource).catch(e => setError(e.message))
  }, [hit])
  const lines = source?.text.split(/\r?\n/) ?? []
  const selected = locatorLines(hit.locator)
  return <aside className="source-panel">
    <header><div><span className="eyebrow">SOURCE SNAPSHOT</span><h2>{hit.title}</h2><p>{hit.heading_path}</p></div>
      <button className="icon-button" onClick={close} aria-label="关闭来源">×</button></header>
    <div className="source-location">{locatorLabel(hit.locator)} · 历史快照</div>
    <HitMeta hit={hit} />
    {hit.media_type === 'pdf' ? <PdfViewer key={hit.chunk_id} hit={hit} /> : error ?
      <p className="panel-error">{error}</p> : !source ?
      <SkeletonLines count={8} className="panel-skeleton" /> : <div className="markdown-source">{lines.map((line, index) => {
        const number = index + 1
        const on = selected !== null && number >= selected[0] && number <= selected[1]
        return <div key={number} className={on ? 'source-line selected' : 'source-line'}>
          <span>{number}</span><code>{line || ' '}</code></div>
      })}</div>}
  </aside>
}