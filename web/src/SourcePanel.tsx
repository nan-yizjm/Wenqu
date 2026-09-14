import { useEffect, useRef, useState } from 'react'
import { api, type SearchHit, type SourceContent } from './api'

function PdfPage({ hit }: { hit: SearchHit }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let cancelled = false
    const render = async () => {
      try {
        const pdfjs = await import('pdfjs-dist')
        pdfjs.GlobalWorkerOptions.workerSrc = new URL(
          'pdfjs-dist/build/pdf.worker.min.mjs', import.meta.url).toString()
        const task = pdfjs.getDocument({ url: api.sourceFileUrl(hit.document_id, hit.version_id) })
        const pdf = await task.promise
        const pageNumber = hit.locator.kind === 'pdf' ? hit.locator.page : 1
        const page = await pdf.getPage(pageNumber)
        const viewport = page.getViewport({ scale: 1.35 })
        const target = canvas.current
        if (!target || cancelled) return
        target.width = viewport.width; target.height = viewport.height
        const context = target.getContext('2d')
        if (!context) throw new Error('浏览器无法创建 PDF 画布')
        await page.render({ canvas: target, canvasContext: context, viewport }).promise
      } catch (reason) { if (!cancelled) setError(reason instanceof Error ? reason.message : 'PDF 加载失败') }
    }
    void render()
    return () => { cancelled = true }
  }, [hit])
  return error ? <p className="panel-error">{error}</p> : <canvas className="pdf-canvas" ref={canvas} />
}

export function SourcePanel({ hit, close }: { hit: SearchHit; close: () => void }) {
  const [source, setSource] = useState<SourceContent | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    setSource(null); setError('')
    void api.source(hit.document_id, hit.version_id).then(setSource).catch(e => setError(e.message))
  }, [hit])
  const lines = source?.text.split(/\r?\n/) ?? []
  return <aside className="source-panel">
    <header><div><span className="eyebrow">SOURCE SNAPSHOT</span><h2>{hit.title}</h2><p>{hit.heading_path}</p></div>
      <button className="icon-button" onClick={close} aria-label="关闭来源">×</button></header>
    <div className="source-location">{hit.locator.kind === 'pdf' ? `第 ${hit.locator.page} 页` : `第 ${hit.locator.start_line}–${hit.locator.end_line} 行`} · 历史快照</div>
    {error ? <p className="panel-error">{error}</p> : hit.media_type === 'pdf' ? <PdfPage hit={hit} /> : !source ?
      <div className="panel-loading">正在打开来源…</div> : <div className="markdown-source">{lines.map((line, index) => {
        const number = index + 1
        const selected = hit.locator.kind === 'markdown' && number >= hit.locator.start_line && number <= hit.locator.end_line
        return <div key={number} className={selected ? 'source-line selected' : 'source-line'}>
          <span>{number}</span><code>{line || ' '}</code></div>
      })}</div>}
  </aside>
}
