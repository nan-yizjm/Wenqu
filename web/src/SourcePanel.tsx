import { useEffect, useRef, useState } from 'react'
import { api, type SearchHit, type SourceContent } from './api'
import { locatorLabel, locatorLines } from './lib/locator'
import { SkeletonLines } from './components/Placeholders'

function PdfPage({ hit }: { hit: SearchHit }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let cancelled = false
    let task: { cancel: () => void } | null = null
    const render = async () => {
      try {
        const pdfjs = await import('pdfjs-dist')
        pdfjs.GlobalWorkerOptions.workerSrc = new URL(
          'pdfjs-dist/build/pdf.worker.min.mjs', import.meta.url).toString()
        const pdf = await pdfjs.getDocument({ url: api.sourceFileUrl(hit.document_id, hit.version_id) }).promise
        const pageNumber = hit.locator.kind === 'pdf' ? hit.locator.page : 1
        const page = await pdf.getPage(pageNumber)
        const target = canvas.current
        if (!target || cancelled) return
        // 先按面板宽度排版（CSS 像素），再把后备缓冲按设备像素比放大，
        // 并让 pdfjs 用同一个倍率绘制。只写 viewport 尺寸的话，高分屏会
        // 把整张低分辨率位图拉伸到物理像素上，字就糊了。
        const unscaled = page.getViewport({ scale: 1 })
        const available = Math.max((target.parentElement?.clientWidth ?? unscaled.width) - 30, 240)
        const viewport = page.getViewport({ scale: available / unscaled.width })
        const ratio = window.devicePixelRatio || 1
        target.width = Math.round(viewport.width * ratio)
        target.height = Math.round(viewport.height * ratio)
        const rendering = page.render({
          canvas: target, viewport,
          transform: ratio === 1 ? undefined : [ratio, 0, 0, ratio, 0, 0],
        })
        task = rendering
        await rendering.promise
      } catch (reason) { if (!cancelled) setError(reason instanceof Error ? reason.message : 'PDF 加载失败') }
    }
    void render()
    return () => { cancelled = true; task?.cancel() }
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
  const selected = locatorLines(hit.locator)
  return <aside className="source-panel">
    <header><div><span className="eyebrow">SOURCE SNAPSHOT</span><h2>{hit.title}</h2><p>{hit.heading_path}</p></div>
      <button className="icon-button" onClick={close} aria-label="关闭来源">×</button></header>
    <div className="source-location">{locatorLabel(hit.locator)} · 历史快照</div>
    {error ? <p className="panel-error">{error}</p> : hit.media_type === 'pdf' ? <PdfPage hit={hit} /> : !source ?
      <SkeletonLines count={8} className="panel-skeleton" /> : <div className="markdown-source">{lines.map((line, index) => {
        const number = index + 1
        const on = selected !== null && number >= selected[0] && number <= selected[1]
        return <div key={number} className={on ? 'source-line selected' : 'source-line'}>
          <span>{number}</span><code>{line || ' '}</code></div>
      })}</div>}
  </aside>
}
