import { useCallback, useEffect, useRef, useState } from 'react'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import { api, type SearchHit } from '../api'
import { SkeletonLines } from './Placeholders'

const HIT_CLASS = 'pdf-hit'
// 占位页的高/宽。真实比例在文档加载后按第一页取，占位高度跟着它走：页一画
// 出来尺寸不变，排在前面的页就不会把目标页顶走。这个值只是拿到之前的后备。
const FALLBACK_ASPECT = 1.414

function compact(text: string) {
  return text.replace(/\s+/g, '')
}

/** 命中的文字块下标。
 *
 * pdfjs 把整页文字切成一个个绝对定位的 span，跨 span 的子串没法只高亮
 * 其中一段，所以这里按整块高亮。预览文字跨页或与提取结果有出入时逐级
 * 缩短再去匹配，匹配不上就干脆不高亮——不猜。
 */
function hitItems(items: string[], needle: string): number[] {
  const page = compact(items.join(''))
  const flat = compact(needle)
  for (const length of [flat.length, 120, 60, 30]) {
    const probe = flat.slice(0, length)
    if (probe.length < 4) break
    const at = page.indexOf(probe)
    if (at < 0) continue
    const end = at + probe.length
    const hits: number[] = []
    let cursor = 0
    items.forEach((item, index) => {
      const next = cursor + compact(item).length
      if (next > cursor && cursor < end && next > at) hits.push(index)
      cursor = next
    })
    return hits
  }
  return []
}

function PdfCanvas({ pdf, pageNumber, width, aspect, needle, root, onRendered }: {
  pdf: PDFDocumentProxy
  pageNumber: number
  width: number
  aspect: number
  needle: string
  root: HTMLElement | null
  onRendered: (pageNumber: number) => void
}) {
  const host = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const overlay = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  const [drawn, setDrawn] = useState(false)
  const [error, setError] = useState('')

  // 上千页的 PDF 一次性渲染会卡死，滚到附近才画。root 必须是那个真正滚动的
  // 容器（来源面板），用默认的视口当 root 的话，被面板滚出去的部分会被祖先
  // 裁剪成空矩形，rootMargin 再大也撑不回来，靠后的目标页就永远不渲染。
  useEffect(() => {
    const node = host.current
    if (!node || visible) return
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) setVisible(true)
    }, { root, rootMargin: '900px 0px' })
    observer.observe(node)
    return () => observer.disconnect()
  }, [visible, root])

  useEffect(() => {
    if (!visible || !width) return
    let cancelled = false
    let task: { cancel: () => void } | null = null
    let layer: { cancel: () => void } | null = null
    const draw = async () => {
      const pdfjs = await import('pdfjs-dist')
      const page = await pdf.getPage(pageNumber)
      const unscaled = page.getViewport({ scale: 1 })
      // 按面板宽度自适应，再把后备缓冲按设备像素比放大：只写 CSS 尺寸的话，
      // 高分屏会把低分辨率位图拉伸到物理像素上，字就糊了。
      const viewport = page.getViewport({ scale: Math.max(width - 34, 240) / unscaled.width })
      const target = canvas.current
      const textHost = overlay.current
      if (!target || !textHost || cancelled) return
      const ratio = window.devicePixelRatio || 1
      target.width = Math.round(viewport.width * ratio)
      target.height = Math.round(viewport.height * ratio)
      target.style.width = `${Math.round(viewport.width)}px`
      target.style.height = `${Math.round(viewport.height)}px`
      const rendering = page.render({
        canvas: target, viewport,
        transform: ratio === 1 ? undefined : [ratio, 0, 0, ratio, 0, 0],
      })
      task = rendering
      textHost.replaceChildren()
      // 文字层按百分比定位、用 --total-scale-factor 缩放字号，这个变量
      // pdfjs 不会自己设，必须由调用方给成当前的 viewport 缩放。
      textHost.style.setProperty('--total-scale-factor', String(viewport.scale))
      const textLayer = new pdfjs.TextLayer({
        textContentSource: await page.getTextContent(), container: textHost, viewport,
      })
      layer = textLayer
      await textLayer.render()
      if (cancelled) return
      for (const index of hitItems(textLayer.textContentItemsStr, needle)) {
        textLayer.textDivs[index]?.classList.add(HIT_CLASS)
      }
      await rendering.promise
      if (cancelled) return
      setDrawn(true)
      onRendered(pageNumber)
    }
    void draw().catch(reason => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : '这一页渲染失败')
    })
    return () => { cancelled = true; task?.cancel(); layer?.cancel() }
  }, [visible, width, pdf, pageNumber, needle, onRendered])

  return <div className="pdf-page" ref={host} data-page={pageNumber}
    style={drawn ? undefined : { width: width - 34, minHeight: (width - 34) * aspect }}>
    <canvas ref={canvas} className="pdf-canvas" />
    <div ref={overlay} className="pdf-text-layer" />
    {error && <p className="panel-error">{error}</p>}
  </div>
}

export function PdfViewer({ hit }: { hit: SearchHit }) {
  const scroller = useRef<HTMLDivElement>(null)
  const pending = useRef<number | null>(hit.locator.kind === 'pdf' ? hit.locator.page : 1)
  const [root, setRoot] = useState<HTMLDivElement | null>(null)
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null)
  const [aspect, setAspect] = useState(FALLBACK_ASPECT)
  const [width, setWidth] = useState(0)
  const [error, setError] = useState('')
  const needle = hit.preview

  // observer 的 root 要等节点挂上去才知道，所以额外存一份 state；ref 继续给
  // scrollTo 用，免得它的身份随状态变化、连带所有页面重画一次。
  const attach = useCallback((node: HTMLDivElement | null) => {
    scroller.current = node
    setRoot(node)
  }, [])

  useEffect(() => {
    let cancelled = false
    setPdf(null); setError('')
    const load = async () => {
      const pdfjs = await import('pdfjs-dist')
      pdfjs.GlobalWorkerOptions.workerSrc = new URL(
        'pdfjs-dist/build/pdf.worker.min.mjs', import.meta.url).toString()
      const document = await pdfjs.getDocument(
        { url: api.sourceFileUrl(hit.document_id, hit.version_id) }).promise
      const base = (await document.getPage(1)).getViewport({ scale: 1 })
      if (!cancelled) {
        setAspect(base.height / base.width)
        setPdf(document)
      }
    }
    void load().catch(reason => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : 'PDF 加载失败')
    })
    return () => { cancelled = true }
  }, [hit.document_id, hit.version_id])

  // 文档就绪前这里是加载提示，没有 .pdf-viewer 可量；所以要等 root 挂上来再量。
  useEffect(() => {
    if (!root) return
    const measure = () => setWidth(root.clientWidth)
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(root)
    return () => observer.disconnect()
  }, [root])

  // 目标页可能排在几十页之后，滚定位要等它自己的高度落地，否则量不准。
  // 来源面板的标题栏是 sticky 的，直接 scrollIntoView 会把目标页顶到它后面，
  // 所以按标题栏实测高度留出这段空白。
  const scrollTo = useCallback((pageNumber: number) => {
    if (pending.current !== pageNumber) return
    pending.current = null
    const node = scroller.current?.querySelector<HTMLElement>(`[data-page="${pageNumber}"]`)
    const panel = scroller.current?.closest<HTMLElement>('.source-panel')
    if (!node || !panel) return
    const header = panel.querySelector('header')
    const covered = header ? header.getBoundingClientRect().bottom - panel.getBoundingClientRect().top : 0
    node.style.scrollMarginTop = `${Math.round(covered) + 12}px`
    node.scrollIntoView({ block: 'start' })
  }, [])

  if (error) return <p className="panel-error">{error}</p>
  if (!pdf) return <SkeletonLines count={10} className="panel-skeleton" />
  return <div className="pdf-viewer" ref={attach}>
    {Array.from({ length: pdf.numPages }, (_, index) => index + 1).map(pageNumber =>
      <PdfCanvas key={pageNumber} pdf={pdf} pageNumber={pageNumber} width={width}
        aspect={aspect} needle={needle} root={root} onRendered={scrollTo} />)}
  </div>
}