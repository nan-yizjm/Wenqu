import { useCallback, useEffect, useRef, useState } from 'react'
import {
  api, type Artifact, type ArtifactKind, type ArtifactStatus, type ArtifactSummary,
  type BacklinkReport, type InfographicExport, type MessageSource, type Mindmap,
  type MindmapNode,
} from './api'
import { AnswerMarkdown } from './components/AnswerMarkdown'
import { SourcePanel } from './SourcePanel'
import { BatchDeleteBar, useSelection } from './components/BatchDelete'

/**
 * 产出页。
 *
 * 这一页的重点不是"能生成东西"，而是**把生成结果的可信度摆出来**：指南给回链
 * 命中率、缺来源的句子逐条列出；思维导图明说自己是构造出来的、不给分数。界面
 * 上因此刻意没有"质量评分"这类含糊的东西——只有能解释来源的数字。
 */
const KINDS: { id: ArtifactKind; label: string; hint: string }[] = [
  { id: 'guide', label: '学习指南', hint: '由模型撰写，每一句都要带 [S1] 形式的回链；完成后给出回链命中率。' },
  { id: 'mindmap', label: '思维导图', hint: '不经过模型：节点直接取自片段的标题层级，所以同一份资料每次结果都一样。' },
]

const STATUS_LABEL: Record<ArtifactStatus, string> = {
  running: '生成中', complete: '已完成', stopped: '已停止', failed: '失败',
}

function percent(rate: number | null) {
  return rate === null ? null : `${Math.round(rate * 100)}%`
}

/** 回链报告：本页唯一的"分数"，所以它要能解释自己是怎么算出来的。 */
function BacklinkCard({ report }: { report: BacklinkReport }) {
  const rate = percent(report.hit_rate)
  const tone = report.hit_rate === null ? 'amber'
    : report.hit_rate >= 0.9 ? 'green' : 'amber'
  return <section className="card studio-report" data-testid="backlink-report">
    <h3>回链命中率</h3>
    <div className="status-line">
      <span className={`dot ${tone}`} data-testid="backlink-rate">
        {rate ?? '无可统计的句子'}
      </span>
      {/* 没有断言行时是"无法统计"而不是 0% 也不是 100%——空产出不该被显示成好或坏。 */}
      {report.hit_rate === null
        ? <span>这篇产出里没有可统计的陈述句（正文为空，或只有标题）。</span>
        : <span>{report.assertions} 句陈述中，{report.with_source} 句带了来源。</span>}
    </div>
    <p className="hint">
      按正文现算，不存分数：存下来的分数会和正文各自演化，而正文才是唯一的事实来源。
      标题、代码块和表格分隔行不算陈述句——把它们算进去会把命中率无理由拉低。
    </p>
    {report.invalid_labels.length > 0 && <p className="error">
      引用了不存在的编号：{report.invalid_labels.join('、')}。这比"没写来源"更值得注意——
      它看起来像有来源。
    </p>}
    {report.missing_count > 0 && <details className="studio-missing">
      <summary>{report.missing_count} 句没有来源（列出前 {report.missing.length} 句）</summary>
      <ul>{report.missing.map(item => <li key={item.line}>
        <code>第 {item.line} 行</code><span>{item.text}</span></li>)}</ul>
    </details>}
  </section>
}

/** 思维导图。节点不带编号时不会出现——没有片段支撑的节点压根不会被建出来。 */
function MindmapBranch({ node, byLabel, open }: {
  node: MindmapNode
  byLabel: Map<string, MessageSource>
  open: (source: MessageSource) => void
}) {
  return <li className="mindmap-node" data-testid="mindmap-node">
    <div className="mindmap-line">
      <span className="mindmap-label">{node.label}</span>
      {node.sources.length > 0 && <span className="mindmap-chips">
        {node.sources.map(label => {
          const hit = byLabel.get(label)
          return hit
            ? <button key={label} className="citation" title={`${hit.title} · ${hit.heading_path}`}
                onClick={() => open(hit)}>{label}</button>
            : <span key={label} className="citation-invalid">{label}</span>
        })}
      </span>}
    </div>
    {node.children.length > 0 && <ul>
      {node.children.map(child => <MindmapBranch key={child.id} node={child}
        byLabel={byLabel} open={open} />)}
    </ul>}
  </li>
}

function MindmapView({ mindmap, sources, open }: {
  mindmap: Mindmap
  sources: MessageSource[]
  open: (source: MessageSource) => void
}) {
  const byLabel = new Map(sources.map(item => [item.label, item]))
  return <section className="card studio-mindmap" data-testid="mindmap-view">
    <h3>{mindmap.tree.label}</h3>
    <ul className="mindmap-tree">
      {mindmap.tree.children.map(child => <MindmapBranch key={child.id} node={child}
        byLabel={byLabel} open={open} />)}
    </ul>
    {/* 这里刻意不报百分比：节点是拿片段建的，覆盖率必然接近 100%，和指南的命中率
        并排显示会让人以为两者可比。 */}
    <p className="hint" data-testid="mindmap-coverage">{mindmap.coverage_note}</p>
    <div className="status-row"><span>节点</span><code>{mindmap.node_count}</code></div>
    <div className="status-row"><span>挂到节点上的片段</span><code>{mindmap.linked_chunks}</code></div>
    <details className="studio-mermaid">
      <summary>复制 Mermaid 到 Obsidian</summary>
      <pre><code>{mindmap.mermaid}</code></pre>
      <p className="hint">本工作台不渲染 Mermaid，只提供可复制的文本。</p>
    </details>
  </section>
}

/**
 * 信息图导出卡。
 *
 * 这一格要解释清楚"图片产出在这个产品里是什么"：它**不是第三种产出类型**，而是把
 * 眼前这份产出的来源画成一张 PNG——所以它不调模型、不需要新表。图里没有一个字是
 * 模型写的，每个编号都指向一份资料片段。
 *
 * 浏览器不可用（`degraded`）时**不显示成"出错"**：那份 HTML 已经导出了，用户可以用
 * 自己的浏览器打开它、或者自己打印成图片。一个能自救的功能不该被显示成故障。
 */
function InfographicCard({ artifactId }: { artifactId: string }) {
  const [record, setRecord] = useState<InfographicExport | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  // 换一份产出就重新读一次导出记录。父组件用 `key` 强制重挂，所以这里不会留下上一份
  // 产出的图——"上一次的状态跟了过来"正是这个页面踩过的坑（见 clearLive 的注释）。
  useEffect(() => {
    let alive = true
    setRecord(null); setError('')
    void (async () => {
      try {
        const found = await api.infographicExport(artifactId)
        if (alive) setRecord(found.export)
      } catch { /* 读不到就当作没导出过；真要导出时会再报一次错 */ }
    })()
    return () => { alive = false }
  }, [artifactId])

  const run = async () => {
    setBusy(true); setError('')
    try { setRecord(await api.exportInfographic(artifactId)) }
    catch (reason) { setError(reason instanceof Error ? reason.message : '导出失败') }
    finally { setBusy(false) }
  }

  const render = record?.render
  return <section className="card studio-infographic" data-testid="infographic-card">
    <h3>信息图（PNG）</h3>
    <p className="hint">
      把这份产出的来源画成一张图：来源分布、章节结构、回链命中率。图里没有一个字来自
      模型——每个编号都指向一份资料片段。渲染由本机 Edge / Chrome 无头完成，不联网。
    </p>
    <div className="studio-actions">
      <button className="primary" disabled={busy} onClick={() => void run()}
        data-testid="infographic-export">
        {busy ? '正在渲染…' : record ? '重新导出' : '导出信息图'}
      </button>
      {error && <span className="error">{error}</span>}
    </div>

    {record && render?.status === 'complete' && <>
      <img className="studio-infographic-image" data-testid="infographic-image"
        src={api.infographicImage(artifactId, render.created_at)}
        alt={`${record.title} 的来源信息图`} />
      <div className="status-row"><span>渲染耗时</span>
        <code data-testid="infographic-milliseconds">{render.milliseconds ?? '—'} 毫秒</code></div>
      <div className="status-row"><span>图片尺寸</span>
        <code>{render.pixels ? `${render.pixels.width}×${render.pixels.height}` : '—'} 像素</code></div>
      <div className="status-row"><span>渲染器</span><code>{render.browser ?? '—'}</code></div>
      <div className="status-row"><span>文件</span>
        <code title={record.files.png_path}>{record.files.png}</code></div>
      <div className="studio-actions">
        <a className="studio-download" data-testid="infographic-download"
          href={api.infographicDownload(artifactId)}>下载 PNG</a>
        <span className="hint">已导出到 {record.files.png_path}</span>
      </div>
      {render.pixels_match === false && <p className="error">
        浏览器没有按请求的尺寸出图：版面 {record.layout.width}×{record.layout.height}
        （{record.layout.scale} 倍 → 应为 {record.layout.pixels.width}×
        {record.layout.pixels.height} 像素）。图能看，但排版可能与预期不同。
      </p>}
    </>}

    {record && render?.status === 'unavailable' &&
      <div className="studio-infographic-degraded" data-testid="infographic-degraded">
        {/* 用 warn 色而不是 danger：这不是功能故障，用户手里有 HTML 这条退路。
            只要把"为什么"和"怎么办"都说清楚，它就不该看起来像报错。 */}
        <p className="studio-degraded-line">没能渲染成图片：{render.message}</p>
        <p className="hint">
          这份产出的 HTML 已经导出，可以用浏览器打开它、或自己打印成图片。也可以把环境变量
          OBSIDIAN_RAG_BROWSER 指向 Edge / Chrome 的可执行文件后重试。
        </p>
        <div className="status-row"><span>HTML 文件</span>
          <code title={record.files.html_path}>{record.files.html_path}</code></div>
      </div>}
  </section>
}

export function StudioPage() {
  const [artifacts, setArtifacts] = useState<ArtifactSummary[]>([])
  const [detail, setDetail] = useState<Artifact | null>(null)
  const [topic, setTopic] = useState('')
  const [kind, setKind] = useState<ArtifactKind>('guide')
  const [running, setRunning] = useState(false)
  const [liveId, setLiveId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [liveSources, setLiveSources] = useState<MessageSource[]>([])
  const [liveReport, setLiveReport] = useState<BacklinkReport | null>(null)
  const [liveMindmap, setLiveMindmap] = useState<Mindmap | null>(null)
  const [liveStatus, setLiveStatus] = useState<ArtifactStatus>('running')
  const [liveNotice, setLiveNotice] = useState('')
  const [liveEvidenceNote, setLiveEvidenceNote] = useState('')
  const [message, setMessage] = useState('')
  const [source, setSource] = useState<MessageSource | null>(null)
  const [selecting, setSelecting] = useState(false)
  const selection = useSelection()
  const abort = useRef<AbortController | null>(null)

  const load = useCallback(async () => {
    try { setArtifacts((await api.artifacts()).artifacts) }
    catch (error) { setMessage(error instanceof Error ? error.message : '无法读取产出列表') }
  }, [])

  useEffect(() => { void load() }, [load])
  // 离开页面时中止在跑的生成：否则流会继续消费，用户却已经看不到它了。
  useEffect(() => () => abort.current?.abort(), [])

  /**
   * 清掉"这一次生成"的现场。
   *
   * 必须在打开历史产出时也调用：生成失败后会留下一条失败说明，而只要它还挂着，
   * `showingLive` 就一直为真、右侧永远显示那次的残留，**点历史里任何一份都打不开**
   * ——删除按钮也就永远点不到。这是无头验收真实抓到的缺陷。
   */
  const clearLive = useCallback(() => {
    setDraft(''); setLiveSources([]); setLiveReport(null); setLiveMindmap(null)
    setLiveNotice(''); setLiveStatus('running'); setLiveId(null)
    setLiveEvidenceNote('')
  }, [])

  const open = async (id: string) => {
    clearLive(); setMessage('')
    try { setDetail(await api.artifact(id)) }
    catch (error) { setMessage(error instanceof Error ? error.message : '无法打开这份产出') }
  }

  const generate = async () => {
    if (!topic.trim()) { setMessage('请先填一个主题。'); return }
    clearLive(); setMessage(''); setDetail(null)
    setRunning(true)
    const controller = new AbortController()
    abort.current = controller
    try {
      await api.streamArtifact(topic.trim(), kind, event => {
        if (event.artifact_id) setLiveId(event.artifact_id)
        if (event.type === 'retrieval') {
          setLiveSources(event.sources ?? [])
          setLiveEvidenceNote(event.evidence_note ?? '')
        }
        else if (event.type === 'token') setDraft(previous => previous + (event.text ?? ''))
        else if (event.type === 'final' || event.type === 'stopped') {
          // 正文以服务端回传为准：final 是渲染后的逐句正文，停止时带回能渲染的部分
          //（渲染不出就回原始 JSON）。token 是原始 JSON 碎片，只是过程产物，不能
          // 直接当正文留着——否则停止后界面残留半截 JSON。
          if (event.content !== undefined) setDraft(event.content)
          setLiveStatus(event.status ?? 'complete')
          if (event.sources) setLiveSources(event.sources)
          if (event.backlink) setLiveReport(event.backlink)
          if (event.mindmap) setLiveMindmap(event.mindmap)
          if (event.message) setLiveNotice(event.message)
        } else if (event.type === 'error') {
          if (event.content !== undefined) setDraft(event.content)
          setLiveStatus('failed')
          setLiveNotice(event.message || '生成失败。')
        }
      }, controller.signal)
      setLiveStatus(previous => (previous === 'running' ? 'complete' : previous))
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') setLiveStatus('stopped')
      else { setLiveStatus('failed'); setLiveNotice(error instanceof Error ? error.message : '生成失败') }
    } finally {
      setRunning(false); abort.current = null
      await load()
    }
  }

  const stop = async () => {
    // 先让服务端停（它才知道流跑在哪），再断开本地读取。反过来的话服务端可能还在写。
    if (liveId) { try { await api.stopArtifact(liveId) } catch { /* 已结束就忽略 */ } }
    abort.current?.abort()
    setRunning(false)
  }

  const remove = async (id: string) => {
    if (!window.confirm('删除这份产出？来源记录会一并删除，但资料本身不受影响。')) return
    try {
      await api.deleteArtifact(id)
      if (detail?.id === id) setDetail(null)
      await load()
    } catch (error) { setMessage(error instanceof Error ? error.message : '删除失败') }
  }

  // 生成中有流式草稿，点历史又看详情——两者同时只能显示一个。
  const showingLive = running || draft !== '' || liveMindmap !== null || liveNotice !== ''
  const shownSources = showingLive ? liveSources : detail?.sources ?? []

  return <div className={`studio-page${source ? ' with-source' : ''}`}>
    <aside className="studio-rail">
      <header><span className="eyebrow">STUDIO</span><h1>产出</h1>
        <p>把资料整理成能追溯来源的指南或导图。</p>
        {artifacts.length > 0 && !selecting && <button className="ghost" onClick={() => setSelecting(true)}>批量选择</button>}
      </header>
      {artifacts.length === 0
        ? <p className="studio-empty">还没有产出。右侧填一个主题开始。</p>
        : <div className="studio-list">{artifacts.map(item => {
            const row = <button className={detail?.id === item.id ? 'active' : ''}
              onClick={() => selecting ? selection.toggle(item.id) : void open(item.id)}>
              <span className="studio-kind">{KINDS.find(entry => entry.id === item.kind)?.label ?? item.kind}</span>
              <strong>{item.title}</strong>
              <small><span className={`status-chip ${item.status}`}>{STATUS_LABEL[item.status]}</span>
                {item.created_at.slice(0, 16).replace('T', ' ')}</small>
            </button>
            return selecting
              ? <div key={item.id} className={`studio-item${selection.isSelected(item.id) ? ' batch-selected' : ''}`}>
                  <input type="checkbox" checked={selection.isSelected(item.id)} onChange={() => selection.toggle(item.id)} />
                  {row}
                </div>
              : <div key={item.id} className="studio-item">{row}</div>
          })}</div>}
      {selecting && artifacts.length > 0 && <BatchDeleteBar
        ids={[...selection.selected]}
        allIds={artifacts.map(item => item.id)}
        allSelected={selection.selected.size > 0 && selection.selected.size === artifacts.length}
        onToggleAll={() => selection.selected.size === artifacts.length
          ? selection.clear() : selection.setAll(artifacts.map(item => item.id))}
        heading={`删除 ${selection.selected.size} 份产出？`}
        lines={['随产出导出的图片也会一并删除。', '正在生成的产出会被跳过，不会被删除。']}
        onCancel={() => { setSelecting(false); selection.clear() }}
        onConfirm={async ids => {
          const result = await api.deleteArtifacts(ids)
          const removed = new Set(ids)
          setDetail(current => current && removed.has(current.id) ? null : current)
          await load()
          return result
        }}
        onDone={() => { setSelecting(false); selection.clear() }} />}
    </aside>

    <main className="studio-main">
      <header className="page-heading">
        <div><span className="eyebrow">FROM YOUR NOTES</span><h1>生成</h1>
          <p>产出只使用资料库里的片段；生成的每一句都会带着来源编号，或者被列出来。</p></div>
      </header>

      <section className="card studio-form">
        <label>主题<input value={topic} maxLength={200} placeholder="例如：KV Cache 的显存优化"
          onChange={event => setTopic(event.target.value)}
          onKeyDown={event => { if (event.key === 'Enter' && !running) void generate() }} /></label>
        <div className="segmented">{KINDS.map(entry => <button key={entry.id}
          className={kind === entry.id ? 'active' : ''} disabled={running}
          onClick={() => setKind(entry.id)}>{entry.label}</button>)}</div>
        <p className="hint">{KINDS.find(entry => entry.id === kind)?.hint}</p>
        <div className="studio-actions">
          <button className="primary" disabled={running} onClick={() => void generate()}>
            {running ? '正在生成…' : '开始生成'}</button>
          {running && <button className="ghost" onClick={() => void stop()}>停止</button>}
          {message && <span className="hint">{message}</span>}
        </div>
      </section>

      {liveStatus === 'failed' && liveNotice &&
        <p className="error studio-notice">{liveNotice}</p>}
      {liveStatus === 'stopped' &&
        <p className="hint studio-notice">已停止。已经写出的部分保留在这份产出里。</p>}

      {showingLive && <section className="studio-output">
        {liveMindmap
          ? <MindmapView mindmap={liveMindmap} sources={liveSources} open={setSource} />
          : <section className="card studio-draft">
              <h3>{KINDS.find(entry => entry.id === kind)?.label}</h3>
              {draft
                ? (liveStatus === 'running'
                  // 逐句 JSON 的过程产物是原始碎片，渲染出来就是闪动的半截 JSON——
                  // 过程只报收到的字数，正文等渲染完成再显示。
                  ? <p className="hint" data-testid="live-progress">
                      正在逐句整理资料…（已收到 {draft.length} 字）</p>
                  : <AnswerMarkdown content={draft} sources={liveSources} open={setSource} />)
                : <p className="hint">正在等待模型输出…</p>}
            </section>}
        {liveEvidenceNote && <p className="hint" data-testid="live-evidence-note">{liveEvidenceNote}</p>}
        {liveReport && <BacklinkCard report={liveReport} />}
        {/* 刚生成完就能导出：产出在流里 `final` 之前已经落库为完整状态，
            不必先去左边列表点一次。`key` 保证换一份产出就重挂。 */}
        {liveId && liveStatus === 'complete' &&
          <InfographicCard key={liveId} artifactId={liveId} />}
      </section>}

      {!showingLive && detail && <section className="studio-output">
        <header className="studio-detail-head">
          <div><span className="studio-kind">{KINDS.find(entry => entry.id === detail.kind)?.label}</span>
            <h2>{detail.title}</h2>
            <p className="hint">主题：{detail.topic} · {detail.created_at.slice(0, 16).replace('T', ' ')}
              {detail.index_version ? ` · 索引 ${detail.index_version.slice(0, 8)}` : ''}</p></div>
          <button className="ghost" onClick={() => void remove(detail.id)}>删除</button>
        </header>
        {detail.evidence_note && <p className="hint" data-testid="evidence-note">{detail.evidence_note}</p>}
        {detail.status === 'failed' && <p className="error">
          {detail.error_code === 'no_evidence'
            ? '当时资料里没有找到与这个主题相关的片段，所以没有产出内容。'
            : `生成失败（${detail.error_code || '未知原因'}）。`}</p>}
        {detail.kind === 'mindmap' && detail.mindmap
          ? <MindmapView mindmap={detail.mindmap} sources={detail.sources} open={setSource} />
          : <section className="card studio-draft">
              <AnswerMarkdown content={detail.content} sources={detail.sources} open={setSource} />
            </section>}
        {detail.backlink && <BacklinkCard report={detail.backlink} />}
        <section className="card">
          <h3>来源（{detail.sources.length}）</h3>
          <div className="source-chips">{detail.sources.map(item => <button key={item.label}
            onClick={() => setSource(item)}><b>{item.label}</b>{item.title} · {item.heading_path}</button>)}</div>
        </section>
        <InfographicCard key={detail.id} artifactId={detail.id} />
      </section>}

      {!showingLive && !detail && <p className="table-empty">
        左侧选一份产出查看，或在上面填主题生成一份新的。</p>}
    </main>

    {source && <SourcePanel hit={source} close={() => setSource(null)} />}
  </div>
}
