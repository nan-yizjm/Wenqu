import { useEffect, useRef, useState } from 'react'
import { api, type BundledResource, type DocumentItem, type ImportJob, type Library, type SearchHit } from './api'
import { locatorLabel, mediaLabel } from './lib/locator'
import { SourcePanel } from './SourcePanel'
import { EmptyState, SkeletonLines } from './components/Placeholders'

export function LibraryPage({ setupReload }: { setupReload: () => Promise<void> }) {
  const [libraries, setLibraries] = useState<Library[]>([])
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [jobs, setJobs] = useState<ImportJob[]>([])
  const [examples, setExamples] = useState<BundledResource[]>([])
  const [folder, setFolder] = useState('')
  const [query, setQuery] = useState('')
  const [hits, setHits] = useState<SearchHit[]>([])
  const [selected, setSelected] = useState<SearchHit | null>(null)
  const [message, setMessage] = useState('')
  const [loaded, setLoaded] = useState(false)
  const uploadInput = useRef<HTMLInputElement>(null)
  const load = async () => {
    const [libraryData, documentData, jobData, resourceData] = await Promise.all(
      [api.libraries(), api.documents(), api.jobs(), api.resources()])
    setLibraries(libraryData.libraries); setDocuments(documentData.documents); setJobs(jobData.jobs)
    setExamples(resourceData.examples)
    await setupReload()
  }
  useEffect(() => { void load().catch(e => setMessage(e instanceof Error ? e.message : '无法读取资料库')).finally(() => setLoaded(true)) }, [])
  useEffect(() => {
    if (!jobs.some(job => ['pending', 'running'].includes(job.status))) return
    const timer = window.setInterval(() => void load(), 900)
    return () => window.clearInterval(timer)
  }, [jobs])
  const chooseFolder = async () => {
    try { const result = await api.pickFolder(); if (result.path) setFolder(result.path) }
    catch (e) { setMessage(e instanceof Error ? e.message : '无法打开文件夹选择器') }
  }
  const connect = async () => {
    if (!folder.trim()) return
    setMessage('正在扫描 Markdown 文件…')
    try { await api.connectFolder(folder.trim()); setFolder(''); await load() }
    catch (e) { setMessage(e instanceof Error ? e.message : '连接失败') }
  }
  const upload = async (file?: File) => {
    if (!file) return
    setMessage(`正在导入 ${file.name}…`)
    try { await api.upload(file); await load() }
    catch (e) { setMessage(e instanceof Error ? e.message : '上传失败') }
    if (uploadInput.current) uploadInput.current.value = ''
  }
  const search = async () => {
    if (!query.trim()) return
    try { const result = await api.search(query.trim()); setHits(result.results); setMessage(result.results.length ? '' : '没有找到匹配的资料片段。') }
    catch (e) { setMessage(e instanceof Error ? e.message : '搜索失败') }
  }
  const importExample = async (name: string) => {
    setMessage(`正在导入 ${name}…`)
    try { await api.importBundledExample(name); await load(); setMessage(`${name} 已导入，可以直接搜索。`) }
    catch (e) { setMessage(e instanceof Error ? e.message : '导入示例失败') }
  }
  const remove = async (document: DocumentItem) => {
    if (!window.confirm(`从工作台移除“${document.display_name}”？原文件不会被删除。`)) return
    await api.removeDocument(document.id); await load(); setHits(hits.filter(hit => hit.document_id !== document.id))
  }
  const retry = async (document: DocumentItem) => {
    try { await api.retryDocument(document.id); setMessage(`正在重试 ${document.display_name}…`); await load() }
    catch (e) { setMessage(e instanceof Error ? e.message : '重试失败') }
  }
  const activeJob = jobs.find(job => ['pending', 'running'].includes(job.status))
  return <div className={selected ? 'library-layout with-source' : 'library-layout'}>
    <div className="library-main"><header className="library-heading"><div><span className="eyebrow">YOUR KNOWLEDGE</span><h1>资料库</h1>
      <p>连接只读的资料文件夹，或把单独的 Markdown / PDF / Jupyter Notebook 保存到工作台。</p></div>
      <button className="primary" onClick={() => uploadInput.current?.click()}>上传文件</button>
      <input ref={uploadInput} className="hidden" type="file" accept=".md,.markdown,.pdf,.ipynb" onChange={event => void upload(event.target.files?.[0])} /></header>
      <section className="add-folder"><div><strong>连接资料文件夹</strong><span>后续刷新只读取原目录，不修改原文件。</span></div>
        <div className="folder-row"><input value={folder} onChange={e => setFolder(e.target.value)} placeholder="选择或粘贴文件夹路径" />
          <button className="ghost" onClick={chooseFolder}>选择</button><button className="primary" onClick={connect}>连接</button></div></section>
      {activeJob && <div className="job-banner"><span className="spinner small" /><div><strong>正在处理资料</strong>
        <span>{activeJob.completed}/{activeJob.total || '…'}，失败 {activeJob.failed}</span></div></div>}
      {message && <div className="inline-message">{message}</div>}
      {libraries.some(library => library.kind === 'folder') && <div className="library-sources">{libraries.filter(library => library.kind === 'folder').map(library =>
        <div key={library.id}><span><strong>{library.name}</strong><small>{library.ready_count || 0}/{library.document_count} 个文件可搜索</small></span>
          <button className="ghost" onClick={async () => { await api.refreshLibrary(library.id); setMessage(`正在刷新 ${library.name}…`); await load() }}>刷新</button></div>)}</div>}
      <section className="search-box"><input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') void search() }} placeholder="搜索你的全部资料，例如：PagedAttention 解决什么问题？" />
        <button className="primary" onClick={search}>搜索</button></section>
      {hits.length > 0 && <section className="search-results"><div className="section-title"><h2>搜索结果</h2><span>{hits.length} 个片段</span></div>
        {hits.map(hit => <button className="result-card" key={hit.chunk_id} onClick={() => setSelected(hit)}><div><span className="file-type">{mediaLabel(hit.media_type)}</span><strong>{hit.title}</strong></div>
          <small>{hit.heading_path} · {locatorLabel(hit.locator)}</small><p>{hit.preview}</p></button>)}</section>}
      <section className="documents"><div className="section-title"><h2>已接入资料</h2><span>{documents.length} 个文件 · {libraries.length} 个来源</span></div>
        {!loaded ? <div className="documents-skeleton"><SkeletonLines count={3} /></div>
          : documents.length === 0 ? <EmptyState glyph="▤" title="还没有接入资料">
            <p>添加第一份资料后，可以在这里查看处理状态和原文版本。</p>
            {examples.length > 0 && <>
              <p className="hint">不确定从哪开始？先导入随安装包提供的合成示例，它不包含任何个人笔记。</p>
              <div className="empty-actions">{examples.map(item => <button key={item.name}
                className="secondary" onClick={() => void importExample(item.name)}>导入随包示例「{item.name}」</button>)}</div>
            </>}
          </EmptyState> :
          <div className="document-list">{documents.map(document => <div className="document-row" key={document.id}><span className="file-icon">{mediaLabel(document.media_type)}</span>
            <div><strong>{document.display_name}</strong><small>{document.library_name} / {document.relative_path}</small>{document.error && <em>{document.error}</em>}</div>
            <span className={`status-chip ${document.status}`}>{document.status === 'ready' ? '可搜索' : document.status === 'failed' ? '失败' : document.status === 'processing' ? '处理中' : document.status}</span>
            <span className="row-actions">{document.status === 'failed' && <button className="row-action" onClick={() => void retry(document)}>重试</button>}<button className="row-action" onClick={() => void remove(document)}>移除</button></span></div>)}</div>}
      </section>
    </div>
    {selected && <SourcePanel hit={selected} close={() => setSelected(null)} />}
  </div>
}
