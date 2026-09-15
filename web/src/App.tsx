import { useEffect, useRef, useState } from 'react'
import { api, type Diagnostics, type ProductSettings, type ResourcesIndex, type SetupState } from './api'
import { LibraryPage } from './LibraryPage'
import { ChatPage } from './ChatPage'
import { FavoritesPage } from './FavoritesPage'

type Page = 'library' | 'chat' | 'favorites' | 'settings'
const nav: { id: Page; icon: string; label: string }[] = [
  { id: 'library', icon: '▤', label: '资料库' },
  { id: 'chat', icon: '✦', label: '知识问答' },
  { id: 'favorites', icon: '☆', label: '收藏' },
  { id: 'settings', icon: '⚙', label: '设置' },
]

function Settings({ setup, reload }: { setup: SetupState; reload: () => Promise<void> }) {
  const [form, setForm] = useState<ProductSettings>(setup.settings)
  const [key, setKey] = useState('')
  const [message, setMessage] = useState('')
  const [modelState, setModelState] = useState(setup.retrieval_model)
  const [restartRequired, setRestartRequired] = useState(false)
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null)
  const [resources, setResources] = useState<ResourcesIndex>({ docs: [], examples: [] })
  const restoreInput = useRef<HTMLInputElement>(null)
  useEffect(() => { void (async () => {
    try {
      const [reported, index] = await Promise.all([api.diagnostics(), api.resources()])
      setDiagnostics(reported); setResources(index)
    } catch { setDiagnostics(null) }
  })() }, [])
  const save = async () => {
    setMessage('正在保存…')
    try {
      await api.saveSettings(form)
      if (form.provider === 'deepseek' && key) await api.saveDeepSeek(key)
      await reload(); setKey(''); setMessage('设置已保存')
    } catch (error) { setMessage(error instanceof Error ? error.message : '保存失败') }
  }
  const prepareModel = async () => {
    setMessage('已开始准备检索模型；首次下载可能需要几分钟。')
    setModelState(await api.prepareRetrievalModel())
    const timer = window.setInterval(async () => {
      try {
        const state = await api.retrievalModelStatus()
        setModelState(state)
        if (state.status === 'ready' || state.status === 'failed') {
          window.clearInterval(timer)
          setMessage(state.status === 'ready' ? '检索模型已验证，可在无独立显卡的电脑上运行。' : `模型准备失败：${state.detail}`)
          await reload()
        }
      } catch { window.clearInterval(timer); setMessage('无法读取模型准备状态，请稍后重试。') }
    }, 1000)
  }
  const backup = async () => {
    setMessage('正在创建完整数据备份…')
    try {
      const result = await api.backup(); const url = URL.createObjectURL(result.blob)
      const link = document.createElement('a'); link.href = url
      const matched = result.disposition?.match(/filename\*?=(?:UTF-8''|\")?([^\";]+)/i)
      link.download = matched ? decodeURIComponent(matched[1]) : 'ObsidianRAG-backup.zip'
      link.click(); URL.revokeObjectURL(url); setMessage('备份已下载。')
    } catch (error) { setMessage(error instanceof Error ? error.message : '备份失败') }
  }
  const restore = async (file?: File) => {
    if (!file || !window.confirm('恢复会替换当前工作台数据。系统会先自动创建安全备份，是否继续？')) return
    setMessage('正在校验并恢复备份…')
    try { await api.restore(file); setRestartRequired(true); setMessage('恢复完成，需要退出并重新打开工作台。') }
    catch (error) { setMessage(error instanceof Error ? error.message : '恢复失败') }
    if (restoreInput.current) restoreInput.current.value = ''
  }
  return <div className="settings-page">
    <header className="page-heading"><div><span className="eyebrow">WORKSPACE SETTINGS</span><h1>设置</h1>
      <p>配置工作台名称、生成模型与检索方式；资料在「资料库」页管理。</p></div></header>
    <div className="settings-grid">
      <section className="card"><h3>工作台</h3><label>显示名称<input value={form.display_name}
        onChange={e => setForm({ ...form, display_name: e.target.value })} /></label>
        <div className="status-row"><span>用户数据</span><code>{setup.data_root}</code></div></section>
      <section className="card"><h3>生成模型</h3><div className="segmented">
        <button className={form.provider === 'ollama' ? 'active' : ''} onClick={() => setForm({ ...form, provider: 'ollama' })}>本机 Ollama</button>
        <button className={form.provider === 'deepseek' ? 'active' : ''} onClick={() => setForm({ ...form, provider: 'deepseek' })}>DeepSeek</button>
      </div>
      {form.provider === 'ollama' ? <>
        <label>服务地址<input value={form.ollama_base_url} onChange={e => setForm({ ...form, ollama_base_url: e.target.value })} /></label>
        <label>模型名称<input value={form.ollama_model} onChange={e => setForm({ ...form, ollama_model: e.target.value })} /></label>
      </> : <label>API Key<input type="password" value={key} placeholder={setup.deepseek_key_configured ? '已安全保存；留空则不修改' : '输入 DeepSeek API Key'} onChange={e => setKey(e.target.value)} /></label>}
      <p className="hint">选择 DeepSeek 后，问题与用于回答的资料片段会发送到该服务。</p></section>
      <section className="card"><h3>检索模型</h3><div className="status-line"><span className={`dot ${modelState.status === 'ready' ? 'green' : 'amber'}`} />
        {modelState.status === 'ready' ? 'multilingual-e5-small 已准备' : modelState.detail || '尚未下载'}</div>
        <p className="hint">固定版本，默认使用 CPU。首次准备会下载模型并执行 384 维归一化向量检查。当前问答使用 BM25 关键词检索，准备与否都不影响现在的搜索结果；语义召回会在后续版本启用后使用它。</p>
        {modelState.status !== 'ready' && !['downloading', 'loading', 'verifying'].includes(modelState.status) &&
          <button className="secondary" onClick={prepareModel}>下载并验证模型</button>}</section>
      <section className="card"><h3>随包文档</h3>
        <p className="hint">这些文件随安装包提供，与本机资料分开保管，不会被检索。</p>
        <div className="support-actions">{resources.docs.map(item => <a key={item.name}
          className="secondary export-link" href={api.bundledDocUrl(item.name)}
          target="_blank" rel="noreferrer">{item.name.replace(/\.md$/, '')}</a>)}
          {!resources.docs.length && <span className="hint">未找到随包文档。</span>}</div></section>
      <section className="card"><h3>运行状态</h3>
        {diagnostics ? <>
          <div className="status-row"><span>产品版本</span><code>{diagnostics.product_version}</code></div>
          <div className="status-row"><span>数据库结构</span><code>v{diagnostics.database_schema}</code></div>
          <div className="status-row"><span>运行方式</span><code>{diagnostics.frozen ? '安装版' : '源码运行'}</code></div>
          <div className="status-line"><span className={`dot ${diagnostics.data_directories.database_exists ? 'green' : 'amber'}`} />
            {diagnostics.data_directories.database_exists ? '用户数据目录可读写' : '用户数据目录不完整'}</div>
          <div className="status-line"><span className={`dot ${diagnostics.retrieval_model.status === 'ready' ? 'green' : 'amber'}`} />
            {diagnostics.retrieval_model.status === 'ready' ? '检索模型已就绪' : '检索模型尚未准备'}</div>
        </> : <p className="hint">无法读取运行状态；本地服务可能刚刚启动。</p>}</section>
      <section className="card"><h3>备份、恢复与诊断</h3><p className="hint">备份包含资料快照、会话、收藏和导出，不包含模型缓存或密钥。恢复前会先保存当前数据。</p>
        <div className="support-actions"><button className="secondary" onClick={() => void backup()}>下载数据备份</button><button className="secondary" onClick={() => restoreInput.current?.click()}>从备份恢复</button>
          <a className="secondary export-link" href={api.diagnosticExportUrl()}>下载脱敏诊断</a></div>
        <input ref={restoreInput} className="hidden" type="file" accept=".zip" onChange={event => void restore(event.target.files?.[0])} />
        {restartRequired && <button className="primary" onClick={() => void api.shutdown()}>退出工作台</button>}</section>
    </div>
    <div className="save-bar"><span>{message}</span>{!restartRequired && <button className="primary" onClick={save}>保存设置</button>}</div>
  </div>
}

function Welcome({ setup, done }: { setup: SetupState; done: () => Promise<void> }) {
  const [name, setName] = useState(setup.settings.display_name)
  const [provider, setProvider] = useState<'ollama' | 'deepseek'>(setup.settings.provider)
  const [key, setKey] = useState('')
  const [error, setError] = useState('')
  const complete = async () => {
    try {
      await api.saveSettings({ display_name: name, provider, onboarding_complete: true })
      if (provider === 'deepseek' && key) await api.saveDeepSeek(key)
      await done()
    } catch (e) { setError(e instanceof Error ? e.message : '保存失败') }
  }
  return <main className="welcome"><div className="welcome-copy"><div className="brand-mark">OR</div>
    <span className="eyebrow">PERSONAL KNOWLEDGE WORKSPACE</span><h1>让你的资料<br />变成可追溯的答案</h1>
    <p>在本机整理 Markdown 与 PDF，搜索原文、继续追问，并把有价值的结论保存下来。</p>
    <div className="privacy-note"><strong>资料留在你的电脑</strong><span>只有选择远端模型时，回答所需片段才会发送给服务商。</span></div>
  </div><div className="setup-card"><span className="step">首次设置 · 1 分钟</span><h2>创建你的工作台</h2>
    <label>工作台名称<input value={name} onChange={e => setName(e.target.value)} /></label>
    <label>回答使用</label><div className="provider-cards">
      <button className={provider === 'ollama' ? 'selected' : ''} onClick={() => setProvider('ollama')}><strong>本机 Ollama</strong><small>资料和回答都留在本机</small></button>
      <button className={provider === 'deepseek' ? 'selected' : ''} onClick={() => setProvider('deepseek')}><strong>DeepSeek</strong><small>配置 API Key 后使用</small></button>
    </div>
    {provider === 'deepseek' && <label>DeepSeek API Key<input type="password" value={key} onChange={e => setKey(e.target.value)} placeholder="保存在 Windows 凭据管理器" /></label>}
    {error && <p className="error">{error}</p>}<button className="primary wide" onClick={complete}>进入工作台 <span>→</span></button>
    <p className="fineprint">下一步将在工作台中添加资料和准备检索模型。</p></div></main>
}

function Recovery({ setup }: { setup: SetupState }) {
  const backups = setup.recovery_backups || []
  const [selected, setSelected] = useState(backups[0]?.name || '')
  const [message, setMessage] = useState('')
  const restore = async () => {
    if (!selected) return
    try { await api.restoreMigration(selected); setMessage('恢复完成。请退出后重新打开，系统会再次执行升级。') }
    catch (error) { setMessage(error instanceof Error ? error.message : '恢复失败') }
  }
  return <main className="recovery-page"><section className="setup-card"><span className="step">安全恢复模式</span><h1>数据库升级没有完成</h1>
    <p>工作台没有继续加载资料或模型，以免扩大损坏。请选择升级前自动备份恢复；当前失败数据库也会另行保留。</p>
    {backups.length ? <><label>可用迁移备份<select value={selected} onChange={event => setSelected(event.target.value)}>{backups.map(item => <option key={item.name} value={item.name}>{item.name} · {Math.ceil(item.size / 1024)} KB</option>)}</select></label>
      <button className="primary wide" onClick={() => void restore()}>恢复所选备份</button></> : <p className="error">没有找到可自动恢复的迁移备份。请保留用户数据目录，并使用脱敏日志寻求帮助。</p>}
    {message && <p className="status-line">{message}</p>}{message && <button className="secondary" onClick={() => void api.shutdown()}>退出工作台</button>}
  </section></main>
}

export default function App() {
  const [setup, setSetup] = useState<SetupState | null>(null)
  const [page, setPage] = useState<Page>('library')
  const [failure, setFailure] = useState('')
  const reload = async () => { try { setSetup(await api.setup()) } catch (e) { setFailure(e instanceof Error ? e.message : '无法连接本地服务') } }
  useEffect(() => { void reload() }, [])
  if (failure) return <main className="fatal"><h1>工作台没有准备好</h1><p>{failure}</p><button onClick={() => location.reload()}>重新连接</button></main>
  if (!setup) return <main className="loading"><div className="spinner" /><p>正在打开知识工作台…</p></main>
  if (setup.recovery_required) return <Recovery setup={setup} />
  if (!setup.settings.onboarding_complete) return <Welcome setup={setup} done={reload} />
  return <div className="shell"><aside><div className="sidebar-brand"><div className="brand-mark small">OR</div><div><strong>{setup.settings.display_name}</strong><span>个人知识工作台</span></div></div>
    <nav>{nav.map(item => <button key={item.id} className={page === item.id ? 'active' : ''} onClick={() => setPage(item.id)}><span>{item.icon}</span>{item.label}</button>)}</nav>
    <div className="sidebar-status"><span className={`dot ${setup.materials.ready_documents ? '' : 'amber'}`} /><div><strong>{setup.materials.ready_documents ? `${setup.materials.ready_documents} 份资料可用` : '等待添加资料'}</strong><small>{setup.materials.chunk_count ? `${setup.materials.chunk_count} 个可检索片段` : '本地服务已就绪'}</small></div></div></aside>
    <main className="content">{page === 'settings' ? <Settings setup={setup} reload={reload} />
      : page === 'library' ? <LibraryPage setupReload={reload} />
      : page === 'chat' ? <ChatPage /> : <FavoritesPage />}</main>
  </div>
}
