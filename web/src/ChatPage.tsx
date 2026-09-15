import { useEffect, useRef, useState } from 'react'
import { api, type ChatMessage, type Conversation, type FeedbackKind, type MessageSource, type SearchHit, type StreamEvent } from './api'
import { SourcePanel } from './SourcePanel'
import { AnswerMarkdown } from './components/AnswerMarkdown'

function hitFromSource(source: MessageSource): SearchHit {
  return { ...source, score: 0, matched_tokens: [] }
}

const FEEDBACK_KINDS: { kind: FeedbackKind; label: string; hint: string }[] = [
  { kind: 'helpful', label: '有帮助', hint: '答案和引用都对' },
  { kind: 'missing', label: '有遗漏', hint: '资料里有但没答到' },
  { kind: 'citation_wrong', label: '引用不对', hint: '[S1] 指错了位置' },
  { kind: 'answer_wrong', label: '回答不对', hint: '与原文不符' },
]

function FeedbackPanel({ saved, onSave }: {
  saved?: { kind: FeedbackKind; note: string }
  onSave: (kind: FeedbackKind, note: string) => Promise<void>
}) {
  const [note, setNote] = useState(saved?.note || '')
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const submit = async (kind: FeedbackKind) => {
    setBusy(true)
    try { await onSave(kind, note) } finally { setBusy(false) }
  }
  return <div className="feedback-panel">
    <span className="feedback-prompt">{saved ? '已记录' : '这条回答'}</span>
    {FEEDBACK_KINDS.map(item => <button key={item.kind} title={item.hint} disabled={busy}
      className={saved?.kind === item.kind ? 'active' : ''}
      onClick={() => void submit(item.kind)}>{item.label}</button>)}
    <button className="ghost" onClick={() => setOpen(!open)}>
      {open ? '收起备注' : saved?.note ? '修改备注' : '＋ 备注'}</button>
    {open && <div className="feedback-note">
      <textarea value={note} maxLength={2000} onChange={event => setNote(event.target.value)}
        placeholder="补充说明（可选）：哪里对、哪里缺、引用是否指对位置。备注只保存在本机。" />
      <small>选择上面任一选项即可连同备注一起保存；再次选择会覆盖上一次记录。</small>
    </div>}
  </div>
}

export function ChatPage() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [active, setActive] = useState<Conversation | null>(null)
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [selected, setSelected] = useState<MessageSource | null>(null)
  const [savedFeedback, setSavedFeedback] = useState<Record<string, { kind: FeedbackKind; note: string }>>({})
  const controller = useRef<AbortController | null>(null)
  const streamingMessage = useRef<string | null>(null)
  const bottom = useRef<HTMLDivElement>(null)

  const loadList = async () => setConversations((await api.conversations()).conversations)
  const openConversation = async (id: string) => { setSelected(null); setActive(await api.conversation(id)) }
  useEffect(() => { void (async () => {
    const items = (await api.conversations()).conversations
    setConversations(items)
    const conversation = items[0] || await api.createConversation()
    setActive(await api.conversation(conversation.id))
    if (!items.length) await loadList()
  })().catch(error => setMessage(error.message)) }, [])
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [active?.messages])

  const run = async (body: { question?: string; retry_message_id?: string; skip_guard?: boolean }) => {
    if (!active || busy) return
    const prompt = body.question?.trim()
    if (!prompt && !body.retry_message_id) return
    const tempId = `temp-${Date.now()}`
    const optimistic = [...(active.messages || [])]
    if (prompt) optimistic.push({ id: `user-${Date.now()}`, conversation_id: active.id, role: 'user', content: prompt,
      status: 'complete', provider: null, model: null, index_version: null, error_code: null, reply_to_message_id: null, sources: [] })
    optimistic.push({ id: tempId, conversation_id: active.id, role: 'assistant', content: '', status: 'streaming',
      provider: null, model: null, index_version: null, error_code: null, reply_to_message_id: null, sources: [] })
    setActive({ ...active, messages: optimistic }); setQuestion(''); setBusy(true); setMessage('')
    const abort = new AbortController(); controller.current = abort
    const applyEvent = (event: StreamEvent) => setActive(current => {
      streamingMessage.current = event.message_id
      if (!current) return current
      const messages = [...(current.messages || [])]
      const index = messages.findIndex(item => item.id === tempId || item.id === event.message_id)
      if (index < 0) return current
      const existing = messages[index]
      const next = { ...existing, id: event.message_id }
      if (event.type === 'retrieval') next.sources = event.sources || []
      if (event.type === 'generation') { next.provider = event.provider || null; next.model = event.model || null }
      if (event.type === 'token') next.content += event.text || ''
      if (event.type === 'final' || event.type === 'stopped' || event.type === 'error') {
        next.content = event.content ?? next.content; next.status = event.status || (event.type === 'error' ? 'failed' : 'complete')
        if (event.sources) next.sources = event.sources
        if (event.rejected) next.error_code = 'guard_rejected'
        if (event.type === 'error') setMessage(event.message || '生成失败')
        if (event.citation_warning) setMessage('回答没有引用有效证据，请打开来源自行核对。')
      }
      messages[index] = next
      return { ...current, messages }
    })
    try {
      await api.streamMessage(active.id, body, applyEvent, abort.signal)
    } catch (error) {
      if (!(error instanceof DOMException && error.name === 'AbortError')) setMessage(error instanceof Error ? error.message : '问答失败')
    } finally {
      setBusy(false); controller.current = null; streamingMessage.current = null
      window.setTimeout(() => { void openConversation(active.id); void loadList() }, 350)
    }
  }
  const stop = async () => {
    if (!active || !streamingMessage.current) { controller.current?.abort(); return }
    try {
      await api.stopMessage(active.id, streamingMessage.current)
      setMessage('正在停止；已生成文本将保留为未完成回答。')
    } catch {
      controller.current?.abort(); setBusy(false); setMessage('连接已中断；正在保存已生成文本。')
    }
  }
  const create = async () => { const item = await api.createConversation(); await loadList(); await openConversation(item.id) }
  const favorite = async (item: ChatMessage) => {
    try { await api.createFavorite(item.id); setMessage('已收藏；可在左侧“收藏”中编辑和导出。') }
    catch (error) { setMessage(error instanceof Error ? error.message : '收藏失败') }
  }
  const feedback = async (item: ChatMessage, kind: FeedbackKind, note: string) => {
    try {
      await api.feedback(item.id, kind, note)
      setSavedFeedback(current => ({ ...current, [item.id]: { kind, note } }))
      setMessage('反馈已保存在本机。')
    } catch (error) { setMessage(error instanceof Error ? error.message : '反馈保存失败') }
  }
  const overrideGuard = async (item: ChatMessage) => {
    const original = active?.messages?.find(message => message.id === item.reply_to_message_id)
    if (!original?.content) { setMessage('找不到原始问题，请重新输入一次。'); return }
    await run({ question: original.content, skip_guard: true })
  }
  return <div className={selected ? 'chat-page with-source' : 'chat-page'}>
    <section className="conversation-rail"><button className="primary new-chat" onClick={create}>＋ 新会话</button>
      <div className="conversation-list">{conversations.map(item => <button key={item.id} className={active?.id === item.id ? 'active' : ''} onClick={() => void openConversation(item.id)}><strong>{item.title}</strong><small>{item.message_count || 0} 条消息</small></button>)}</div>
    </section>
    <section className="chat-main"><header><div><span className="eyebrow">GROUNDED ANSWERS</span><h1>{active?.title || '知识问答'}</h1></div><span className="chat-note">每次回答都会重新检索当前资料</span></header>
      <div className="message-list">{!active?.messages?.length && <div className="chat-empty"><div>✦</div><h2>从自己的资料开始提问</h2><p>答案中的引用可以直接打开当时使用的原文快照。</p></div>}
        {active?.messages?.map(item => <article key={item.id} className={`message ${item.role} ${item.status}`}>
          <div className="message-label">{item.role === 'user' ? '你' : '知识工作台'}{item.status === 'stopped' ? ' · 未完成' : item.status === 'failed' ? ' · 失败' : ''}</div>
          {item.role === 'user' ? <div className="question-text">{item.content}</div>
            : item.status === 'streaming' ? <div className="answer-text">{item.content}</div>
            : <AnswerMarkdown content={item.content} sources={item.sources} open={setSelected} />}
          {item.role === 'assistant' && item.sources.length > 0 && <div className="source-chips">{item.sources.map(source => <button key={source.label} onClick={() => setSelected(source)}><b>{source.label}</b>{source.title}</button>)}</div>}
          {item.role === 'assistant' && item.status === 'complete' && item.error_code !== 'guard_rejected' && <div className="answer-actions">
            {item.sources.length > 0 && <button onClick={() => void favorite(item)}>☆ 收藏</button>}
            <FeedbackPanel saved={savedFeedback[item.id]} onSave={(kind, note) => feedback(item, kind, note)} />
          </div>}
          {item.role === 'assistant' && item.error_code === 'guard_rejected' && !busy && <button className="retry" onClick={() => void overrideGuard(item)}>仍然按资料提问</button>}
          {item.role === 'assistant' && ['failed', 'stopped'].includes(item.status) && !busy && <button className="retry" onClick={() => void run({ retry_message_id: item.id })}>重新生成</button>}
        </article>)}<div ref={bottom} /></div>
      <div className="composer"><textarea value={question} onChange={event => setQuestion(event.target.value)} placeholder="询问你的资料；Shift + Enter 换行" disabled={busy}
        onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void run({ question }) } }} />
        {busy ? <button className="stop" onClick={() => void stop()}>■ 停止</button> : <button className="primary" onClick={() => void run({ question })}>发送</button>}
        {message && <p>{message}</p>}<small>停止远端回答只会断开本地连接，不保证服务商已取消计算或计费。</small></div>
    </section>
    {selected && <SourcePanel hit={hitFromSource(selected)} close={() => setSelected(null)} />}
  </div>
}
