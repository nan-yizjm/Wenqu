import { useEffect, useState } from 'react'
import { api, type Favorite, type FavoriteSummary, type MessageSource, type SearchHit } from './api'
import { SourcePanel } from './SourcePanel'

const asHit = (source: MessageSource): SearchHit => ({ ...source, score: 0, matched_tokens: [] })

export function FavoritesPage() {
  const [items, setItems] = useState<FavoriteSummary[]>([])
  const [active, setActive] = useState<Favorite | null>(null)
  const [source, setSource] = useState<MessageSource | null>(null)
  const [title, setTitle] = useState(''); const [note, setNote] = useState('')
  const [message, setMessage] = useState('')
  const load = async (preferred?: string) => {
    const list = (await api.favorites()).favorites; setItems(list)
    const id = preferred || active?.id || list[0]?.id
    if (!id) { setActive(null); return }
    const detail = await api.favorite(id); setActive(detail); setTitle(detail.title); setNote(detail.note)
  }
  useEffect(() => { void load().catch(error => setMessage(error.message)) }, [])
  const choose = async (id: string) => { setSource(null); await load(id) }
  const save = async () => { if (!active) return; const value = await api.updateFavorite(active.id, { title, note }); setActive(value); setMessage('已保存'); await load(value.id) }
  const remove = async () => { if (!active || !window.confirm(`删除收藏“${active.title}”？`)) return; await api.deleteFavorite(active.id); setActive(null); setSource(null); await load(); }
  return <div className={source ? 'favorites-page with-source' : 'favorites-page'}>
    <section className="favorites-list"><header><span className="eyebrow">SAVED KNOWLEDGE</span><h1>收藏</h1><p>{items.length} 条可导出结论</p></header>
      {items.length === 0 ? <div className="favorites-empty">在知识问答中收藏带来源的完整回答。</div> : items.map(item => <button key={item.id} className={active?.id === item.id ? 'active' : ''} onClick={() => void choose(item.id)}><strong>{item.title}</strong><small>{item.source_count} 个来源 · {item.provider || '未知模型'}</small></button>)}</section>
    <section className="favorite-detail">{!active ? <div className="chat-empty"><div>☆</div><h2>还没有收藏</h2><p>收藏会保留回答、问题和当时使用的证据快照。</p></div> : <>
      <header><div><span className="eyebrow">SAVED ANSWER</span><h1>{active.title}</h1></div><div><a className="secondary export-link" href={api.favoriteExportUrl(active.id)}>导出 Markdown</a><button className="row-action" onClick={() => void remove()}>删除</button></div></header>
      <div className="favorite-form"><label>标题<input value={title} onChange={event => setTitle(event.target.value)} /></label><label>备注<textarea value={note} onChange={event => setNote(event.target.value)} placeholder="记录为什么值得保留，或下一步要做什么" /></label><button className="primary" onClick={() => void save()}>保存修改</button><span>{message}</span></div>
      <article className="favorite-content"><h3>问题</h3><p>{active.question}</p><h3>回答</h3><div>{active.answer}</div><h3>来源</h3><div className="favorite-sources">{active.sources.map(item => <button key={item.label} onClick={() => setSource(item)}><b>{item.label}</b><span><strong>{item.title}</strong><small>{item.heading_path}</small></span></button>)}</div></article>
    </>}</section>
    {source && <SourcePanel hit={asHit(source)} close={() => setSource(null)} />}
  </div>
}
