import { useEffect, useState } from 'react'
import { api, type Favorite, type FavoriteFilters, type FavoriteSummary, type FavoritesView, type FeedbackKind, type MessageSource } from './api'
import { SourcePanel } from './SourcePanel'
import { AnswerMarkdown } from './components/AnswerMarkdown'
import { HitMeta, RelevanceBar } from './components/HitMeta'
import { EmptyState, SkeletonLines } from './components/Placeholders'
import { BatchDeleteBar, useSelection } from './components/BatchDelete'
import { FEEDBACK_KINDS } from './lib/feedback'

const RANGES: { value: NonNullable<FavoriteFilters['days']>; label: string }[] = [
  { value: '7d', label: '最近 7 天' }, { value: '30d', label: '最近 30 天' },
]
// 中英文逗号都当分隔符：中文输入法下打出来的多半是「，」。
const parseTags = (text: string) => text.split(/[,，]/).map(value => value.trim()).filter(Boolean)

export function FavoritesPage() {
  const [view, setView] = useState<FavoritesView | null>(null)
  const [filters, setFilters] = useState<FavoriteFilters>({})
  const [active, setActive] = useState<Favorite | null>(null)
  const [source, setSource] = useState<MessageSource | null>(null)
  const [title, setTitle] = useState(''); const [note, setNote] = useState('')
  const [tagText, setTagText] = useState(''); const [collection, setCollection] = useState('')
  const [message, setMessage] = useState('')
  const [loaded, setLoaded] = useState(false)
  const [selecting, setSelecting] = useState(false)
  const selection = useSelection()
  const favorites = view?.favorites ?? []
  const narrowed = view ? view.favorites.length !== view.total : false
  const filtering = Object.values(filters).some(Boolean)

  const load = async (preferred?: string) => {
    const data = await api.favorites(filters); setView(data)
    const ids = data.favorites.map(item => item.id)
    // 筛选之后原来那条可能已经不在列表里了，此时改选列表里的第一条——
    // 继续显示一条"当前筛选看不到"的收藏只会让人以为筛选没生效。
    const id = preferred || (active && ids.includes(active.id) ? active.id : '') || ids[0]
    if (!id) { setActive(null); return }
    const detail = await api.favorite(id)
    setActive(detail); setTitle(detail.title); setNote(detail.note); setTagText(detail.tags.join(', '))
  }
  useEffect(() => { void load().catch(error => setMessage(error.message)).finally(() => setLoaded(true)) }, [filters])
  const choose = async (id: string) => { setSource(null); await load(id) }
  const save = async () => {
    if (!active) return
    try {
      const value = await api.updateFavorite(active.id, { title, note, tags: parseTags(tagText) })
      setActive(value); setTagText(value.tags.join(', ')); setMessage('已保存'); await load(value.id)
    } catch (error) { setMessage(error instanceof Error ? error.message : '保存失败') }
  }
  const remove = async () => { if (!active || !window.confirm(`删除收藏“${active.title}”？`)) return; await api.deleteFavorite(active.id); setActive(null); setSource(null); await load() }
  const exitSelecting = () => { setSelecting(false); selection.clear() }
  const favoriteRow = (item: FavoriteSummary) => <button className={active?.id === item.id ? 'active' : ''}
    onClick={() => selecting ? selection.toggle(item.id) : void choose(item.id)}>
    <strong>{item.title}</strong>
    <small>{item.source_count} 个来源 · {item.provider || '未知模型'}</small>
    {item.tags.length > 0 && <span className="tag-row">{item.tags.map(tag => <i key={tag}>{tag}</i>)}</span>}
  </button>
  const filter = (patch: FavoriteFilters) => setFilters({ ...filters, ...patch })
  return <div className={source ? 'favorites-page with-source' : 'favorites-page'}>
    <section className="favorites-list"><header><span className="eyebrow">SAVED KNOWLEDGE</span><h1>收藏</h1>
      <p>{favorites.length} 条可导出结论{narrowed ? `（共 ${view?.total} 条）` : ''}</p>
      {view && view.total > 0 && !selecting && <button className="ghost" onClick={() => setSelecting(true)}>批量选择</button>}
      </header>
      {view && view.total > 0 && <div className="favorite-filters">
        <label>资料库<select value={filters.library || ''} onChange={event => filter({ library: event.target.value || undefined })}>
          <option value="">全部</option>
          {view.libraries.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
        <label>时间<select value={filters.days || ''} onChange={event => filter({ days: (event.target.value || undefined) as FavoriteFilters['days'] })}>
          <option value="">全部</option>
          {RANGES.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
        <label>反馈<select value={filters.feedback || ''} onChange={event => filter({ feedback: (event.target.value || undefined) as FeedbackKind })}>
          <option value="">全部</option>
          {FEEDBACK_KINDS.map(item => <option key={item.kind} value={item.kind}>{item.label}</option>)}</select></label>
        {filtering && <button className="ghost" onClick={() => setFilters({})}>清除筛选</button>}
      </div>}
      {view && view.tags.length > 0 && <div className="tag-cloud">{view.tags.map(tag =>
        <button key={tag} className={filters.tag === tag ? 'active' : ''}
          onClick={() => filter({ tag: filters.tag === tag ? undefined : tag })}>{tag}</button>)}</div>}
      {view && view.total > 0 && <div className="collection-export">
        <input value={collection} maxLength={80} onChange={event => setCollection(event.target.value)}
          placeholder="专题标题（可选）" />
        {favorites.length > 0
          ? <a className="secondary export-link" href={api.collectionExportUrl(
            favorites.map(item => item.id), collection)}>导出专题（{favorites.length}）</a>
          : <span className="disabled-export">导出专题（0）</span>}</div>}
      {favorites.length === 0 ? <div className="favorites-empty">
        {narrowed ? <>没有符合筛选的收藏。<button className="link" onClick={() => setFilters({})}>清除筛选</button></>
          : '在知识问答中收藏带来源的完整回答。'}</div>
        : favorites.map(item => selecting
          ? <div key={item.id} className={`favorite-item${selection.isSelected(item.id) ? ' batch-selected' : ''}`}>
              <input type="checkbox" checked={selection.isSelected(item.id)} onChange={() => selection.toggle(item.id)} />
              {favoriteRow(item)}
            </div>
          : <div key={item.id} className="favorite-item">{favoriteRow(item)}</div>)}
      {selecting && favorites.length > 0 && <BatchDeleteBar
        ids={[...selection.selected]}
        allIds={favorites.map(item => item.id)}
        allSelected={selection.selected.size > 0 && selection.selected.size === favorites.length}
        onToggleAll={() => selection.selected.size === favorites.length
          ? selection.clear() : selection.setAll(favorites.map(item => item.id))}
        heading={`删除 ${selection.selected.size} 条收藏？`}
        lines={['收藏是副本，删除后不影响原会话与原回答。']}
        onCancel={exitSelecting}
        onConfirm={ids => api.deleteFavorites(ids)}
        onDone={async () => { exitSelecting(); await load() }} />}
      </section>
    <section className="favorite-detail">{!loaded ? <div className="detail-skeleton"><SkeletonLines count={6} /></div>
      : !active ? (narrowed ? <EmptyState glyph="☆" title="筛选后没有可显示的收藏" tall>
        <p>当前筛选条件下没有收藏；清掉筛选就能看到全部。</p>
        <button className="link" onClick={() => setFilters({})}>清除筛选</button></EmptyState>
        : <EmptyState glyph="☆" title="还没有收藏" tall>
          <p>收藏会保留回答、问题和当时使用的证据快照。</p></EmptyState>) : <>
      <header><div><span className="eyebrow">SAVED ANSWER</span><h1>{active.title}</h1></div><div><a className="secondary export-link" href={api.favoriteExportUrl(active.id)}>导出 Markdown</a><button className="row-action" onClick={() => void remove()}>删除</button></div></header>
      <div className="favorite-form"><label>标题<input value={title} onChange={event => setTitle(event.target.value)} /></label>
        <label>备注<textarea value={note} onChange={event => setNote(event.target.value)} placeholder="记录为什么值得保留，或下一步要做什么" /></label>
        <label>标签<input value={tagText} onChange={event => setTagText(event.target.value)} placeholder="用逗号分开，最多 8 个，例如：检索, 待复现" /></label>
        <button className="primary" onClick={() => void save()}>保存修改</button><span>{message}</span></div>
      <article className="favorite-content"><h3>问题</h3><p>{active.question}</p><h3>回答</h3><AnswerMarkdown content={active.answer} sources={active.sources} open={setSource} /><h3>来源</h3><div className="favorite-sources">{active.sources.map((item, index) => <button key={item.label} title={item.preview} onClick={() => setSource(item)}><b>{item.label}</b><span><strong>{item.title}</strong><small>{item.heading_path}</small><RelevanceBar score={item.score} scores={active.sources.map(source => source.score)} /><HitMeta hit={item} rank={index} /></span></button>)}</div></article>
    </>}</section>
    {source && <SourcePanel hit={source} close={() => setSource(null)} />}
  </div>
}