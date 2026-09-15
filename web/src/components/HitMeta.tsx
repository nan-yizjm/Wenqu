import { channelBadges, rankLabel, relevanceRatio } from '../lib/relevance'
import { splitByTokens } from '../lib/highlight'

/** 结果卡与来源卡片共用的命中信息：相对相关度、命中通道、命中词。 */

export function Preview({ text, tokens }: { text: string; tokens?: string[] }) {
  return <>{splitByTokens(text, tokens || []).map((segment, index) =>
    segment.hit ? <mark key={index}>{segment.text}</mark> : segment.text)}</>
}

/**
 * 相关度条。只在能拿到整组分数时使用——单条结果算不出"相对谁"。
 */
export function RelevanceBar({ score, scores }: { score?: number | null; scores: readonly (number | null | undefined)[] }) {
  const ratio = relevanceRatio(score, scores)
  if (ratio === null) return null

  return <span className="relevance" title="本次搜索结果内的相对相关度，不是绝对置信度">
    <span className="relevance-track"><span style={{ width: `${Math.round(ratio * 100)}%` }} /></span>
    <small>相关度 {Math.round(ratio * 100)}%（本次结果内相对值）</small>
  </span>
}

export function HitMeta({ hit, rank }: {
  hit: { matched_tokens?: string[]; channels?: Record<string, number> }
  rank?: number
}) {
  const badges = channelBadges(hit.channels)
  const tokens = hit.matched_tokens || []
  if (rank === undefined && !badges.length && !tokens.length) return null

  return <div className="hit-meta">
    {rank !== undefined && <span className="hit-rank">{rankLabel(rank)}</span>}
    {badges.map(badge => <span className="hit-badge" key={badge}>{badge}</span>)}
    {tokens.map(token => <span className="hit-token" key={token}>{token}</span>)}
  </div>
}