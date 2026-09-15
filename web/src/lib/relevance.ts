/** 检索结果的相关度表达。只做本次结果集内的相对值，不假装是绝对置信度。 */

/** 名次（一基），用于结果卡和来源列表的"第 N 名"。 */
export function rankLabel(index: number): string {
  return `第 ${index + 1} 名`
}

/**
 * 本次结果集内 min-max 归一的相关度，0–1；没有可比的分数时返回 null。
 *
 * 不显示原始分：融合分（RRF 约 0.01–0.03）、BM25 分（任意正数）和重排 logit
 * 量级互不可比，同一个数字在两种检索方式下含义完全不同。所以只给"和本次最好
 * 那条比"，并且由调用方明确标注这是相对值。
 */
export function relevanceRatio(score: number | null | undefined, scores: readonly (number | null | undefined)[]): number | null {
  if (typeof score !== 'number') return null
  const usable = scores.filter((item): item is number => typeof item === 'number')
  if (!usable.length) return null
  const min = Math.min(...usable)
  const max = Math.max(...usable)
  if (max <= min) return 1

  return (score - min) / (max - min)
}

const CHANNEL_NAMES: Record<string, string> = { bm25: '关键词', vector: '语义' }

/** 命中通道徽章，例如 `['关键词 第 2', '语义 第 1']`。 */
export function channelBadges(channels?: Record<string, number> | null): string[] {
  if (!channels) return []

  return Object.entries(channels).map(
    ([name, rank]) => `${CHANNEL_NAMES[name] ?? name} 第 ${rank}`,
  )
}