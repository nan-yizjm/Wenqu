import { describe, expect, it } from 'vitest'
import { channelBadges, rankLabel, relevanceRatio } from './relevance'

describe('relevanceRatio', () => {
  it('normalizes within this result set', () => {
    expect(relevanceRatio(4, [4, 2, 0])).toBe(1)
    expect(relevanceRatio(2, [4, 2, 0])).toBe(0.5)
    expect(relevanceRatio(0, [4, 2, 0])).toBe(0)
  })

  it('returns a full bar when every score is equal', () => {
    // 全都一样时 min-max 没有分母；把它们画成空条会让人以为一条都没命中。
    expect(relevanceRatio(3, [3, 3])).toBe(1)
  })

  it('returns null without a comparable score', () => {
    // 引入分数之前的旧收藏、以及没有分数的来源，都走这一支。
    expect(relevanceRatio(null, [1, 2])).toBeNull()
    expect(relevanceRatio(undefined, [])).toBeNull()
    expect(relevanceRatio(1, [null, undefined])).toBeNull()
  })

  it('ignores missing scores in the set', () => {
    expect(relevanceRatio(2, [null, 2, 4])).toBe(0)
  })

  it('never leaves the 0-1 range', () => {
    expect(relevanceRatio(0.018, [0.018, 0.016])).toBe(1)
    expect(relevanceRatio(0.016, [0.018, 0.016])).toBe(0)
  })
})

describe('channelBadges', () => {
  it('names the channels in Chinese', () => {
    expect(channelBadges({ bm25: 2, vector: 1 })).toEqual(['关键词 第 2', '语义 第 1'])
  })

  it('is empty when only one channel took part or nothing is known', () => {
    expect(channelBadges(undefined)).toEqual([])
    expect(channelBadges({})).toEqual([])
  })
})

describe('rankLabel', () => {
  it('is one-based', () => {
    expect(rankLabel(0)).toBe('第 1 名')
    expect(rankLabel(4)).toBe('第 5 名')
  })
})