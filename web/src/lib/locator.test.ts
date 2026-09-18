import { describe, expect, it } from 'vitest'
import { locatorLabel, locatorLines, mediaLabel } from './locator'

describe('locatorLabel', () => {
  it('names the line range for a markdown chunk', () => {
    expect(locatorLabel({ kind: 'markdown', start_line: 12, end_line: 18 })).toBe('第 12–18 行')
  })

  it('names the page for a pdf chunk', () => {
    expect(locatorLabel({ kind: 'pdf', page: 7 })).toBe('第 7 页')
  })

  // 记忆条目没有行号。编一个"第 1–1 行"会让界面看起来像在指某份笔记的某一行，
  // 而它根本不是文档——所以这里必须说清它的出处是"哪次派生"。
  it('says where a memory item came from instead of inventing a line range', () => {
    expect(locatorLabel({ kind: 'memory', id: 'm1', derived_from: '关于排版的旧对话' }))
      .toBe('来自记忆 · 关于排版的旧对话')
  })

  it('still says "来自记忆" when the derivation is unknown', () => {
    expect(locatorLabel({ kind: 'memory', id: 'm1', derived_from: '' })).toBe('来自记忆')
  })

  // 网络来源要答案的是"这条多新"。拿不到发布时间就如实说拿不到——没有时间的外部
  // 事实本来就无从判断时效，编一个今天的时间比不显示更坏。
  it('gives the publication date for a web source, not a line range', () => {
    expect(locatorLabel({
      kind: 'web', url: 'https://arxiv.org/abs/2309.06180',
      published_at: '2023-09-12T00:00:00Z', retrieved_at: '2026-09-18T10:00:00Z',
    })).toBe('发布于 2023-09-12')
  })

  it('says the publication time is unknown rather than guessing one', () => {
    expect(locatorLabel({
      kind: 'web', url: 'https://example.com/a', published_at: null,
      retrieved_at: '2026-09-18T10:00:00Z',
    })).toBe('发布时间未知')
  })
})

describe('locatorLines', () => {
  it('highlights a line range only for sources that have one', () => {
    expect(locatorLines({ kind: 'notebook', cell: 2, cell_type: 'code', start_line: 4, end_line: 9 }))
      .toEqual([4, 9])
    expect(locatorLines({ kind: 'pdf', page: 3 })).toBeNull()
    // 记忆条目没有行号可高亮：面板要按"没有可高亮区间"处理，而不是拿 undefined 去比大小。
    expect(locatorLines({ kind: 'memory', id: 'm1', derived_from: 'x' })).toBeNull()
    // 网络来源同样没有行号：它不在本地任何文件里。
    expect(locatorLines({
      kind: 'web', url: 'https://example.com/a', published_at: null, retrieved_at: null,
    })).toBeNull()
  })
})

describe('mediaLabel', () => {
  it('does not label memory or web as a markdown document', () => {
    expect(mediaLabel('memory')).toBe('MEM')
    expect(mediaLabel('web')).toBe('WEB')
    // 文档类型不受影响。
    expect(mediaLabel('markdown')).toBe('MD')
    expect(mediaLabel('pdf')).toBe('PDF')
    expect(mediaLabel('notebook')).toBe('IPYNB')
  })
})
