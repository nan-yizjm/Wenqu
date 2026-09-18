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
})

describe('locatorLines', () => {
  it('highlights a line range only for sources that have one', () => {
    expect(locatorLines({ kind: 'notebook', cell: 2, cell_type: 'code', start_line: 4, end_line: 9 }))
      .toEqual([4, 9])
    expect(locatorLines({ kind: 'pdf', page: 3 })).toBeNull()
    // 记忆条目没有行号可高亮：面板要按"没有可高亮区间"处理，而不是拿 undefined 去比大小。
    expect(locatorLines({ kind: 'memory', id: 'm1', derived_from: 'x' })).toBeNull()
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
