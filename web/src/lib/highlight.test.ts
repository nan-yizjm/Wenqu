import { describe, expect, it } from 'vitest'
import { splitByTokens } from './highlight'

const joined = (segments: ReturnType<typeof splitByTokens>) => segments.map(item => item.text).join('')
const marks = (segments: ReturnType<typeof splitByTokens>) =>
  segments.filter(item => item.hit).map(item => item.text)

describe('splitByTokens', () => {
  it('marks a match without losing any character', () => {
    const segments = splitByTokens('分页管理 KV Cache', ['cache'])
    expect(joined(segments)).toBe('分页管理 KV Cache')
    expect(marks(segments)).toEqual(['Cache'])
  })

  it('ignores case, because tokens are lowercased upstream', () => {
    expect(marks(splitByTokens('PagedAttention 很省显存', ['pagedattention']))).toEqual(['PagedAttention'])
  })

  it('prefers the longest match instead of overlapping', () => {
    // 中文分词给的是二元组：分页 / 页管 / 管理 会互相重叠。"页管" 被前一个
    // 匹配吃掉，剩下的 "管理" 才不会和它叠在一起。
    const segments = splitByTokens('分页管理', ['分页', '页管', '管理'])
    expect(marks(segments)).toEqual(['分页', '管理'])
    expect(joined(segments)).toBe('分页管理')
  })

  it('returns one plain segment when nothing matches', () => {
    expect(splitByTokens('无关的正文', [])).toEqual([{ text: '无关的正文', hit: false }])
    expect(splitByTokens('无关的正文', ['幻觉'])).toEqual([{ text: '无关的正文', hit: false }])
  })

  it('handles the empty string', () => {
    expect(splitByTokens('', ['a'])).toEqual([{ text: '', hit: false }])
  })

  it('drops blank tokens instead of matching everywhere', () => {
    expect(marks(splitByTokens('abc', ['', 'b']))).toEqual(['b'])
  })
})