/** 在预览文本里标出检索命中的词。 */

export type Segment = { text: string; hit: boolean }

/**
 * 按命中词把文本切成片段。
 *
 * 长词优先、不重叠：中文分词出的是二元组（"分页"、"页管"、"管理"），逐个标
 * 会互相盖住；从左往右取最长匹配，分段就干净了。大小写不敏感，因为 BM25 的
 * token 是小写的，而原文是原样的。
 */
export function splitByTokens(text: string, tokens: readonly string[]): Segment[] {
  const needles = [...new Set(tokens.filter(Boolean).map(token => token.toLowerCase()))]
    .sort((left, right) => right.length - left.length)
  if (!needles.length) return [{ text, hit: false }]
  const segments: Segment[] = []
  let plain = ''
  let index = 0
  while (index < text.length) {
    const length = needles.find(
      needle => text.slice(index, index + needle.length).toLowerCase() === needle)?.length
    if (!length) { plain += text[index]; index += 1; continue }
    if (plain) { segments.push({ text: plain, hit: false }); plain = '' }
    segments.push({ text: text.slice(index, index + length), hit: true })
    index += length
  }
  if (plain) segments.push({ text: plain, hit: false })

  return segments.length ? segments : [{ text, hit: false }]
}