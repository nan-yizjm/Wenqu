/** 回答 Markdown 渲染用的引用改写与 URL 白名单。 */

const CITATION_PATTERN = /\[(S\d+)\]/g

type MdNode = {
  type: string
  value?: string
  url?: string
  children?: MdNode[]
}

/** 只放行外部 http(s) 链接与页内引用锚点；`javascript:` 等一律清空。 */
export function sanitizeUrl(url: string | undefined): string {
  if (!url) return ''
  if (url.startsWith('#cite-')) return url
  if (/^https?:\/\//i.test(url)) return url

  return ''
}

/** 引用锚点里的标签，例如 `#cite-S2` → `S2`。 */
export function citationLabel(href: string | undefined): string | null {
  if (!href || !href.startsWith('#cite-')) return null

  return href.slice('#cite-'.length)
}

function splitCitations(value: string, labels: Set<string>): MdNode[] {
  const parts: MdNode[] = []
  let cursor = 0
  for (const match of value.matchAll(CITATION_PATTERN)) {
    const label = match[1]
    const start = match.index ?? 0
    // 模型可能写出并不存在的 [S9]；只有真实来源才变成链接。
    if (!labels.has(label)) continue
    if (start > cursor) parts.push({ type: 'text', value: value.slice(cursor, start) })
    parts.push({
      type: 'link',
      url: `#cite-${label}`,
      children: [{ type: 'text', value: match[0] }],
    })
    cursor = start + match[0].length
  }
  if (!parts.length) return [{ type: 'text', value }]
  if (cursor < value.length) parts.push({ type: 'text', value: value.slice(cursor) })

  return parts
}

function rewrite(node: MdNode, labels: Set<string>): void {
  if (!node.children) return
  const children: MdNode[] = []
  for (const child of node.children) {
    // 只改写文本节点：代码块与行内代码的内容不是文本节点，天然不会被误伤。
    if (child.type === 'text' && child.value) children.push(...splitCitations(child.value, labels))
    else { rewrite(child, labels); children.push(child) }
  }
  node.children = children
}

/**
 * 把正文里的 `[S1]` 转成 `#cite-S1` 链接，交给渲染层换成可点击的引用按钮。
 *
 * 在 mdast 文本节点上改写，而不是先对整段字符串做替换，这样列表、表格和
 * 代码块的结构不会被破坏。
 */
export function remarkCitations(labels: Set<string>) {
  return () => (tree: MdNode) => { rewrite(tree, labels) }
}