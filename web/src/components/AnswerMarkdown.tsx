import { memo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSanitize from 'rehype-sanitize'
import type { MessageSource } from '../api'
import { citationLabel, remarkCitations, sanitizeUrl } from '../lib/markdown'

type AnswerMarkdownProps = {
  content: string
  sources: MessageSource[]
  open: (source: MessageSource) => void
}

/** 把回答渲染成 Markdown；引用、外链和图片都经过白名单处理。 */
export const AnswerMarkdown = memo(function AnswerMarkdown({ content, sources, open }: AnswerMarkdownProps) {
  const byLabel = new Map(sources.map(source => [source.label, source]))
  const labels = new Set(byLabel.keys())
  return <div className="answer-markdown">
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkCitations(labels)]}
      rehypePlugins={[rehypeSanitize]}
      urlTransform={sanitizeUrl}
      components={{
        a: ({ href, children }) => {
          const source = byLabel.get(citationLabel(href) || '')
          if (source) return <button className="citation" aria-label={`引用 ${source.label}：${source.title}`}
                onClick={() => open(source)}>{children}</button>
          // 模型可以自己写 [S9](#cite-S9)；标签不在来源里就退回纯文本。
          if (href && href.startsWith('#cite-')) return <span className="citation-invalid">{children}</span>
          if (!href) return <span>{children}</span>

          return <a href={href} target="_blank" rel="noreferrer noopener">{children}</a>
        },
        // 资料留在本机：远端图片会变成对外请求，因此一律只渲染文字。
        img: ({ alt }) => <span className="remote-image">［图片：{alt || '未命名'}］</span>,
      }}
    >{content}</ReactMarkdown>
  </div>
})