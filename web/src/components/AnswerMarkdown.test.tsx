import { render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { MessageSource } from '../api'
import { AnswerMarkdown } from './AnswerMarkdown'

const source: MessageSource = {
  chunk_id: 'c1',
  document_id: 'd1',
  version_id: 'v1',
  label: 'S1',
  title: '资料.md',
  media_type: 'markdown',
  heading_path: '章节',
  locator: { kind: 'markdown', start_line: 1, end_line: 2 },
  preview: '片段',
}

function show(content: string, sources: MessageSource[] = [source]) {
  const open = vi.fn()
  const view = render(<AnswerMarkdown content={content} sources={sources} open={open} />)
  return { ...view, open }
}

describe('AnswerMarkdown', () => {
  it('renders markdown structure instead of flat text', () => {
    const { container } = show('# 标题\n\n- 一\n- 二\n\n| a | b |\n|---|---|\n| 1 | 2 |\n')

    expect(container.querySelector('h1')?.textContent).toBe('标题')
    expect(container.querySelectorAll('li')).toHaveLength(2)
    expect(container.querySelectorAll('th')).toHaveLength(2)
  })

  it('turns a known citation into a button that opens the source', () => {
    const { container, open } = show('结论是分页管理 [S1]。')
    const citation = container.querySelector<HTMLButtonElement>('button.citation')

    expect(citation?.textContent).toBe('[S1]')
    citation?.click()
    expect(open).toHaveBeenCalledWith(source)
  })

  it('keeps a citation the model invented as plain text', () => {
    const { container } = show('结论见 [S9]。')

    expect(container.querySelector('button.citation')).toBeNull()
    expect(container.textContent).toContain('[S9]')
  })

  it('keeps a hand-written citation link to an unknown label inert', () => {
    const { container } = show('结论见 [S9](#cite-S9)。')

    expect(container.querySelector('button.citation')).toBeNull()
    expect(container.querySelector('a')).toBeNull()
    expect(container.textContent).toContain('S9')
  })

  it('does not rewrite citations inside code', () => {
    const { container } = show('```\n引用写作 [S1]\n```\n\n行内 `[S1]` 也一样。\n')
    const codes = container.querySelectorAll('code')

    expect(container.querySelectorAll('button.citation')).toHaveLength(0)
    expect(codes[0].textContent).toBe('引用写作 [S1]\n')
    expect(codes[1].textContent).toBe('[S1]')
    expect(codes[1].querySelector('a')).toBeNull()
  })

  it('never renders raw html from an answer', () => {
    const { container } = show(
      '<script>alert(1)</script>\n\n<img src=x onerror="alert(1)">\n\n<b onmouseover="alert(1)">粗</b>\n')

    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('[onerror]')).toBeNull()
    expect(container.querySelector('[onmouseover]')).toBeNull()
  })

  it('drops executable link targets', () => {
    const { container } = show('[点我](javascript:alert(1)) 和 [再来](data:text/html,x)')

    expect(container.querySelector('a[href^="javascript:"]')).toBeNull()
    expect(container.querySelector('a[href^="data:"]')).toBeNull()
    expect(container.textContent).toContain('点我')
  })

  it('keeps a plain http link usable with a safe rel', () => {
    const { container } = show('[参考](https://example.com/a)')
    const link = container.querySelector('a')

    expect(link?.getAttribute('href')).toBe('https://example.com/a')
    expect(link?.getAttribute('rel')).toContain('noreferrer')
  })

  it('renders a remote image as text so nothing is fetched', () => {
    const { container } = show('![架构图](https://tracker.example.com/pixel.png)')

    expect(container.querySelector('img')).toBeNull()
    expect(container.textContent).toContain('架构图')
    expect(container.textContent).not.toContain('tracker.example.com')
  })
})