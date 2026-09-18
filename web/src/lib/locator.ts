import type { Locator } from '../api'

export function locatorLabel(locator: Locator): string {
  if (locator.kind === 'pdf') return `第 ${locator.page} 页`
  if (locator.kind === 'notebook') {
    return `单元格 ${locator.cell} · 第 ${locator.start_line}–${locator.end_line} 行`
  }
  // 记忆条目没有行号可定位：它的"出处"是它派生自哪次对话或哪份笔记。
  if (locator.kind === 'memory') {
    return locator.derived_from ? `来自记忆 · ${locator.derived_from}` : '来自记忆'
  }
  return `第 ${locator.start_line}–${locator.end_line} 行`
}

/** 行号区间；PDF 与记忆条目都没有行号，返回 null。 */
export function locatorLines(locator: Locator): [number, number] | null {
  return 'start_line' in locator ? [locator.start_line, locator.end_line] : null
}

export function mediaLabel(mediaType: string): string {
  if (mediaType === 'pdf') return 'PDF'
  if (mediaType === 'notebook') return 'IPYNB'
  // 记忆与网络不是文档，用三个字母和文档类型区分开，免得被读成"这是一份笔记"。
  if (mediaType === 'memory') return 'MEM'
  if (mediaType === 'web') return 'WEB'
  return 'MD'
}
