import type { Locator } from '../api'

export function locatorLabel(locator: Locator): string {
  if (locator.kind === 'pdf') return `第 ${locator.page} 页`
  if (locator.kind === 'notebook') {
    return `单元格 ${locator.cell} · 第 ${locator.start_line}–${locator.end_line} 行`
  }
  return `第 ${locator.start_line}–${locator.end_line} 行`
}

/** 行号区间；PDF 没有行号，返回 null。 */
export function locatorLines(locator: Locator): [number, number] | null {
  return 'start_line' in locator ? [locator.start_line, locator.end_line] : null
}

export function mediaLabel(mediaType: string): string {
  return mediaType === 'pdf' ? 'PDF' : mediaType === 'notebook' ? 'IPYNB' : 'MD'
}