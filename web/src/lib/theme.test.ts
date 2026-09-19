import { afterEach, describe, expect, it } from 'vitest'
import {
  applyTemplate, applyTheme, bootstrapAppearance, readStoredTemplate, resolveTheme,
  TEMPLATE_OPTIONS, TEMPLATE_STORAGE_KEY, THEME_STORAGE_KEY,
} from './theme'

afterEach(() => {
  delete document.documentElement.dataset.theme
  delete document.documentElement.dataset.template
  window.localStorage.clear()
})

describe('resolveTheme', () => {
  it('follows the system preference in system mode', () => {
    expect(resolveTheme('system', true)).toBe('dark')
    expect(resolveTheme('system', false)).toBe('light')
  })

  it('ignores the system preference once the user picks a side', () => {
    expect(resolveTheme('dark', false)).toBe('dark')
    expect(resolveTheme('light', true)).toBe('light')
  })
})

describe('applyTheme', () => {
  it('writes the resolved theme onto the root element', () => {
    applyTheme('dark')
    expect(document.documentElement.dataset.theme).toBe('dark')

    applyTheme('light')
    expect(document.documentElement.dataset.theme).toBe('light')
  })

  it('caches the choice so the next launch can paint without a flash', () => {
    applyTheme('dark')
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark')
  })

  it('still applies the theme when storage is unavailable', () => {
    const setItem = window.localStorage.setItem
    window.localStorage.setItem = () => { throw new Error('quota') }
    try {
      expect(() => applyTheme('dark')).not.toThrow()
      expect(document.documentElement.dataset.theme).toBe('dark')
    } finally {
      window.localStorage.setItem = setItem
    }
  })
})

describe('applyTemplate', () => {
  it('writes the template onto the root element and caches the choice', () => {
    applyTemplate('sand')

    expect(document.documentElement.dataset.template).toBe('sand')
    expect(window.localStorage.getItem(TEMPLATE_STORAGE_KEY)).toBe('sand')
  })

  it('falls back to paper when nothing is stored or the value is unknown', () => {
    expect(readStoredTemplate()).toBe('paper')

    window.localStorage.setItem(TEMPLATE_STORAGE_KEY, 'slate')
    expect(readStoredTemplate()).toBe('slate')

    // 存储里可能是上个版本留下的值：不认识就退回默认，而不是原样写到 DOM 上。
    window.localStorage.setItem(TEMPLATE_STORAGE_KEY, '不存在的皮肤')
    expect(readStoredTemplate()).toBe('paper')
  })

  it('paints both axes before the first render so a refresh does not flash', () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'dark')
    window.localStorage.setItem(TEMPLATE_STORAGE_KEY, 'slate')

    bootstrapAppearance()

    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(document.documentElement.dataset.template).toBe('slate')
  })

  it('bootstraps to the default template with an empty storage', () => {
    bootstrapAppearance()

    expect(document.documentElement.dataset.template).toBe('paper')
  })

  it('offers a swatch and a hint for every template the picker shows', () => {
    // 样本色必须成套齐备：缺一个色块，选项卡片上就会出现一块空白。
    expect(TEMPLATE_OPTIONS.map(option => option.value)).toEqual(['paper', 'slate', 'sand'])
    for (const option of TEMPLATE_OPTIONS) {
      for (const mode of ['light', 'dark'] as const) {
        const swatch = option.swatch[mode]
        expect(swatch.bg).toMatch(/^#[0-9a-f]{6}$/i)
        expect(swatch.surface).toMatch(/^#[0-9a-f]{6}$/i)
        expect(swatch.accent).toMatch(/^#[0-9a-f]{6}$/i)
      }
      // 深浅两份必须真的不同：相同就意味着某一侧漏写了、会显示成另一侧。
      expect(option.swatch.light.bg).not.toBe(option.swatch.dark.bg)
      expect(option.hint).not.toBe('')
    }
  })
})