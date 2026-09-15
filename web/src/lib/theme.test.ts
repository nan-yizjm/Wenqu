import { afterEach, describe, expect, it } from 'vitest'
import { applyTheme, resolveTheme, THEME_STORAGE_KEY } from './theme'

afterEach(() => {
  delete document.documentElement.dataset.theme
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