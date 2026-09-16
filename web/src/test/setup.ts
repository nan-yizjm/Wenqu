import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// vitest 没开 globals，@testing-library/react 的自动清理不会注册。不显式卸载的话
// DOM 会跨用例累积，用 screen 查询时就会撞上"找到多个元素"。
afterEach(cleanup)

// jsdom 没有 matchMedia，主题相关的代码需要一个可控替身。
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  configurable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  }),
})