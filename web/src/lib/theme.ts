import { useEffect, useState } from 'react'

export type Theme = 'system' | 'light' | 'dark'
export type Template = 'paper' | 'slate' | 'sand'

export const THEME_STORAGE_KEY = 'obsidian-rag-theme'
export const TEMPLATE_STORAGE_KEY = 'obsidian-rag-template'
export const DEFAULT_TEMPLATE: Template = 'paper'

function prefersDark(): boolean {
  return window.matchMedia('(prefers-color-scheme: dark)').matches
}

export function resolveTheme(theme: Theme, systemDark: boolean = prefersDark()): 'light' | 'dark' {
  return theme === 'system' ? (systemDark ? 'dark' : 'light') : theme
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = resolveTheme(theme)
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme)
  } catch {
    // 隐私模式禁用 localStorage 时忽略：下次启动会退回默认的「跟随系统」。
  }
}

/**
 * 换一套皮肤。模板只改 CSS 变量的值（色板、边框、圆角刻度），组件标记不动——
 * 所以它不需要 React 重渲染，写进 `data-template` 由 CSS 级联完成。设置页把选择
 * 存进 settings 表（跨设备、跟数据走），这里顺手写一份 localStorage：
 * 首屏渲染在 `setup` 请求回来之前，那份存值让刷新不闪。
 */
export function applyTemplate(template: Template): void {
  document.documentElement.dataset.template = template
  try {
    window.localStorage.setItem(TEMPLATE_STORAGE_KEY, template)
  } catch {
    // 同 applyTheme：存不下就退回默认模板，不是错误。
  }
}

export function readStoredTemplate(): Template {
  try {
    const stored = window.localStorage.getItem(TEMPLATE_STORAGE_KEY)
    if (stored === 'paper' || stored === 'slate' || stored === 'sand') return stored
  } catch {
    // 忽略：读不到就用默认。
  }
  return DEFAULT_TEMPLATE
}

/**
 * 首屏先按 localStorage 把外观摆好（`main.tsx` 在 render 之前调用）。
 *
 * 为什么值得单独一步：主题与模板的真正来源是服务端的 settings，而那份数据要等
 * 一次网络请求。在那之前页面已经画出来了——不先摆好的话，每次刷新都会从默认
 * 皮肤闪一下再跳到你选的那套。localStorage 是这台机器上的上一次选择，
 * 它与服务端不一致的窗口只存在于"刚在别处改过设置"的那一次。
 */
export function bootstrapAppearance(): void {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY)
    if (stored === 'system' || stored === 'light' || stored === 'dark') applyTheme(stored)
  } catch {
    // 忽略：跟随系统即可。
  }
  applyTemplate(readStoredTemplate())
}

export function watchSystemTheme(theme: Theme): () => void {
  if (theme !== 'system') return () => {}
  const query = window.matchMedia('(prefers-color-scheme: dark)')
  const listener = () => {
    document.documentElement.dataset.theme = resolveTheme('system')
  }
  query.addEventListener('change', listener)
  return () => query.removeEventListener('change', listener)
}

export const THEME_OPTIONS: { value: Theme; label: string }[] = [
  { value: 'system', label: '跟随系统' },
  { value: 'light', label: '浅色' },
  { value: 'dark', label: '深色' },
]

/**
 * 三套模板。`swatch` 是设置页上的小色板样本——它必须显示**这套模板**的颜色，
 * 而页面上生效的 CSS 变量只能是当前模板的值，所以样本色以数据形式写在这里，
 * 由组件内联进样式（design token 门禁只守 `styles.css`，这里不是它的辖区）。
 * 浅/深各备一份：深色界面里显示浅色样本，用户看到的就不是"切过去的样子"。
 */
export const TEMPLATE_OPTIONS: {
  value: Template; label: string; hint: string
  swatch: { light: Swatch; dark: Swatch }
}[] = [
  { value: 'paper', label: '纸墨', hint: '米白纸感 · 深绿强调 · 圆角柔和',
    swatch: { light: { bg: '#f5f4ef', surface: '#ffffff', accent: '#244d3e' },
              dark: { bg: '#121416', surface: '#202427', accent: '#7cc3a1' } } },
  { value: 'slate', label: '石板', hint: '冷灰蓝 · 利落边框 · 紧凑圆角',
    swatch: { light: { bg: '#f4f6f9', surface: '#ffffff', accent: '#2d4a7c' },
              dark: { bg: '#0f1319', surface: '#1b222b', accent: '#7ea6e0' } } },
  { value: 'sand', label: '暖砂', hint: '暖米底 · 赤陶强调 · 圆角圆润',
    swatch: { light: { bg: '#faf5ee', surface: '#fffdf9', accent: '#9a5b32' },
              dark: { bg: '#171310', surface: '#251f19', accent: '#d99a6c' } } },
]

export type Swatch = { bg: string; surface: string; accent: string }

/**
 * 当前**实际**生效的明暗（`system` 会解析成 light/dark，并跟随系统变化重算）。
 *
 * 设置页的色板样本要用它：界面上是深色时，样本也该显示各模板的深色版本——
 * 否则用户看到的样本和"切过去的样子"是两回事。
 */
export function useResolvedTheme(theme: Theme): 'light' | 'dark' {
  const [systemDark, setSystemDark] = useState(prefersDark)
  useEffect(() => {
    const query = window.matchMedia('(prefers-color-scheme: dark)')
    const listener = () => setSystemDark(query.matches)
    query.addEventListener('change', listener)
    return () => query.removeEventListener('change', listener)
  }, [])
  return resolveTheme(theme, systemDark)
}