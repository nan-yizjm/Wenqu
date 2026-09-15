export type Theme = 'system' | 'light' | 'dark'

export const THEME_STORAGE_KEY = 'obsidian-rag-theme'

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