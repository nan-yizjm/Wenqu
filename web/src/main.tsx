import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { ErrorBoundary } from './components/ErrorBoundary'
import { bootstrapAppearance } from './lib/theme'
import './styles.css'

// 在首次 render 之前把主题与模板摆好：它们真正的来源是这次启动后才会拿到的
// settings，不先按本机存的上一份摆好，每次刷新都会从默认皮肤闪一下。
bootstrapAppearance()

createRoot(document.getElementById('root')!).render(
  <StrictMode><ErrorBoundary><App /></ErrorBoundary></StrictMode>)
