import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8765' } },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    // testing-library 的 `findBy*` / `waitFor` 默认等 1 秒。全量跑时 14 个文件抢
    // CPU，一次假的"元素没出现"就够毁掉一条真实断言的可信度——这条 3 秒的上限
    // 与其说是放宽，不如说是不让机器忙闲决定测试结果。真的是 bug 时它照样会挂。
    asyncUtilTimeout: 3000,
  },
})