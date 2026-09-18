import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { BatchResult } from '../api'
import { BatchDeleteBar } from './BatchDelete'
// 这一组测试钉的是批量删除条自己的行为：两段式确认不用 window.confirm、
// 结果必须把 skipped 说出来（只报"已删除 N 项"会把不完整的删除说成成功的）。
const base = {
  ids: ['a'],
  allIds: ['a', 'b'],
  heading: '删除 1 项？',
  lines: ['这是后果说明。'],
  allSelected: false,
  onToggleAll: vi.fn(),
  onCancel: vi.fn(),
  onConfirm: vi.fn(),
  onDone: vi.fn(),
}

const setup = (over: Partial<typeof base> = {}) => {
  const props = { ...base, ...over }
  render(<BatchDeleteBar {...props} />)
  return props
}

beforeEach(() => {
  for (const key of ['onToggleAll', 'onCancel', 'onConfirm', 'onDone'] as const) {
    base[key].mockReset()
  }
})

describe('BatchDeleteBar 的两段式确认', () => {
  it('没有选中任何项时删除按钮不可点', () => {
    setup({ ids: [] })

    expect(screen.getByText('已选 0 项')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '删除' })).toBeDisabled()
  })

  it('点删除先给确认框，「再想想」会退回去而不发请求', async () => {
    const props = setup()

    fireEvent.click(screen.getByRole('button', { name: '删除' }))
    expect(await screen.findByText('删除 1 项？')).toBeInTheDocument()
    expect(screen.getByText('这是后果说明。')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: '再想想' }))
    expect(screen.queryByText('删除 1 项？')).not.toBeInTheDocument()
    expect(props.onConfirm).not.toHaveBeenCalled()
  })

  it('确认后如实展示结果——skipped 的每一条都要说清是谁、为什么', async () => {
    const result: BatchResult = {
      deleted: 1, files_removed: 2,
      skipped: [{ id: 'busy-1', label: '还在生成的产出', code: 'busy', reason: '这份产出还在生成，请先停止再删除。' }],
    }
    const props = setup({ onConfirm: vi.fn().mockResolvedValue(result) })

    fireEvent.click(screen.getByRole('button', { name: '删除' }))
    fireEvent.click(await screen.findByRole('button', { name: '删除' }))

    expect(await screen.findByText(/已删除 1 项/)).toBeInTheDocument()
    expect(screen.getByText(/一并清理导出文件 2 个/)).toBeInTheDocument()
    expect(screen.getByText('以下 1 项未删除：')).toBeInTheDocument()
    expect(screen.getByText(/还在生成的产出：这份产出还在生成/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '完成' }))
    expect(props.onDone).toHaveBeenCalled()
  })

  it('请求失败时回到确认框并显示错误，不假装删成功', async () => {
    const props = setup({ onConfirm: vi.fn().mockRejectedValue(new Error('后端没有响应')) })

    fireEvent.click(screen.getByRole('button', { name: '删除' }))
    fireEvent.click(await screen.findByRole('button', { name: '删除' }))

    expect(await screen.findByText('后端没有响应')).toBeInTheDocument()
    expect(screen.queryByText(/已删除/)).not.toBeInTheDocument()
    expect(props.onDone).not.toHaveBeenCalled()
  })

  it('全选框把状态交给父页面，自己不持有第二份选中态', () => {
    const props = setup()

    const checkbox = screen.getByRole('checkbox')
    expect(checkbox).not.toBeChecked()
    fireEvent.click(checkbox)
    expect(props.onToggleAll).toHaveBeenCalled()
  })

  it('删除进行中按钮禁用，防止双击发两次请求', async () => {
    let resolve!: (value: BatchResult) => void
    setup({ ids: ['a', 'b'], heading: '删除 2 项？',
      onConfirm: vi.fn().mockImplementation(() => new Promise<BatchResult>(r => { resolve = r })) })

    fireEvent.click(screen.getByRole('button', { name: '删除' }))
    fireEvent.click(await screen.findByRole('button', { name: '删除' }))
    expect(screen.getByRole('button', { name: '正在删除…' })).toBeDisabled()

    resolve({ deleted: 2, skipped: [] })
    await waitFor(() => expect(screen.getByText('已删除 2 项')).toBeInTheDocument())
  })
})
