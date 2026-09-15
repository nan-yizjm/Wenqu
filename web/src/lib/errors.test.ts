import { describe, expect, it } from 'vitest'
import { describeFailure, OFFLINE_HINT } from './errors'

describe('describeFailure', () => {
  it('prefers the server message when there is one', () => {
    expect(describeFailure(422, { message: '路径不存在', error: 'invalid_folder' })).toBe('路径不存在')
  })

  it('ignores an empty message and falls back to the code hint', () => {
    expect(describeFailure(409, { message: '', error: 'restart_required' })).not.toBe('')
  })

  it('maps a known code to its hint', () => {
    expect(describeFailure(404, { error: 'source_not_found' })).not.toContain('本地服务返回')
  })

  it('keeps the raw code when nothing matches, so a bug report is actionable', () => {
    const text = describeFailure(500, { error: 'brand_new_code' })
    expect(text).toContain('500')
    expect(text).toContain('brand_new_code')
  })

  it('still says something useful when the body is not JSON', () => {
    expect(describeFailure(500, null)).toContain('500')
    expect(describeFailure(500, 'gateway timeout')).toContain('500')
  })

  it('never returns an empty string', () => {
    expect(describeFailure(400, {})).not.toBe('')
    expect(OFFLINE_HINT.length).toBeGreaterThan(0)
  })
})