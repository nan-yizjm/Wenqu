import { describe, expect, it } from 'vitest'
import { citationLabel, sanitizeUrl } from './markdown'

describe('sanitizeUrl', () => {
  it('keeps http and https links', () => {
    expect(sanitizeUrl('https://example.com/a')).toBe('https://example.com/a')
    expect(sanitizeUrl('http://example.com/a')).toBe('http://example.com/a')
  })

  it('keeps in-page citation anchors', () => {
    expect(sanitizeUrl('#cite-S3')).toBe('#cite-S3')
  })

  it('drops everything else', () => {
    for (const url of [
      'javascript:alert(1)',
      'JavaScript:alert(1)',
      'data:text/html,<script>alert(1)</script>',
      'vbscript:x',
      'file:///C:/Windows/System32/',
      '//evil.example.com/a',
      '/api/v1/system/shutdown',
      'mailto:a@b.c',
      undefined,
      '',
    ]) {
      expect(sanitizeUrl(url)).toBe('')
    }
  })
})

describe('citationLabel', () => {
  it('reads the label out of a citation anchor', () => {
    expect(citationLabel('#cite-S12')).toBe('S12')
  })

  it('ignores anything that is not a citation anchor', () => {
    expect(citationLabel(undefined)).toBeNull()
    expect(citationLabel('https://example.com/#cite-S1')).toBeNull()
    expect(citationLabel('#other')).toBeNull()
  })
})