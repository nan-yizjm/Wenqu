import { describe, expect, it } from 'vitest'
import { crumbs } from './paths'

describe('crumbs', () => {
  it('keeps the backslash after the drive so each level is absolute', () => {
    expect(crumbs('C:\\Users\\zjm\\Desktop')).toEqual([
      { name: 'C:', path: 'C:\\' },
      { name: 'Users', path: 'C:\\Users' },
      { name: 'zjm', path: 'C:\\Users\\zjm' },
      { name: 'Desktop', path: 'C:\\Users\\zjm\\Desktop' },
    ])
  })

  it('keeps the leading slash on posix paths', () => {
    expect(crumbs('/home/zjm/notes')).toEqual([
      { name: 'home', path: '/home' },
      { name: 'zjm', path: '/home/zjm' },
      { name: 'notes', path: '/home/zjm/notes' },
    ])
  })

  it('collapses repeated separators instead of emitting an empty level', () => {
    expect(crumbs('C:\\\\Users\\\\Desktop').map(item => item.name))
      .toEqual(['C:', 'Users', 'Desktop'])
  })

  it('has no level for a bare root', () => {
    expect(crumbs('/')).toEqual([])
  })
})
