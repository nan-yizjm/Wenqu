import type { ReactNode } from 'react'

export function EmptyState({ glyph, title, tall, children }: {
  glyph: string
  title: string
  tall?: boolean
  children?: ReactNode
}) {
  return <div className={tall ? 'empty-state tall' : 'empty-state'}>
    <span className="empty-glyph" aria-hidden="true">{glyph}</span>
    <h2>{title}</h2>
    <div className="empty-body">{children}</div>
  </div>
}

export function SkeletonLines({ count = 3, className = '' }: { count?: number; className?: string }) {
  return <div className={`skeleton-lines ${className}`.trim()} role="status" aria-label="正在加载">
    {Array.from({ length: count }, (_, index) =>
      <span key={index} className="skeleton" style={{ width: `${100 - (index % 3) * 12}%` }} />)}
  </div>
}