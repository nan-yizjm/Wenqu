import type { FeedbackKind } from '../api'

/** 反馈的四种类别。问答页当成一排按钮（`hint` 作 title），收藏页当筛选项。 */
export const FEEDBACK_KINDS: { kind: FeedbackKind; label: string; hint: string }[] = [
  { kind: 'helpful', label: '有帮助', hint: '答案和引用都对' },
  { kind: 'missing', label: '有遗漏', hint: '资料里有但没答到' },
  { kind: 'citation_wrong', label: '引用不对', hint: '[S1] 指错了位置' },
  { kind: 'answer_wrong', label: '回答不对', hint: '与原文不符' },
]