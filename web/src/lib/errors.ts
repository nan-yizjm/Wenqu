// 后端在缺少可读 message 时只回一个机器码，直接抛给用户等于什么都不说。
// 这里把用户真会撞到的码翻译成「发生了什么 + 下一步做什么」。
const HINTS: Record<string, string> = {
  local_access_required: '工作台只允许本机访问，请从 127.0.0.1 打开它。',
  local_origin_required: '请求不是从工作台页面发出的，已被拒绝。请刷新本页重试。',
  database_recovery_required: '数据库升级没有完成。请先恢复升级前的自动备份。',
  restart_required: '需要退出并重新打开工作台才能继续。',
  unknown_setting: '当前版本不认识这个设置项，请更新到最新版本。',
  invalid_folder: '这个文件夹读不到，请确认路径存在、可读，并且里面有 Markdown 文件。',
  invalid_document: '这个文件暂时不能导入。目前支持 Markdown、PDF 与 Jupyter Notebook。',
  invalid_backup: '备份文件没有通过校验，请选择由本工作台导出的备份压缩包。',
  invalid_request: '提交的内容没有通过校验，请检查填写格式后重试。',
  question_required: '请先输入问题。',
  source_not_found: '找不到这段原文，它可能已经随资料一起被移除。',
  source_unavailable: '这段原文的版本已经不在工作台里了。',
  favorite_unavailable: '这条回答没有可用的来源快照，无法收藏。',
  retry_unavailable: '这份资料当前不能重试，请先重新导入。',
  backup_not_found: '找不到这个备份文件。',
  document_not_found: '这份资料已经不在工作台里了，请刷新页面。',
  conversation_not_found: '这个会话已经不存在了，请新建一个会话。',
  conversation_busy: '这个会话正在生成回答，等它结束或点「停止」之后才能删除。',
  favorite_not_found: '这条收藏已经被删除了。',
  message_not_found: '这条消息已经不存在了，请刷新页面。',
}

export function describeFailure(status: number, body: unknown): string {
  const payload = (body ?? {}) as { message?: unknown; error?: unknown }
  if (typeof payload.message === 'string' && payload.message) return payload.message
  const code = typeof payload.error === 'string' ? payload.error : ''
  if (code && HINTS[code]) return HINTS[code]
  const suffix = code ? `（${code}）` : ''
  return `本地服务返回 ${status}${suffix}。可以重试；若反复出现，请在「设置」里下载脱敏诊断。`
}

export const OFFLINE_HINT = '连不上本地服务：它可能已经退出。请重新打开工作台，或刷新本页重试。'