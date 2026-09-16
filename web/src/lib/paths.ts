/** 把绝对路径切成面包屑，每一级都得是**绝对**路径。
 *
 * 后端只认绝对路径，所以不能简单用 `join('…/…')`：Windows 上 `C:` 单独
 * 一级是"当前盘当前目录"，拿去列目录会跑偏；必须保留盘符后的反斜杠，
 * 让第一级就是 `C:\`。POSIX 同理，第一级要带上开头的 `/`。
 */
export function crumbs(path: string): { name: string; path: string }[] {
  const separator = path.includes('\\') ? '\\' : '/'
  const parts = path.split(/[\\/]+/).filter(Boolean)
  const trail: { name: string; path: string }[] = []
  let current = ''
  for (const part of parts) {
    if (!current) {
      current = /^[A-Za-z]:$/.test(part) ? part + separator : separator + part
    } else {
      current = current.endsWith(separator) ? current + part : `${current}${separator}${part}`
    }
    trail.push({ name: part, path: current })
  }
  return trail
}
