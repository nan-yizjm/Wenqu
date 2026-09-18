// 对「正在运行的交付版应用」做 UI 验收：真的去点，并断言点完发生了什么。
//
// 为什么需要它：组件测试只渲染组件本身，绕过了页面；此前正是因此漏掉了
// 「LibraryPage 没有挂载 FolderPicker」这种缺陷——组件测试全绿，点「选择」依然没反应。
// 这个脚本走的是打包后的真实产物：无头 Chrome 打开正在跑的便携版 → 点击 → 断言 → 截图。
//
// 用法：
//   1) 先让应用跑起来（例如 .\dist\ObsidianRAG\ObsidianRAG.exe）
//   2) node scripts/ui_acceptance.cjs [端口，默认 8765]
//   3) 退出码 0 表示全部断言通过；截图落在 UI_SHOT_DIR（默认系统临时目录）
//
// 只读安全性：脚本不会点「连接」「删除会话」等会改数据的控件，只做浏览与取消。
// 需要环境变量 CHROME_PATH 时用它指向 chrome.exe，否则按常见安装位置探测。
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');

function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    path.join(process.env.LOCALAPPDATA || '', 'Google', 'Chrome', 'Application', 'chrome.exe'),
    '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ].filter(Boolean);
  const found = candidates.find(p => { try { return fs.statSync(p).isFile(); } catch { return false; } });
  if (!found) throw new Error('找不到 chrome.exe，请用 CHROME_PATH 指定');
  return found;
}

const CHROME = findChrome();
const PORT = Number(process.argv[2] || 8765);
const APP = `http://127.0.0.1:${PORT}/`;
const DEBUG_PORT = Number(process.env.UI_DEBUG_PORT || 9335);
const OUT = process.env.UI_SHOT_DIR || path.join(os.tmpdir(), 'obsidian-rag-ui-acceptance');
const sleep = ms => new Promise(r => setTimeout(r, ms));

const results = [];
function check(name, passed, detail) {
  results.push({ name, passed, detail });
  console.log(`${passed ? 'PASS' : 'FAIL'}  ${name}${detail ? `  — ${detail}` : ''}`);
}

// Chrome 的 user-data-dir 是脚本自己建的临时目录，必须自己收掉。只 kill 不删会
// 在 TEMP 里每次留一个 50 MB 上下的 profile（实测 4 次运行留下 176 MB）。
// 删之前要等进程真的退出：Chrome 还占着的时候删，只会留下删不掉的半个目录。
let spawnedChrome = null;
let spawnedProfile = null;
async function cleanup() {
  if (!spawnedChrome) return;
  spawnedChrome.kill();
  await Promise.race([new Promise(r => spawnedChrome.once('exit', r)), sleep(3000)]);
  try { fs.rmSync(spawnedProfile, { recursive: true, force: true }); } catch { /* 留给系统清 */ }
  spawnedChrome = null;
}

class Session {
  constructor(ws) { this.ws = ws; this.id = 0; this.pending = new Map(); this.consoleErrors = []; }
  static async open(url) {
    const ws = new WebSocket(url);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
    const s = new Session(ws);
    ws.onmessage = ev => {
      const msg = JSON.parse(ev.data);
      if (msg.id && s.pending.has(msg.id)) {
        const { res, rej } = s.pending.get(msg.id); s.pending.delete(msg.id);
        msg.error ? rej(new Error(JSON.stringify(msg.error))) : res(msg.result);
      } else if (msg.method === 'Runtime.exceptionThrown') {
        s.consoleErrors.push(msg.params.exceptionDetails?.exception?.description || 'unknown');
      }
    };
    return s;
  }
  send(method, params) {
    const id = ++this.id;
    return new Promise((res, rej) => {
      this.pending.set(id, { res, rej });
      this.ws.send(JSON.stringify({ id, method, params: params || {} }));
    });
  }
  async eval(expression) {
    const r = await this.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) throw new Error('页面内 JS 抛错: ' + JSON.stringify(r.exceptionDetails.exception));
    return r.result.value;
  }
  async shot(name, w, h) {
    await this.send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile: false });
    await sleep(400);
    const r = await this.send('Page.captureScreenshot', { format: 'png' });
    fs.mkdirSync(OUT, { recursive: true });
    const file = path.join(OUT, name);
    fs.writeFileSync(file, Buffer.from(r.data, 'base64'));
    console.log(`      截图 -> ${file}`);
  }
}

const clickByText = (text, scope) => `(() => {
  const root = ${scope ? `document.querySelector('${scope}')` : 'document'};
  if (!root) return 'no-scope';
  const b = [...root.querySelectorAll('button')].find(x => x.textContent.trim() === ${JSON.stringify(text)});
  if (!b) return 'no-button';
  b.click(); return 'ok';
})()`;

(async () => {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'cdp-accept-'));
  const chrome = spawn(CHROME, [
    '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
    `--remote-debugging-port=${DEBUG_PORT}`, `--user-data-dir=${profile}`,
    '--window-size=1440,900', 'about:blank',
  ], { stdio: 'ignore' });
  spawnedChrome = chrome; spawnedProfile = profile;

  let targets = null;
  for (let i = 0; i < 50; i++) {
    try { targets = await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/list`).then(r => r.json()); if (targets.length) break; } catch { }
    await sleep(300);
  }
  if (!targets) throw new Error('Chrome 远程调试端口没起来');

  const s = await Session.open(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
  await s.send('Page.enable');
  await s.send('Runtime.enable');
  await s.send('Page.navigate', { url: APP });
  await sleep(3000);

  // ---- 1. 应用起来了 ----
  const title = await s.eval('document.title');
  check('应用页面已加载', title.includes('Obsidian RAG'), `title=${title}`);

  // ---- 2. 删除会话：每个会话都要有删除按钮 ----
  await s.eval(`(() => { const b=[...document.querySelectorAll('nav button')].find(x=>x.textContent.includes('知识问答')); if(b) b.click(); })()`);
  await sleep(2000);
  const chat = await s.eval(`({
    会话行数: document.querySelectorAll('.conversation-item').length,
    删除按钮数: document.querySelectorAll('.conversation-delete').length,
    删除按钮可点: [...document.querySelectorAll('.conversation-delete')].every(b => !b.disabled),
    有打开按钮: document.querySelectorAll('.conversation-open').length,
    选中态行数: document.querySelectorAll('.conversation-item.active, .conversation-open.active').length,
  })`);
  check('每个会话都有删除按钮', chat.会话行数 > 0 && chat.删除按钮数 === chat.会话行数, JSON.stringify(chat));
  check('删除按钮可点击且与「打开」并列', chat.删除按钮可点 && chat.有打开按钮 === chat.会话行数, JSON.stringify(chat));
  await s.shot('accept-chat-1440.png', 1440, 900);
  await s.shot('accept-chat-760.png', 760, 900);

  // ---- 3. 文件夹选择器：点「选择」必须真的打开 ----
  await s.eval(`(() => { const b=[...document.querySelectorAll('nav button')].find(x=>x.textContent.includes('资料库')); if(b) b.click(); })()`);
  await sleep(1800);
  const opened = await s.eval(clickByText('选择'));
  check('找到并点击了「选择」按钮', opened === 'ok', opened);
  await sleep(2200);

  const picker = await s.eval(`(() => {
    const m = document.querySelector('.folder-picker');
    if (!m) return { 出现: false };
    return { 出现: true,
      标题: (m.querySelector('h2') || {}).textContent,
      盘符: [...m.querySelectorAll('.picker-roots button')].map(x => x.textContent.trim()),
      面包屑: [...m.querySelectorAll('.picker-crumbs button')].map(x => x.textContent.trim()),
      当前路径: (m.querySelector('footer code') || {}).textContent,
      目录数: m.querySelectorAll('.picker-entry').length,
      前几个: [...m.querySelectorAll('.picker-entry')].slice(0, 5).map(x => x.textContent.trim()),
      确认按钮可用: !(m.querySelector('.picker-actions .primary') || {}).disabled,
    };
  })()`);
  check('点击「选择」打开了应用内选择器', picker.出现 === true, JSON.stringify(picker));
  if (picker.出现) {
    check('选择器列出了真实子目录', picker.目录数 > 0, `${picker.目录数} 个，前几个=${JSON.stringify(picker.前几个)}`);
    check('确认按钮在加载完成后可用', picker.确认按钮可用 === true, '');
  }
  await s.shot('accept-picker-1440.png', 1440, 900);

  // ---- 4. 逐级进入：路径与列表必须真的变 ----
  const before = picker.当前路径;
  const entered = await s.eval(`(() => {
    const e = document.querySelector('.picker-entry');
    if (!e) return 'no-entry';
    const name = e.textContent.trim();
    e.click(); return name;
  })()`);
  await sleep(1800);
  const after = await s.eval(`(() => {
    const m = document.querySelector('.folder-picker');
    return { 当前路径: (m.querySelector('footer code')||{}).textContent,
             目录数: m.querySelectorAll('.picker-entry').length };
  })()`);
  check('点进子目录后路径确实变了', before !== after.当前路径,
        `${before} -> ${after.当前路径}（进入 ${entered}）`);
  await s.shot('accept-picker-entered.png', 1440, 900);

  // ---- 5. 选中：路径回填输入框并关闭选择器 ----
  await s.eval(clickByText('选中此文件夹'));
  await sleep(1500);
  const picked = await s.eval(`({
    选择器还在: !!document.querySelector('.folder-picker'),
    输入框的值: (document.querySelector('.folder-row input') || {}).value,
  })`);
  check('选中后路径回填到输入框', picked.输入框的值 === after.当前路径,
        `输入框=${picked.输入框的值}`);
  check('选中后选择器关闭', picked.选择器还在 === false, '');

  // ---- 6. 取消：不改动输入框 ----
  const kept = picked.输入框的值;
  await s.eval(clickByText('选择'));
  await sleep(1800);
  const reopened = await s.eval(`!!document.querySelector('.folder-picker')`);
  check('再次点击「选择」仍能打开', reopened === true, '');
  await s.eval(clickByText('取消'));
  await sleep(1200);
  const cancelled = await s.eval(`({
    选择器还在: !!document.querySelector('.folder-picker'),
    输入框的值: (document.querySelector('.folder-row input') || {}).value,
  })`);
  check('取消后输入框保持原样', cancelled.输入框的值 === kept && cancelled.选择器还在 === false,
        `输入框=${cancelled.输入框的值}`);

  // ---- 7. 窄屏 ----
  await s.shot('accept-library-760.png', 760, 900);
  const overflow = await s.eval(`({
    文档宽度: document.documentElement.scrollWidth,
    视口宽度: window.innerWidth,
    侧栏文字可见: [...document.querySelectorAll('.nav-label')].some(x => x.offsetParent !== null),
  })`);
  check('760px 下没有横向溢出', overflow.文档宽度 <= overflow.视口宽度 + 1, JSON.stringify(overflow));
  check('760px 下侧栏折成图标条（文字隐藏）', overflow.侧栏文字可见 === false, '');

  // ---- 8. 设置页的记忆开关（只切换开关，不点「保存设置」，不改任何数据） ----
  await s.eval(`(() => { const b=[...document.querySelectorAll('nav button')].find(x=>x.textContent.includes('设置')); if(b) b.click(); })()`);
  await sleep(2000);
  const memoryCard = await s.eval(`(() => {
    const card = [...document.querySelectorAll('.card')].find(c => (c.querySelector('h3')||{}).textContent === '记忆');
    if (!card) return { 卡片: false };
    const buttons = [...card.querySelectorAll('.segmented button')];
    return { 卡片: true,
      选项: buttons.map(b => b.textContent.trim()),
      选中: buttons.filter(b => b.className.includes('active')).map(b => b.textContent.trim()),
      状态行: (card.querySelector('.status-line') || {}).textContent };
  })()`);
  check('设置页有记忆卡片且选项为开启/关闭', memoryCard.卡片 === true
        && JSON.stringify(memoryCard.选项) === JSON.stringify(['开启', '关闭']), JSON.stringify(memoryCard));
  // 出厂默认必须是关，而且界面要把"关"说清楚——这是这个接缝唯一的对外承诺。
  check('记忆默认关闭且状态行如实说明',
        JSON.stringify(memoryCard.选中) === JSON.stringify(['关闭'])
        && /记忆已关闭/.test(memoryCard.状态行 || ''), JSON.stringify(memoryCard));
  await s.shot('accept-settings.png', 1440, 900);

  // ---- 9. 设置页的联网开关，以及"开启前必须看过告知" ----
  // 全程只切换开关、只看告知，不保存——保存会把 web_enabled 写进设置，那是在改
  // 用户的数据。这里要验的是"点开启会不会先摆出告知"，不是"能不能真的联网"。
  const webCard = await s.eval(`(() => {
    const card = [...document.querySelectorAll('.card')].find(c => (c.querySelector('h3')||{}).textContent === '联网补充');
    if (!card) return { 卡片: false };
    const buttons = [...card.querySelectorAll('.segmented button')];
    return { 卡片: true,
      选项: buttons.map(b => b.textContent.trim()),
      选中: buttons.filter(b => b.className.includes('active')).map(b => b.textContent.trim()),
      状态行: (card.querySelector('.status-line') || {}).textContent,
      告知可见: !!card.querySelector('.web-disclosure') };
  })()`);
  check('设置页有联网卡片且选项为开启/关闭', webCard.卡片 === true
        && JSON.stringify(webCard.选项) === JSON.stringify(['开启', '关闭']), JSON.stringify(webCard));
  check('联网默认关闭且状态行说明不会发出请求',
        JSON.stringify(webCard.选中) === JSON.stringify(['关闭'])
        && /不会发出任何网络请求/.test(webCard.状态行 || ''), JSON.stringify(webCard));
  // 未确认告知之前不该先看到告知栏——它只在点「开启」之后出现。
  check('未开启时不显示告知栏', webCard.告知可见 === false, JSON.stringify(webCard));

  await s.eval(`(() => {
    const card = [...document.querySelectorAll('.card')].find(c => (c.querySelector('h3')||{}).textContent === '联网补充');
    const b = [...card.querySelectorAll('.segmented button')].find(x => x.textContent.trim() === '开启');
    if (b) b.click();
  })()`);
  await sleep(500);
  const disclosure = await s.eval(`(() => {
    const box = document.querySelector('.web-disclosure');
    if (!box) return { 可见: false };
    const card = box.closest('.card');
    const buttons = [...card.querySelectorAll('.segmented button')];
    return { 可见: true, 文字: box.textContent,
      仍选中: buttons.filter(b => b.className.includes('active')).map(b => b.textContent.trim()) };
  })()`);
  check('点「开启」先摆出告知栏', disclosure.可见 === true, JSON.stringify(disclosure));
  // 告知必须把"发什么"和"不发什么"都说清，只说一句"会联网"等于没说。
  check('告知写明只发问题本身、不发笔记正文',
        /问题本身/.test(disclosure.文字 || '') && /不会发送/.test(disclosure.文字 || '')
        && /笔记正文/.test(disclosure.文字 || ''), JSON.stringify(disclosure));
  // 还没确认，开关不能自己变成打开——否则用户点保存会撞上后端的 422，而他并不
  // 知道自己在确认什么。
  check('未确认前开关不跳到开启', JSON.stringify(disclosure.仍选中) === JSON.stringify(['关闭']),
        JSON.stringify(disclosure));
  // 告知栏在设置页下方，截图前先滚到它——否则截图里看不到这次验收到底看见了什么。
  await s.eval(`(() => {
    const box = document.querySelector('.web-disclosure');
    if (box) box.scrollIntoView({ block: 'center' });
  })()`);
  await sleep(400);
  await s.shot('accept-settings-web.png', 1440, 900);

  // 撤回这一步的界面变化：点「取消」，回到出厂状态（全程没有保存，所以库里没变）。
  await s.eval(`(() => {
    const box = document.querySelector('.web-disclosure');
    if (!box) return;
    const b = [...box.querySelectorAll('button')].find(x => x.textContent.trim() === '取消');
    if (b) b.click();
  })()`);
  await sleep(300);
  check('取消后告知栏收起', await s.eval(`!document.querySelector('.web-disclosure')`) === true, '');

  const errors = s.consoleErrors;
  check('页面无未捕获异常', errors.length === 0, errors.slice(0, 3).join(' | '));

  await cleanup();
  const failed = results.filter(r => !r.passed);
  console.log(`\n合计 ${results.length} 项，通过 ${results.length - failed.length}，失败 ${failed.length}`);
  if (failed.length) { console.log('失败项：'); failed.forEach(f => console.log('  - ' + f.name)); process.exit(1); }
})().catch(async e => { console.error('脚本失败:', e.message); await cleanup(); process.exit(2); });
