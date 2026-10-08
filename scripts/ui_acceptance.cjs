// 对「正在运行的交付版应用」做 UI 验收：真的去点，并断言点完发生了什么。
//
// 为什么需要它：组件测试只渲染组件本身，绕过了页面；此前正是因此漏掉了
// 「LibraryPage 没有挂载 FolderPicker」这种缺陷——组件测试全绿，点「选择」依然没反应。
// 这个脚本走的是打包后的真实产物：无头 Chrome 打开正在跑的便携版 → 点击 → 断言 → 截图。
//
// 用法：
//   1) 先让应用跑起来（例如 .\dist\Wenqu\Wenqu.exe）
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
  check('应用页面已加载', title.includes('Wenqu'), `title=${title}`);

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

  // ---- 10. 产出页：表单、诚实分数，以及能自己清理 ----
  // 这一段会**真的生成一份思维导图**（会写一条产出记录），所以最后一步把它删掉，
  // 让跑完之后库里和跑之前一样。选思维导图是因为它零模型调用，不依赖本机有没有
  // 配置生成模型——这样这条验收在任何机器上都能跑。
  await s.eval(`(() => { const b=[...document.querySelectorAll('nav button')].find(x=>x.textContent.includes('产出')); if(b) b.click(); })()`);
  await sleep(1800);
  const studio = await s.eval(`(() => ({
    页面: !!document.querySelector('.studio-page'),
    主题输入: !!document.querySelector('.studio-form input'),
    类型选项: [...document.querySelectorAll('.studio-form .segmented button')].map(b=>b.textContent.trim()),
    有生成按钮: [...document.querySelectorAll('.studio-form button')].some(b=>b.textContent.trim()==='开始生成'),
  }))()`);
  check('产出页可打开且表单齐备', studio.页面 === true && studio.主题输入 === true
        && studio.有生成按钮 === true, JSON.stringify(studio));
  check('产出类型只有指南与思维导图',
        JSON.stringify(studio.类型选项) === JSON.stringify(['学习指南', '思维导图']),
        JSON.stringify(studio));
  await s.shot('accept-studio.png', 1440, 900);

  // 空主题不该开工：既不该发请求，也不该留下一条空记录。
  await s.eval(clickByText('开始生成', '.studio-form'));
  await sleep(500);
  check('空主题被拦下并给了明确说法',
        await s.eval(`[...document.querySelectorAll('.studio-actions .hint')]
          .some(x => x.textContent.includes('请先填一个主题'))`) === true, '');

  // 主题取库里第一份可用资料的标题：这样思维导图一定会命中片段，走到"建树"那条
  // 分支。用一个库里没有的词只能测到失败路径——失败路径也要能过，但那是另一条。
  const docTopic = await s.eval(`(async () => {
    const list = await (await fetch('/api/v1/documents')).json();
    const ready = (list.documents || []).find(x => x.status === 'ready');
    return ready ? ready.display_name.replace(/\\.md$/i, '') : '知识工作台';
  })()`);
  // React 受控输入要走过原生 setter，直接改 value 不会触发 onChange。
  await s.eval(`(() => {
    const input = document.querySelector('.studio-form input');
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    setter.call(input, ${JSON.stringify(docTopic)});
    input.dispatchEvent(new Event('input', { bubbles: true }));
  })()`);
  await s.eval(clickByText('思维导图', '.studio-form'));
  await sleep(300);
  const chosenKind = await s.eval(`(() => {
    const b = [...document.querySelectorAll('.studio-form .segmented button')].find(x => x.className.includes('active'));
    return b ? b.textContent.trim() : '';
  })()`);
  check('可以切到思维导图', chosenKind === '思维导图', chosenKind);

  await s.eval(clickByText('开始生成', '.studio-form'));
  await sleep(3500);
  const outcome = await s.eval(`(() => ({
    导图树: !!document.querySelector('[data-testid="mindmap-view"]'),
    节点数: document.querySelectorAll('[data-testid="mindmap-node"]').length,
    命中率卡片: !!document.querySelector('[data-testid="backlink-report"]'),
    说明: (document.querySelector('.studio-notice') || {}).textContent || '',
  }))()`);
  // 资料库为空时正确行为是明确失败，而不是产出一份空导图——两种情况都算通过，
  // 但"什么都不发生"不算。
  check('思维导图要么建出树、要么明确说明失败',
        outcome.导图树 ? outcome.节点数 > 0 : /没有找到|失败/.test(outcome.说明),
        JSON.stringify(outcome));
  // 这一条是这一页最重要的界面承诺：导图的覆盖是构造出来的，绝不能和指南的命中率
  // 并排显示（一个必然接近 1 的数字会让人以为两者可比）。
  check('思维导图不显示命中率卡片', outcome.命中率卡片 === false, JSON.stringify(outcome));
  await s.shot('accept-studio-mindmap.png', 1440, 900);

  // ---- 11. 信息图导出：点一下，要么出一张真图，要么给出 HTML 退路 ----
  // 导出器对指南和思维导图都成立，所以复用上一步刚生成的那份产出——这一步于是也不
  // 依赖本机有没有配置生成模型，验收在任何机器上都能跑。
  //
  // 判据落在「点完之后页面上真的多出一张能加载的图」：信息图曾经的风险正是
  // 「组件测试全绿、按钮点下去什么也没发生」，只有真点真加载才抓得到。
  // 上一步明确失败（资料库为空）时没有产出可导，跳过这一段而不是误判为通过。
  if (outcome.导图树) {
    await s.eval(`(() => {
      const b = document.querySelector('[data-testid="infographic-export"]');
      if (b) b.scrollIntoView({ block: 'center' });
    })()`);
    await sleep(300);
    const clicked = await s.eval(`(() => {
      const b = document.querySelector('[data-testid="infographic-export"]');
      if (!b) return 'no-button';
      b.click(); return 'ok';
    })()`);
    check('产出详情里有「导出信息图」按钮', clicked === 'ok', clicked);

    // 渲染要起一次真浏览器、再把 PNG 取回来，秒级；给它 20 秒。
    let info = { 图: false, 图已加载: false, 降级: false };
    for (let i = 0; i < 40; i++) {
      info = await s.eval(`(() => {
        const img = document.querySelector('[data-testid="infographic-image"]');
        const degraded = document.querySelector('[data-testid="infographic-degraded"]');
        return {
          图: !!img,
          图已加载: !!img && img.complete && img.naturalWidth > 0,
          图宽: img ? img.naturalWidth : 0,
          图高: img ? img.naturalHeight : 0,
          降级: !!degraded,
          耗时: (document.querySelector('[data-testid="infographic-milliseconds"]') || {}).textContent || '',
          有下载链接: !!document.querySelector('[data-testid="infographic-download"]'),
          尺寸不符警告: document.querySelectorAll('.studio-infographic .error').length,
          说明: degraded ? degraded.textContent : '',
        };
      })()`);
      if ((info.图 && info.图已加载) || info.降级) break;
      await sleep(500);
    }
    // 两种结果都算通过：本机没有 Edge/Chrome 时正确行为是降级并留一份 HTML 给用户，
    // 而不是让整条验收挂掉。"什么都不发生"才不算。
    check('信息图要么出图、要么给出 HTML 退路',
          (info.图 && info.图已加载) || info.降级, JSON.stringify(info));

    if (info.图 && info.图已加载) {
      // 浏览器解出了尺寸，说明这张图是真的被取回并解码了，不是个坏掉的 <img>。
      check('出的是真图（浏览器解出了尺寸）', info.图宽 > 0 && info.图高 > 0,
            `${info.图宽}×${info.图高}`);
      check('界面上如实写出渲染耗时',
            /毫秒/.test(info.耗时) && Number(info.耗时.replace(/\D/g, '')) > 0, info.耗时);
      check('出图后给出下载入口', info.有下载链接 === true, '');
      // 版面高度是算出来的：截图的窗口尺寸给多少就该出多少像素。不符时页面会自己
      // 挂一条警告——验收要求它不出现，因为出警告意味着排版和预期已经对不上了。
      check('图片尺寸与版面算术一致（页面没挂尺寸警告）',
            info.尺寸不符警告 === 0, `警告数=${info.尺寸不符警告}`);
      // 内联那张图验的是读图路由，这里再验一次带 Content-Disposition 的下载路由。
      const download = await s.eval(`(async () => {
        const a = document.querySelector('[data-testid="infographic-download"]');
        if (!a) return { 状态: 0 };
        const r = await fetch(a.getAttribute('href'));
        const blob = await r.blob();
        return { 状态: r.status, 类型: r.headers.get('content-type') || '', 大小: blob.size };
      })()`);
      check('下载链接能取回一张 PNG', download.状态 === 200
            && String(download.类型).includes('image/png') && download.大小 > 1000,
            JSON.stringify(download));
    } else if (info.降级) {
      // 浏览器不可用不是功能故障：HTML 已经导出，用户手里有退路。要验的是页面把
      // 「为什么」和「怎么办」都说清楚了，并且指得出那份 HTML 在哪。
      check('降级时说明了原因并给出 HTML 退路',
            /没能渲染成图片/.test(info.说明) && /HTML/.test(info.说明)
            && /浏览器/.test(info.说明), JSON.stringify(info.说明));
      check('降级时不显示一张不存在的图', info.图 === false, '');
    }
    await s.shot('accept-studio-infographic.png', 1440, 900);
  }

  // 收尾：把自己造出来的那条记录删掉。删除会弹原生 confirm，而无头 Chrome 下它会
  // 阻塞脚本求值，所以先把确认短路掉——这是测试钩子，不是被测行为。
  //
  // 数条数之前要**等列表渲染**：在一个此前没有任何产出的数据根上，刚生成的那条要等
  // 前端重绘出来，抢在前面取数会数到 0，断言 `remaining === listed - 1` 就成了
  // `0 === -1` —— 一次假 FAIL（实测在真实数据根上挂过，脚本判断从"删不掉"变成
  // "数早了"）。删除之后同样要等它重绘再判定少了一条。
  let listed = 0;
  for (let i = 0; i < 20; i++) {
    listed = await s.eval(`document.querySelectorAll('.studio-list button').length`);
    if (listed > 0) break;
    await sleep(300);
  }
  if (listed > 0) {
    await s.eval(`window.confirm = () => true`);
    await s.eval(`(() => { const b = document.querySelector('.studio-list button'); if (b) b.click(); })()`);
    await sleep(1200);
    await s.eval(clickByText('删除', '.studio-detail-head'));
    let remaining = listed;
    for (let i = 0; i < 20; i++) {
      remaining = await s.eval(`document.querySelectorAll('.studio-list button').length`);
      if (remaining < listed) break;
      await sleep(300);
    }
    // 失败时把列表里的标题一起带上：只看到两个数字对不上，没法判断是"没删掉"
    // 还是"数错了"。空列表时 `.studio-list` 根本不存在（渲染的是 `.studio-empty`），
    // 所以条数为 0 也可能是正常终态。
    const 剩余标题 = await s.eval(
      `[...document.querySelectorAll('.studio-list button')].map(b => b.textContent.replace(/\\s+/g, ' ').slice(0, 24))`);
    check('产出页删得掉自己生成的记录（跑完不留痕迹）', remaining === listed - 1,
          `${listed} -> ${remaining} | 剩余：${JSON.stringify(剩余标题)}`);
  }

  // ---- 13. 批量删除：资料库页的完整选择 → 确认 → 删除流程 ----
  // 造两篇专用资料（fetch 直传，绕开无头环境里点不开的文件选择器），从界面上
  // 走完"批量选择 → 勾选 → 确认 → 删除"，最后从服务端核对它们真的没了——
  // 造的数据自己收走。**只勾自己造的行**：实例里可能还有用户的真实资料，
  // 全选会把它们一起送进删除请求。
  const uploadDoc = async name => {
    const fd = new FormData();
    fd.append('file', new File([`# ${name}\n\n批量删除验收专用内容。`], `${name}.md`,
      { type: 'text/markdown' }));
    return fetch(`${APP}api/v1/documents/upload`, { method: 'POST', body: fd }).then(r => r.json());
  };
  const uploadA = await uploadDoc('批量删除验收甲');
  const uploadB = await uploadDoc('批量删除验收乙');
  check('验收用资料上传成功', !!uploadA.document_id && !!uploadB.document_id,
        JSON.stringify([uploadA, uploadB]));
  const madeIds = [uploadA.document_id, uploadB.document_id].filter(Boolean);

  // fetch 直传不触发界面刷新（只有页面上的上传控件才会），所以等作业完成后
  // 切走再切回，让 LibraryPage 重新挂载加载——这是脚本绕开 UI 上传的副作用，
  // 不是产品缺陷：真实用户从界面上传，列表立即更新。
  let seen = false;
  for (let i = 0; i < 15 && !seen; i++) {
    await sleep(2000);
    await s.eval(`(() => { const b=[...document.querySelectorAll('nav button')].find(x=>x.textContent.includes('知识问答')); if(b) b.click(); })()`);
    await sleep(700);
    await s.eval(`(() => { const b=[...document.querySelectorAll('nav button')].find(x=>x.textContent.includes('资料库')); if(b) b.click(); })()`);
    await sleep(1200);
    seen = await s.eval(`['批量删除验收甲', '批量删除验收乙'].every(name =>
      [...document.querySelectorAll('.document-row')].some(r => r.textContent.includes(name)))`);
  }
  check('两篇验收资料出现在资料列表里', seen === true, '');

  const entry = await s.eval(`(() => {
    const b = [...document.querySelectorAll('button')].find(x => x.textContent.trim() === '批量选择');
    if (!b) return { 存在: false };
    b.click(); return { 存在: true };
  })()`);
  check('资料库页有「批量选择」入口', entry.存在 === true, JSON.stringify(entry));
  await sleep(400);
  const idleBar = await s.eval(`(() => {
    const count = document.querySelector('.batch-count');
    const del = document.querySelector('[data-testid="batch-confirm"]');
    return { 条: !!count, 计数: count ? count.textContent.replace(/\\s+/g, '') : '',
      删除禁用: del ? del.disabled : null };
  })()`);
  check('进入选择模式：0 选中时删除不可点',
        idleBar.计数 === '已选0项' && idleBar.删除禁用 === true, JSON.stringify(idleBar));

  // 勾选自己造的第一篇。checkbox 的 change 由真实点击触发。
  const tick = name => s.eval(`(() => {
    const row = [...document.querySelectorAll('.document-row')]
      .find(r => r.textContent.includes(${JSON.stringify(name)}));
    const box = row && row.querySelector('.batch-check');
    if (!box) return 'no-row';
    if (box.checked) return 'already';
    box.click(); return 'ok';
  })()`);
  check('勾选「批量删除验收甲」', await tick('批量删除验收甲') === 'ok', '');
  await sleep(300);
  const onePicked = await s.eval(`document.querySelector('.batch-count').textContent.replace(/\\s+/g, '')`);
  check('勾选后计数变成 1 项', onePicked === '已选1项', onePicked);

  await s.eval(`document.querySelector('[data-testid="batch-confirm"]').click()`);
  await sleep(400);
  const batchConfirm = await s.eval(`(() => {
    const box = document.querySelector('.batch-confirm');
    if (!box) return { 出现: false };
    return { 出现: true, 文字: box.textContent };
  })()`);
  check('批量删除第一下只展开确认，不直接删',
        batchConfirm.出现 === true, JSON.stringify(batchConfirm));
  check('资料确认框写明「原文件不会被删除」',
        /原文件不会被删除/.test(batchConfirm.文字 || ''), JSON.stringify(batchConfirm));
  await s.shot('accept-batch-confirm.png', 1440, 900);
  // 甲乙都是上传来源，不涉及文件夹；确认框不该把"暂时移除"那句也对上传资料说。
  check('上传来源不触发「暂时移除」的说明',
        !/暂时移除/.test(batchConfirm.文字 || ''), JSON.stringify(batchConfirm));

  await s.eval(`(() => {
    const box = document.querySelector('.batch-confirm');
    const b = box && [...box.querySelectorAll('button')].find(x => x.textContent.trim() === '再想想');
    if (b) b.click();
  })()`);
  await sleep(300);
  check('「再想想」收起确认框且什么都没删',
        await s.eval(`!document.querySelector('.batch-confirm')`) === true, '');

  check('勾选「批量删除验收乙」', await tick('批量删除验收乙') === 'ok', '');
  await sleep(300);
  const twoPicked = await s.eval(`document.querySelector('.batch-count').textContent.replace(/\\s+/g, '')`);
  check('两篇都勾上后计数是 2 项', twoPicked === '已选2项', twoPicked);

  await s.eval(`document.querySelector('[data-testid="batch-confirm"]').click()`);
  await sleep(400);
  await s.eval(`document.querySelector('[data-testid="batch-confirm"]').click()`);
  // 删除请求 + 列表刷新（实测端点约 1.4 秒，给结果条留足时间）。
  let resultText = '';
  for (let i = 0; i < 40; i++) {
    const box = await s.eval(`(() => {
      const box = document.querySelector('[data-testid="batch-result"]');
      return box ? box.textContent : '';
    })()`);
    if (box) { resultText = box; break; }
    await sleep(500);
  }
  check('删除后如实报告「已删除 2 项」', /已删除 2 项/.test(resultText), resultText);

  await s.eval(`(() => {
    const b = [...document.querySelectorAll('button')].find(x => x.textContent.trim() === '完成');
    if (b) b.click();
  })()`);
  await sleep(800);
  // 判据落在服务端：列表里不再有这两篇，且它们的 id 已经查不到。
  const docsAfter = await fetch(`${APP}api/v1/documents`).then(r => r.json());
  const remainingIds = (docsAfter.documents || []).map(x => x.id);
  check('服务端确认两篇验收资料已被移除',
        madeIds.every(id => !remainingIds.includes(id)),
        `剩余 ${remainingIds.length} 篇`);

  // ---- 14. 退出入口：正常使用下必须存在，且不许把"关标签"当成退出 ----
  // 这一段默认**不真的退出**：退出会把正在被验收的这个实例关掉，跑完就没得看了。
  // 要真走一遍，设 UI_ACCEPT_EXIT=1——那一次运行的收尾就是"应用确实停了"。
  // （放在自造产出被删掉之后：先退出了，那个 DELETE 就发不出去了。）
  const exitEntry = await s.eval(`(() => {
    const button = document.querySelector('[data-testid="exit-open"]');
    if (!button) return { 存在: false };
    const scope = button.closest('.exit-control');
    return { 存在: true,
      文字: button.textContent.trim(),
      在侧栏: !!button.closest('.shell aside'),
      说明: (scope.querySelector('.exit-note') || {}).textContent || '' };
  })()`);
  // 必须是**常驻**的：设置页那个只在"保存后需要重启"时才出现，正常使用下等于没有。
  check('正常使用下侧栏就有退出入口',
        exitEntry.存在 === true && exitEntry.在侧栏 === true, JSON.stringify(exitEntry));
  check('退出入口旁边写明关掉标签不算退出',
        /关掉浏览器标签不会停止本地服务/.test(exitEntry.说明 || ''), exitEntry.说明);

  await s.eval(`document.querySelector('[data-testid="exit-open"]').click()`);
  await sleep(400);
  const confirmBox = await s.eval(`(() => {
    const box = document.querySelector('[data-testid="exit-confirm"]');
    if (!box) return { 出现: false };
    return { 出现: true,
      文字: (box.querySelector('[data-testid="exit-question"]') || {}).textContent || '',
      有确认: !!box.querySelector('[data-testid="exit-confirm-button"]'),
      有取消: !!box.querySelector('[data-testid="exit-cancel-button"]') };
  })()`);
  // 退出是不可逆的日常动作，第一次点击不该直接把它执行掉。
  check('第一下只展开确认，不直接退出',
        confirmBox.出现 === true && confirmBox.有确认 && confirmBox.有取消, JSON.stringify(confirmBox));
  check('确认里说明退出不删资料',
        /资料、索引和会话都留在原处/.test(confirmBox.文字 || ''), JSON.stringify(confirmBox));
  await s.shot('accept-exit-1440.png', 1440, 900);

  await s.eval(`document.querySelector('[data-testid="exit-cancel-button"]').click()`);
  await sleep(400);
  const afterCancel = await s.eval(`({
    入口还在: !!document.querySelector('[data-testid="exit-open"]'),
    确认已收起: !document.querySelector('[data-testid="exit-confirm"]'),
  })`);
  // 取消必须真的什么都没发生——从浏览器外面问一次服务，别只信页面上的 DOM。
  const healthAfterCancel = await fetch(`${APP}api/v1/health`)
    .then(response => response.status).catch(() => 0);
  check('取消之后应用照旧运行', afterCancel.入口还在 === true && afterCancel.确认已收起 === true
        && healthAfterCancel === 200, JSON.stringify({ ...afterCancel, healthAfterCancel }));

  if (process.env.UI_ACCEPT_EXIT === '1') {
    await s.eval(`document.querySelector('[data-testid="exit-open"]').click()`);
    await sleep(300);
    await s.eval(`document.querySelector('[data-testid="exit-confirm-button"]').click()`);
    await sleep(1500);
    check('确认退出后进到收尾页', await s.eval(`!!document.querySelector('[data-testid="signed-off"]')`) === true, '');
    // 真正的判据在浏览器外面：端口不再应答，才算"服务停了"。
    let stopped = false;
    for (let i = 0; i < 30; i++) {
      const alive = await fetch(`${APP}api/v1/health`).then(() => true).catch(() => false);
      if (!alive) { stopped = true; break; }
      await sleep(500);
    }
    check('退出之后本地服务真的停了（端口不再应答）', stopped === true, `${APP}api/v1/health`);
    await s.shot('accept-exit-signed-off.png', 1440, 900);
  } else {
    console.log('      未真的退出（设 UI_ACCEPT_EXIT=1 可走一遍；那会把应用关掉）');
  }

  const errors = s.consoleErrors;
  check('页面无未捕获异常', errors.length === 0, errors.slice(0, 3).join(' | '));

  await cleanup();
  const failed = results.filter(r => !r.passed);
  console.log(`\n合计 ${results.length} 项，通过 ${results.length - failed.length}，失败 ${failed.length}`);
  if (failed.length) { console.log('失败项：'); failed.forEach(f => console.log('  - ' + f.name)); process.exit(1); }
})().catch(async e => { console.error('脚本失败:', e.message); await cleanup(); process.exit(2); });
