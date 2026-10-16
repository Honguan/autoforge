// AutoForge GUI — plain ES module, no build step, no external dependencies.
// All dynamic text goes through textContent (never innerHTML) so data/log contents cannot inject markup.

const TOKEN = document.querySelector('meta[name="autoforge-token"]').content;
const app = document.getElementById("app");
let timers = [];

// ---------------------------------------------------------------- helpers

async function api(method, path, body, raw = false) {
  const opts = { method, headers: { "X-AutoForge-Token": TOKEN } };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) { opts.body = JSON.stringify(body); opts.headers["Content-Type"] = "application/json"; }
  const res = await fetch(path, opts);
  if (raw && res.ok) return res;
  let env;
  try { env = await res.json(); } catch { throw Object.assign(new Error(`HTTP ${res.status}`), { code: "HTTP" }); }
  if (!env.ok) {
    const e = new Error(env.error.message);
    Object.assign(e, { code: env.error.code, hint: env.error.hint, details: env.data });
    throw e;
  }
  return env.data;
}

function h(tag, attrs = {}, ...children) {
  const el = document.createElementNS(tag === "svg" || attrs.svg ? "http://www.w3.org/2000/svg" : "http://www.w3.org/1999/xhtml", tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false || k === "svg") continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.setAttribute("class", v);
    else if (k === "value" && "value" in el) el.value = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
const s = (tag, attrs = {}, ...c) => h(tag, { ...attrs, svg: true }, ...c);
// Like el.replaceChildren, but skips null/false (the native one would render them as the text "null").
const fill = (el, ...children) => el.replaceChildren(...children.flat(Infinity).filter((c) => c !== null && c !== undefined && c !== false));

function toast(msg, error = false) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "toast" + (error ? " error" : "");
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), error ? 7000 : 3500);
}

function showError(e) {
  toast(`${e.code ? `[${e.code}] ` : ""}${e.message}${e.hint ? ` — ${e.hint}` : ""}`, true);
}

function fmt(v) {
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v)) return String(v);
  return Math.abs(v) >= 1000 || Math.abs(v) < 0.001 ? v.toExponential(3) : v.toFixed(4);
}

const STATUS_ZH = { pending: "等待中", running: "訓練中", paused: "已暫停", succeeded: "完成", failed: "失敗", stopped: "已停止" };
const badge = (st) => h("span", { class: `badge ${st}` }, STATUS_ZH[st] || st);
const panel = (...c) => h("section", { class: "panel" }, ...c);
const every = (ms, fn) => { const id = setInterval(fn, ms); timers.push(id); };

function lineChart(series, title) {
  // series: [{name, points: [[x, y], ...]}]
  const W = 520, H = 180, L = 46, R = 10, T = 16, B = 22;
  const pts = series.flatMap((x) => x.points).filter((p) => Number.isFinite(p[1]));
  if (!pts.length) return null;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  let [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  if (x0 === x1) x1 = x0 + 1;
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const X = (x) => L + ((x - x0) / (x1 - x0)) * (W - L - R);
  const Y = (y) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": title });
  svg.append(s("line", { x1: L, y1: H - B, x2: W - R, y2: H - B, class: "axis" }),
    s("line", { x1: L, y1: T, x2: L, y2: H - B, class: "axis" }),
    s("text", { x: 4, y: T + 4 }, fmt(y1)), s("text", { x: 4, y: H - B }, fmt(y0)),
    s("text", { x: L, y: H - 4 }, fmt(x0)), s("text", { x: W - R - 30, y: H - 4 }, fmt(x1)));
  series.forEach((ser, i) => {
    const good = ser.points.filter((p) => Number.isFinite(p[1]));
    const k = (i % 4) + 1;
    if (good.length > 1) svg.append(s("polyline", { class: `line s${k}`, points: good.map((p) => `${X(p[0])},${Y(p[1])}`).join(" ") }));
    if (good.length <= 40) good.forEach((p) => svg.append(s("circle", { cx: X(p[0]), cy: Y(p[1]), r: 2.5, class: `dot d${k}` })));
  });
  const legend = h("div", { class: "row small muted" }, series.map((ser, i) =>
    h("span", { class: "chip" }, `● ${ser.name}`)));
  return h("div", {}, h("h3", {}, title), svg, series.length > 1 ? legend : null);
}

// ---------------------------------------------------------------- task presets

const TASKS = {
  "tabular-classification": { zh: "表格分類", target: "必填", metric: "f1_macro", inM: "tabular", outM: "label" },
  "tabular-regression": { zh: "表格數值預測", target: "必填", metric: "rmse", inM: "tabular", outM: "number" },
  "text-classification": { zh: "文字分類", target: "必填", metric: "f1_macro", inM: "text", outM: "label" },
  "image-classification": { zh: "圖片分類（資料夾）", target: "無", metric: "accuracy", inM: "image", outM: "label" },
  "timeseries-forecast": { zh: "時間序列預測", target: "必填", metric: "mae", inM: "series", outM: "values" },
  "anomaly-detection": { zh: "異常偵測", target: "選填", metric: "f1", inM: "tabular", outM: "flag" },
  "llm-sft": { zh: "LLM 微調", target: "無", metric: "val_loss", inM: "text", outM: "text" },
  "llm-preference": { zh: "LLM 偏好對齊（DPO）", target: "無", metric: "reward_accuracy", inM: "text", outM: "text" },
  "object-detection": { zh: "物件偵測（資料夾）", target: "無", metric: "map50", inM: "image", outM: "boxes" },
  "image-segmentation": { zh: "影像分割（資料夾）", target: "無", metric: "miou", inM: "image", outM: "mask" },
  "text-embedding": { zh: "Embedding 微調", target: "無", metric: "recall_at_1", inM: "text", outM: "vector" },
  "speech-to-text": { zh: "語音辨識（Whisper）", target: "無", metric: "wer", inM: "audio", outM: "text" },
  "text-to-image": { zh: "圖像生成 LoRA（資料夾）", target: "無", metric: "val_loss", inM: "text", outM: "image" },
  "reinforcement-learning": { zh: "強化學習（環境設定檔）", target: "無", metric: "mean_reward", inM: "environment", outM: "action" },
  "recommendation": { zh: "推薦系統", target: "選填", metric: "recall_at_k", inM: "interactions", outM: "items" },
};
const taskInfo = (t) => TASKS[t] || { zh: t, target: "選填", metric: "score", inM: "data", outM: "prediction" };

// ---------------------------------------------------------------- pages

async function pageHome() {
  const [hw, runs, models] = await Promise.all([api("GET", "/api/hardware"), api("GET", "/api/runs?limit=6"), api("GET", "/api/models")]);
  const gpus = hw.gpus.length ? hw.gpus.map((g) => `${g.name}（${g.vram_total_gb} GB）`).join("、") : "沒有偵測到 NVIDIA GPU";
  app.append(
    h("div", { class: "row" }, h("h1", {}, "總覽"), h("span", { class: "spacer" }),
      h("a", { class: "btn primary", href: "#/new" }, "＋ 開始新訓練")),
    h("div", { class: "grid" },
      panel(h("h2", {}, "這台電腦"),
        h("div", { class: "kv" },
          h("span", { class: "k" }, "CPU"), h("span", {}, `${hw.cpu.model}（${hw.cpu.logical_cores} 執行緒）`),
          h("span", { class: "k" }, "記憶體"), h("span", {}, `${hw.ram_available_gb} / ${hw.ram_total_gb} GB 可用`),
          h("span", { class: "k" }, "GPU"), h("span", {}, gpus),
          h("span", { class: "k" }, "訓練裝置"), h("span", {}, hw.best_device === "cuda" ? "GPU（CUDA）" : "CPU")),
        hw.detection_notes.length ? h("details", { class: "small" }, h("summary", {}, "偵測備註"),
          h("ul", {}, hw.detection_notes.map((n) => h("li", {}, n)))) : null),
      panel(h("h2", {}, "數量"), h("div", { class: "row" },
        h("div", { class: "stat" }, h("span", { class: "v" }, runs.length), h("span", { class: "k" }, "最近訓練")),
        h("div", { class: "stat" }, h("span", { class: "v" }, models.length), h("span", { class: "k" }, "已註冊模型"))))),
    panel(h("h2", {}, "最近的訓練"), runsTable(runs)));
}

function runsTable(runs) {
  if (!runs.length) return h("p", { class: "empty" }, "還沒有訓練紀錄。按「開始新訓練」試試看。");
  return h("div", { class: "tablewrap" }, h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, "訓練"), h("th", {}, "狀態"), h("th", {}, "方法"), h("th", {}, "指標"), h("th", {}, "開始者"))),
    h("tbody", {}, runs.map((r) => h("tr", {},
      h("td", {}, h("a", { href: `#/runs/${r.run_id}` }, r.run_id)),
      h("td", {}, badge(r.status)),
      h("td", {}, `${r.pack || "自訂程式"} / ${r.recipe || ""}`),
      h("td", { class: "small" }, Object.entries(Object.keys(r.metrics).length ? r.metrics : r.latest_metrics)
        .slice(0, 3).map(([k, v]) => `${k}=${fmt(v)}`).join("  ")),
      h("td", { class: "small muted" }, r.actor))))));
}

async function pageRuns() {
  const [runs, studies] = await Promise.all([api("GET", "/api/runs?limit=100"), api("GET", "/api/hpo")]);
  app.append(h("h1", {}, "訓練紀錄"), panel(runsTable(runs)));
  if (studies.length) app.append(panel(h("h2", {}, "自動調參"), h("table", {}, h("tbody", {}, studies.map((st) =>
    h("tr", {}, h("td", {}, h("a", { href: `#/hpo/${st.study_id}` }, st.study_id)), h("td", {}, badge(st.status)),
      h("td", {}, `${st.completed}/${st.n_trials} 次`),
      h("td", {}, st.best_value != null ? `${st.metric} 最佳 ${fmt(st.best_value)}` : "")))))));
}

// ---- new training wizard

async function pageNew() {
  const st = { step: 1, dataPath: "", profile: null, task: "", target: "", metric: "", pack: "", recipe: "", params: {} };
  const root = h("div");
  app.append(h("h1", {}, "新訓練"), root);
  const steps = ["1 資料", "2 任務", "3 方法", "4 確認並開始"];

  async function render() {
    fill(root, h("div", { class: "steps" }, steps.map((t, i) => h("span", { class: i + 1 === st.step ? "on" : "" }, t))));
    try {
      if (st.step === 1) await stepData();
      if (st.step === 2) await stepTask();
      if (st.step === 3) await stepMethod();
      if (st.step === 4) await stepReview();
    } catch (e) { showError(e); }
  }

  async function loadProfile(path) {
    st.dataPath = path;
    st.profile = await api("GET", `/api/data/profile?path=${encodeURIComponent(path)}`);
    render();
  }

  async function stepData() {
    const files = await api("GET", "/api/data/files");
    const fileInput = h("input", { type: "file", id: "upload", accept: ".csv,.jsonl,.json,.txt" });
    const pathInput = h("input", { type: "text", id: "data-path", placeholder: "/完整路徑/data.csv 或圖片資料夾", value: st.dataPath, size: 50 });
    root.append(panel(h("h2", {}, "選擇訓練資料"),
      h("div", { class: "row" }, fileInput, h("button", { class: "primary", id: "upload-btn", onclick: async () => {
        if (!fileInput.files.length) return toast("先選一個檔案", true);
        const fd = new FormData();
        fd.append("file", fileInput.files[0]);
        try { const up = await api("POST", "/api/data/upload", fd); toast(`已上傳 ${up.name}`); await loadProfile(up.path); } catch (e) { showError(e); }
      } }, "上傳")),
      h("p", { class: "small muted" }, "或輸入這台電腦上的路徑（圖片分類請輸入「每個類別一個子資料夾」的資料夾）："),
      h("div", { class: "row" }, pathInput, h("button", { id: "use-path", onclick: () => loadProfile(pathInput.value.trim()).catch(showError) }, "使用這個路徑")),
      files.length ? h("div", {}, h("h3", {}, "之前上傳的資料"), h("div", { class: "row" }, files.slice(0, 8).map((f) =>
        h("button", { onclick: () => loadProfile(f.path).catch(showError) }, f.name)))) : null));
    if (st.profile) root.append(profileView(st.profile), h("div", { class: "row" }, h("span", { class: "spacer" }),
      h("button", { class: "primary", id: "next-1", onclick: () => { st.step = 2; render(); } }, "下一步：選任務 →")));
  }

  function profileView(p) {
    const head = h("div", { class: "row" }, h("h2", {}, "資料概況"), h("span", { class: "chip" }, `敏感度：${p.sensitivity}`));
    if (p.format !== "csv") return panel(head, h("div", { class: "kv" },
      Object.entries(p).filter(([k]) => !["path", "access"].includes(k)).map(([k, v]) => [h("span", { class: "k" }, k), h("span", {}, typeof v === "object" ? JSON.stringify(v) : String(v))])));
    return panel(head, h("p", { class: "small muted" }, `${p.rows} 列，${Object.keys(p.columns).length} 個欄位`),
      h("div", { class: "tablewrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["欄位", "型別", "缺值", "不同值", "範圍 / 常見值"].map((t) => h("th", {}, t)))),
        h("tbody", {}, Object.entries(p.columns).map(([name, c]) => h("tr", {}, h("td", { class: "mono" }, name), h("td", {}, c.dtype),
          h("td", {}, c.missing), h("td", {}, c.distinct),
          h("td", { class: "small" }, c.dtype === "numeric" ? `${fmt(c.min)} ~ ${fmt(c.max)}（平均 ${fmt(c.mean)}）`
            : Object.entries(c.top_values || {}).map(([k, v]) => `${k}(${v})`).join("、"))))))));
  }

  async function stepTask() {
    const catalog = await api("GET", "/api/tasks");
    const cols = st.profile && st.profile.columns ? Object.keys(st.profile.columns) : [];
    const taskSel = h("select", { id: "task", onchange: () => { st.task = taskSel.value; st.metric = taskInfo(st.task).metric; render(); } },
      h("option", { value: "" }, "— 選擇任務 —"),
      catalog.map((t) => h("option", { value: t.task, selected: t.task === st.task }, `${taskInfo(t.task).zh}（${t.task}）`)));
    const info = st.task ? taskInfo(st.task) : null;
    const targetSel = h("select", { id: "target", onchange: () => (st.target = targetSel.value) },
      h("option", { value: "" }, info && info.target === "必填" ? "— 選擇 —" : "（不需要）"),
      cols.map((c) => h("option", { value: c, selected: c === st.target }, c)));
    const metricIn = h("input", { id: "metric", value: st.metric, onchange: () => (st.metric = metricIn.value.trim()) });
    root.append(panel(h("h2", {}, "你想讓模型做什麼？"),
      h("div", { class: "grid" },
        h("label", { class: "field" }, "任務類型", taskSel),
        h("label", { class: "field" }, `要預測的欄位（${info ? info.target : "依任務"}）`, targetSel),
        h("label", { class: "field" }, "主要評估指標", metricIn))),
      h("div", { class: "row" }, h("button", { onclick: () => { st.step = 1; render(); } }, "← 上一步"), h("span", { class: "spacer" }),
        h("button", { class: "primary", id: "next-2", onclick: () => {
          if (!st.task) return toast("請選擇任務類型", true);
          if (info.target === "必填" && !st.target) return toast("請選擇要預測的欄位", true);
          st.step = 3; render();
        } }, "下一步：選方法 →")));
  }

  async function stepMethod() {
    const res = await api("GET", `/api/packs/search?task=${encodeURIComponent(st.task)}&fit=1`);
    const box = h("div", { class: "grid" });
    if (!res.hits.length) box.append(h("p", { class: "empty" }, "找不到這台電腦能跑的方法。可能需要 GPU，或到「方法包」頁面看看。"));
    for (const hit of res.hits) {
      for (const rid of hit.runnable_recipes) {
        const rec = hit.entry.recipes.find((r) => r.id === rid);
        const selected = st.pack === hit.entry.name && st.recipe === rid;
        box.append(h("div", { class: `card selectable${selected ? " selected" : ""}`, "data-recipe": rid, onclick: async () => {
          try {
            if (!hit.installed) { toast(`安裝 ${hit.entry.name}…`); await api("POST", "/api/packs/install", { source: hit.entry.name }); }
            const info = await api("GET", `/api/packs/${hit.entry.name}`);
            const r = info.recipes.find((x) => x.id === rid);
            Object.assign(st, { pack: hit.entry.name, recipe: rid, params: { ...r.defaults }, recipeInfo: r, knowledge: info.knowledge,
              missing: info.missing_dependencies });
            render();
          } catch (e) { showError(e); }
        } }, h("strong", {}, rid), h("div", { class: "small muted" }, hit.entry.name, hit.installed ? "" : "（點選會先安裝）"),
          h("p", { class: "small" }, rec ? rec.good_for : "")));
      }
    }
    root.append(panel(h("h2", {}, "選擇方法（只列出這台電腦跑得動的）"), box));
    if (st.recipe && st.recipeInfo) {
      const r = st.recipeInfo;
      const tbody = h("tbody");
      for (const [k, v] of Object.entries(st.params)) {
        const inp = h("input", { "data-param": k, value: JSON.stringify(v), class: "mono", onchange: () => {
          try { st.params[k] = JSON.parse(inp.value); } catch { st.params[k] = inp.value; }
        } });
        const space = r.search_space[k];
        tbody.append(h("tr", {}, h("td", { class: "mono" }, k), h("td", {}, inp),
          h("td", { class: "small muted" }, space ? (space.choices ? `可選：${space.choices.map((x) => JSON.stringify(x)).join(", ")}` : `建議 ${space.low} ~ ${space.high}`) : "")));
      }
      root.append(panel(h("h2", {}, `參數：${r.name}`),
        r.avoid_when ? h("p", { class: "small" }, `不適合：${r.avoid_when}`) : null,
        st.missing && st.missing.length ? h("p", { class: "small problems" }, `缺少套件：${st.missing.join("、")}（請先 pip install）`) : null,
        h("div", { class: "tablewrap" }, h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "名稱"), h("th", {}, "值（JSON）"), h("th", {}, "範圍"))), tbody)),
        h("details", {}, h("summary", {}, "這個方法包的使用說明（KNOWLEDGE.md）"), h("pre", {}, st.knowledge || ""))));
    }
    root.append(h("div", { class: "row" }, h("button", { onclick: () => { st.step = 2; render(); } }, "← 上一步"), h("span", { class: "spacer" }),
      h("button", { class: "primary", id: "next-3", disabled: !st.recipe, onclick: () => { st.step = 4; render(); } }, "下一步：確認 →")));
  }

  function plan() {
    const info = taskInfo(st.task);
    return { task: { task_type: st.task, input_modality: info.inM, output_modality: info.outM, target: st.target || null, metric: st.metric },
      data_path: st.dataPath, pack: st.pack, recipe: st.recipe, hyperparameters: st.params };
  }

  async function stepReview() {
    const p = plan();
    const result = h("div");
    root.append(panel(h("h2", {}, "訓練計畫"), h("pre", { id: "plan-json" }, JSON.stringify(p, null, 2)), result),
      h("div", { class: "row" }, h("button", { onclick: () => { st.step = 3; render(); } }, "← 上一步"), h("span", { class: "spacer" }),
        h("button", { class: "primary", id: "start", onclick: async (ev) => {
          ev.target.disabled = true;
          try { const run = await api("POST", "/api/runs", { plan: p }); toast("已開始訓練"); location.hash = `#/runs/${run.run_id}`; }
          catch (e) { ev.target.disabled = false; showError(e); if (e.details && e.details.problems) showCheck(result, e.details); }
        } }, "開始訓練")));
    try { showCheck(result, await api("POST", "/api/plans/validate", { plan: p })); } catch (e) { showError(e); if (e.details) showCheck(result, e.details); }
  }

  function showCheck(el, c) {
    fill(el, h("h3", {}, c.ok ? `✓ 檢查通過，會用 ${c.device === "cuda" ? "GPU" : "CPU"} 訓練` : "✗ 計畫有問題"),
      c.problems && c.problems.length ? h("ul", { class: "problems" }, c.problems.map((x) => h("li", {}, x))) : null,
      c.warnings && c.warnings.length ? h("ul", { class: "warnings" }, c.warnings.map((x) => h("li", {}, x))) : null);
  }

  render();
}

// ---- run detail

async function pageRun(id) {
  const root = h("div");
  app.append(root);
  let logsBox = null;

  async function refresh(first = false) {
    const [r, series, logs] = await Promise.all([api("GET", `/api/runs/${id}`), api("GET", `/api/runs/${id}/series`),
      api("GET", `/api/runs/${id}/logs?tail=200`)]);
    const live = ["pending", "running", "paused"].includes(r.status);
    const keep = logsBox && logsBox.scrollTop + logsBox.clientHeight < logsBox.scrollHeight - 10 ? logsBox.scrollTop : null;
    const charts = h("div", { class: "grid" });
    const groups = {};
    for (const [name, pts] of Object.entries(series)) {
      const base = name.replace(/^(train_|val_|valid_|eval_)/, "");
      (groups[base] = groups[base] || []).push({ name, points: pts });
    }
    for (const [base, sers] of Object.entries(groups)) { const c = lineChart(sers, base); if (c) charts.append(c); }
    logsBox = h("pre", { id: "logs" }, logs.lines.join("\n") || "（還沒有輸出）");
    fill(root, 
      h("div", { class: "row" }, h("h1", {}, "訓練 ", h("span", { class: "mono small" }, id)), badge(r.status), h("span", { class: "spacer" }),
        r.status === "running" ? h("button", { onclick: () => act("pause") }, "暫停") : null,
        r.status === "paused" ? h("button", { onclick: () => act("resume") }, "繼續") : null,
        live ? h("button", { class: "danger", onclick: () => act("stop") }, "停止") : null,
        !live ? h("button", { onclick: () => act("retry") }, "重新訓練") : null),
      h("div", { class: "grid" },
        panel(h("h2", {}, "結果"), Object.keys(r.metrics).length
          ? h("div", { class: "row" }, Object.entries(r.metrics).map(([k, v]) => h("div", { class: "stat" }, h("span", { class: "v" }, fmt(v)), h("span", { class: "k" }, k))))
          : h("p", { class: "muted" }, live ? "訓練中…" : "沒有結果"),
          r.error ? h("pre", {}, r.error) : null),
        panel(h("h2", {}, "設定"), h("div", { class: "kv" },
          h("span", { class: "k" }, "方法"), h("span", {}, `${r.pack || "自訂程式"} / ${r.recipe || ""}`),
          h("span", { class: "k" }, "裝置"), h("span", {}, r.device),
          h("span", { class: "k" }, "資料"), h("span", { class: "small mono" }, r.plan.data_path),
          h("span", { class: "k" }, "開始者"), h("span", {}, r.actor),
          r.study_id ? [h("span", { class: "k" }, "所屬調參"), h("a", { href: `#/hpo/${r.study_id}` }, r.study_id)] : null),
          h("details", {}, h("summary", {}, "超參數"), h("pre", {}, JSON.stringify(r.plan.hyperparameters, null, 2))))),
      charts.childNodes.length ? panel(h("h2", {}, "訓練曲線"), charts) : null,
      panel(h("h2", {}, "輸出紀錄"), logsBox),
      r.status === "succeeded" ? nextSteps(r) : null,
      !live ? diagnosePanel() : null);
    if (keep === null) logsBox.scrollTop = logsBox.scrollHeight; else logsBox.scrollTop = keep;
    return live;
  }

  function nextSteps(r) {
    const name = h("input", { id: "model-name", placeholder: "模型名稱，例如 churn", value: "" });
    const metricSel = h("select", { id: "hpo-metric" }, Object.keys(r.metrics).map((k) => h("option", { value: k, selected: k === r.plan.task.metric }, k)));
    const trials = h("input", { id: "hpo-trials", type: "number", min: 2, max: 200, value: 10 });
    return h("div", { class: "grid" },
      panel(h("h2", {}, "註冊成模型"), h("p", { class: "small muted" }, "註冊後就能預測、開 API，或讓其他 AI 透過 MCP 使用。"),
        h("div", { class: "row" }, name, h("button", { class: "primary", id: "register", onclick: async () => {
          try { const c = await api("POST", "/api/models", { run_id: id, name: name.value.trim() }); toast(`已註冊 ${c.name}:v${c.version}`);
            location.hash = `#/models/${c.name}`; } catch (e) { showError(e); }
        } }, "註冊"))),
      panel(h("h2", {}, "自動調參"), h("p", { class: "small muted" }, "以這次的參數為基準（第 0 次），自動嘗試更好的組合。"),
        h("div", { class: "row" }, h("label", { class: "field" }, "要優化的指標", metricSel), h("label", { class: "field" }, "嘗試次數", trials),
          h("button", { id: "hpo-start", onclick: async () => {
            try { const run = await api("GET", `/api/runs/${id}`);
              const stdy = await api("POST", "/api/hpo", { plan: run.plan, metric: metricSel.value, n_trials: Number(trials.value) });
              location.hash = `#/hpo/${stdy.study_id}`; } catch (e) { showError(e); }
          } }, "開始調參"))));
  }

  function diagnosePanel() {
    const out = h("div");
    return panel(h("div", { class: "row" }, h("h2", {}, "診斷"), h("span", { class: "spacer" }),
      h("button", { id: "diagnose", onclick: async () => {
        try { const d = await api("GET", `/api/runs/${id}/diagnose`);
          fill(out, h("p", {}, d.summary), d.findings.map((f) => h("div", { class: `finding ${f.severity}` },
            h("strong", {}, f.code), " ", f.message, f.suggestion ? h("div", { class: "small" }, `建議：${f.suggestion}`) : null)));
        } catch (e) { showError(e); }
      } }, "執行診斷")), out);
  }

  async function act(action) {
    try { const r = await api("POST", `/api/runs/${id}/${action}`); toast(`已送出：${action}`);
      if (action === "retry") location.hash = `#/runs/${r.run_id}`; else await refresh(); } catch (e) { showError(e); }
  }

  const live = await refresh(true);
  if (live) every(1500, async () => { try { if (!(await refresh())) { timers.forEach(clearInterval); timers = []; } } catch (e) { /* keep polling */ } });
}

// ---- hyperparameter search

async function pageHpo(id) {
  const root = h("div");
  app.append(root);
  async function refresh() {
    const v = await api("GET", `/api/hpo/${id}`);
    const pts = v.trials.filter((t) => "value" in t).map((t) => [t.number, t.value]);
    fill(root, 
      h("div", { class: "row" }, h("h1", {}, "自動調參 ", h("span", { class: "mono small" }, id)), badge(v.status), h("span", { class: "spacer" }),
        ["pending", "running"].includes(v.status) ? h("button", { class: "danger", onclick: async () => {
          try { await api("POST", `/api/hpo/${id}/stop`); toast("已停止"); refresh(); } catch (e) { showError(e); } } }, "停止") : null),
      panel(h("div", { class: "row" },
        h("div", { class: "stat" }, h("span", { class: "v" }, `${v.completed}/${v.n_trials}`), h("span", { class: "k" }, "完成次數")),
        h("div", { class: "stat" }, h("span", { class: "v" }, v.baseline_value != null ? fmt(v.baseline_value) : "–"), h("span", { class: "k" }, `基準 ${v.metric}`)),
        h("div", { class: "stat" }, h("span", { class: "v", id: "best-value" }, v.best_value != null ? fmt(v.best_value) : "–"), h("span", { class: "k" }, `最佳 ${v.metric}`)),
        h("div", { class: "stat" }, h("span", { class: "v" }, v.improvement != null ? (v.improvement >= 0 ? "+" : "") + fmt(v.improvement) : "–"), h("span", { class: "k" }, "比基準進步"))),
        v.best_run_id ? h("p", {}, "最佳那次訓練：", h("a", { href: `#/runs/${v.best_run_id}` }, v.best_run_id), "（可以從那裡註冊成模型）") : null),
      pts.length ? panel(lineChart([{ name: v.metric, points: pts }], `每次嘗試的 ${v.metric}（${v.direction === "maximize" ? "越高越好" : "越低越好"}）`)) : null,
      panel(h("h2", {}, "每次嘗試"), h("div", { class: "tablewrap" }, h("table", {},
        h("thead", {}, h("tr", {}, ["#", v.metric, "參數", "訓練"].map((t) => h("th", {}, t)))),
        h("tbody", {}, v.trials.map((t) => h("tr", {}, h("td", {}, t.baseline ? `${t.number}（基準）` : t.number),
          h("td", {}, "value" in t ? fmt(t.value) : h("span", { class: "muted" }, "失敗")),
          h("td", { class: "mono small" }, JSON.stringify(t.params)),
          h("td", {}, t.run_id ? h("a", { href: `#/runs/${t.run_id}` }, "查看") : ""))))))));
    return ["pending", "running"].includes(v.status);
  }
  if (await refresh()) every(2000, async () => { try { if (!(await refresh())) { timers.forEach(clearInterval); timers = []; } } catch { /* retry */ } });
}

// ---- models

async function pageModels() {
  const models = await api("GET", "/api/models");
  app.append(h("h1", {}, "模型"), panel(models.length ? h("div", { class: "tablewrap" }, h("table", {},
    h("thead", {}, h("tr", {}, ["模型", "任務", "指標", "版本"].map((t) => h("th", {}, t)))),
    h("tbody", {}, models.map((m) => h("tr", {}, h("td", {}, h("a", { href: `#/models/${m.model.split(":")[0]}` }, m.model)),
      h("td", {}, taskInfo(m.task).zh), h("td", { class: "small" }, Object.entries(m.metrics).slice(0, 3).map(([k, v]) => `${k}=${fmt(v)}`).join("  ")),
      h("td", {}, m.versions.length))))))
    : h("p", { class: "empty" }, "還沒有模型。訓練完成後，在訓練頁面按「註冊」。")));
}

async function pageModel(ref) {
  const c = await api("GET", `/api/models/${encodeURIComponent(ref)}`);
  const inputs = {};
  const form = h("div", { class: "grid" }, c.features.map((f) => {
    let inp;
    if (f.dtype === "numeric") inp = h("input", { type: "number", step: "any", "data-feature": f.name });
    else if (f.dtype === "text") inp = h("textarea", { "data-feature": f.name });
    else if (f.known_values) {
      const listId = `dl-${f.name.replace(/\W/g, "_")}`;
      inp = h("input", { "data-feature": f.name, list: listId });
      inputs[f.name] = { el: inp, f };
      return h("label", { class: "field" }, f.name, inp, h("datalist", { id: listId }, f.known_values.map((v) => h("option", { value: v }))));
    } else inp = h("input", { "data-feature": f.name, placeholder: f.dtype === "image" ? "伺服器上的圖片路徑" : f.dtype === "any" ? "JSON" : "" });
    inputs[f.name] = { el: inp, f };
    return h("label", { class: "field" }, `${f.name}${f.required ? "" : "（選填）"}${f.description ? `：${f.description}` : ""}`, inp);
  }));
  const result = h("div", { id: "predict-result" });
  app.append(
    h("div", { class: "row" }, h("h1", {}, `模型 ${c.name}:v${c.version}`), h("span", { class: "spacer" }),
      h("button", { onclick: () => download(c, "bundle") }, "下載 bundle"),
      c.exporters && Object.keys(c.exporters).map((fmtName) => h("button", { onclick: () => download(c, fmtName) }, `匯出 ${fmtName}`))),
    h("div", { class: "grid" },
      panel(h("h2", {}, "說明"), h("div", { class: "kv" },
        h("span", { class: "k" }, "任務"), h("span", {}, `${taskInfo(c.task.task_type).zh}${c.task.target ? `，預測 ${c.task.target}` : ""}`),
        h("span", { class: "k" }, "方法"), h("span", {}, `${c.pack || "自訂"} ${c.pack_version || ""} / ${c.recipe || ""}`),
        h("span", { class: "k" }, "來源訓練"), h("a", { href: `#/runs/${c.run_id}` }, c.run_id),
        h("span", { class: "k" }, "可預測結果"), h("span", {}, c.labels ? c.labels.join("、") : "–"),
        h("span", { class: "k" }, "版本"), h("span", {}, c.versions.map((v) => `v${v}`).join("、"))),
        c.description ? h("p", {}, c.description) : null),
      panel(h("h2", {}, "驗證分數"), h("div", { class: "row" }, Object.entries(c.metrics).map(([k, v]) =>
        h("div", { class: "stat" }, h("span", { class: "v" }, fmt(v)), h("span", { class: "k" }, k)))))),
    c.servable ? panel(h("h2", {}, "試著預測"), form, h("div", { class: "row" }, h("button", { class: "primary", id: "predict", onclick: async () => {
      const row = {};
      for (const [name, { el, f }] of Object.entries(inputs)) {
        const v = el.value.trim();
        if (v === "") continue;
        row[name] = f.dtype === "numeric" ? Number(v) : f.dtype === "any" ? (() => { try { return JSON.parse(v); } catch { return v; } })() : v;
      }
      try {
        const out = await api("POST", `/api/models/${encodeURIComponent(c.name)}/predict`, { rows: [row] });
        const p = out.predictions[0];
        const isImg = typeof p.prediction === "string" && /^data:image\/(png|jpeg|webp);base64,/.test(p.prediction);
        const text = typeof p.prediction === "object" ? JSON.stringify(p.prediction) : String(p.prediction);
        fill(result, h("h3", {}, "預測結果：", isImg ? null : h("span", { class: "mono", id: "prediction" }, text.length > 600 ? text.slice(0, 600) + "…" : text)),
          isImg ? h("img", { id: "prediction", src: p.prediction, alt: "預測結果", class: "pred-img" }) : null,
          p.probabilities ? h("div", {}, Object.entries(p.probabilities).sort((a, b) => b[1] - a[1]).map(([k, v]) => h("span", { class: "chip" }, `${k}: ${(v * 100).toFixed(1)}%`))) : null,
          p.score !== undefined ? h("p", {}, `分數：${fmt(p.score)}`) : null,
          out.warnings.length ? h("ul", { class: "warnings" }, out.warnings.map((w) => h("li", {}, w))) : null);
      } catch (e) { showError(e); }
    } }, "預測")), result) : panel(h("p", {}, "這個模型沒有預測程式，不能直接預測。")),
    panel(h("h2", {}, "給其他程式或 AI 使用"), h("pre", {}, [
      "# REST API", "autoforge serve", `curl -s localhost:8000/v1/models/${c.name}/predict -H 'content-type: application/json' -d '{"rows":[{...}]}'`,
      "", "# MCP（Claude Desktop / Claude Code）", "autoforge serve-mcp   → 工具名稱 predict_" + c.name.replace(/[^a-zA-Z0-9_]/g, "_")].join("\n"))));
}

async function download(c, format) {
  try {
    toast("匯出中…");
    const res = await api("GET", `/api/models/${encodeURIComponent(`${c.name}:v${c.version}`)}/export?format=${format}`, undefined, true);
    const blob = await res.blob();
    const name = (res.headers.get("content-disposition") || "").match(/filename="?([^"]+)"?/);
    const a = h("a", { href: URL.createObjectURL(blob), download: name ? name[1] : `${c.name}.${format}` });
    document.body.append(a); a.click(); a.remove();
  } catch (e) { showError(e); }
}

// ---- packs

async function pagePacks() {
  const installed = await api("GET", "/api/packs");
  const results = h("div", { class: "grid" });
  const q = h("input", { id: "pack-q", placeholder: "關鍵字，例如 csv、image、llm", size: 30 });
  const fit = h("input", { type: "checkbox", id: "pack-fit", checked: true });
  async function search() {
    try {
      const res = await api("GET", `/api/packs/search?q=${encodeURIComponent(q.value)}&fit=${fit.checked ? 1 : 0}`);
      fill(results, ...(res.hits.length ? res.hits.map((hit) => h("div", { class: "card" },
        h("div", { class: "row" }, h("strong", {}, hit.entry.name), h("span", { class: "chip" }, hit.entry.version), h("span", { class: "spacer" }),
          hit.installed ? h("span", { class: "badge succeeded" }, "已安裝") : h("button", { onclick: async (ev) => {
            ev.target.disabled = true;
            try { await api("POST", "/api/packs/install", { source: hit.entry.name }); toast(`已安裝 ${hit.entry.name}`); search(); } catch (e) { ev.target.disabled = false; showError(e); }
          } }, "安裝")),
        h("p", { class: "small" }, hit.entry.summary),
        h("div", {}, hit.entry.tasks.map((t) => h("span", { class: "chip" }, taskInfo(t).zh))),
        h("p", { class: "small muted" }, `這台電腦能跑：${hit.runnable_recipes.join("、") || "（無）"}`)))
        : [h("p", { class: "empty" }, "沒有符合的方法包")]));
    } catch (e) { showError(e); }
  }
  app.append(h("h1", {}, "方法包"),
    panel(h("h2", {}, "已安裝"), installed.length ? h("table", {}, h("tbody", {}, installed.map((p) =>
      h("tr", {}, h("td", {}, h("strong", {}, p.name)), h("td", {}, p.version), h("td", { class: "small mono muted" }, p.source)))))
      : h("p", { class: "empty" }, "還沒有安裝方法包")),
    panel(h("h2", {}, "從索引搜尋"), h("div", { class: "row" }, q, h("label", { class: "row small" }, fit, "只顯示這台電腦跑得動的"),
      h("button", { class: "primary", onclick: search }, "搜尋")), results));
  search();
}

// ---------------------------------------------------------------- router

const ROUTES = [
  [/^#?\/?$/, "home", pageHome], [/^#\/new$/, "new", pageNew], [/^#\/runs$/, "runs", pageRuns],
  [/^#\/runs\/([\w.-]+)$/, "runs", pageRun], [/^#\/hpo\/([\w.-]+)$/, "runs", pageHpo],
  [/^#\/models$/, "models", pageModels], [/^#\/models\/([\w.:-]+)$/, "models", pageModel], [/^#\/packs$/, "packs", pagePacks],
];

async function route() {
  timers.forEach(clearInterval);
  timers = [];
  fill(app, h("p", { class: "loading" }, "載入中…"));
  const hash = location.hash || "#/";
  const match = ROUTES.find(([re]) => re.test(hash));
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", match && a.dataset.nav === match[1]));
  if (!match) { app.append(h("h1", {}, "找不到頁面"), h("a", { href: "#/" }, "回總覽")); return; }
  try { await match[2](...hash.match(match[0]).slice(1).map(decodeURIComponent)); }
  catch (e) { app.append(panel(h("h2", {}, "發生錯誤"), h("p", {}, e.message), e.hint ? h("p", { class: "muted" }, e.hint) : null)); }
}

window.addEventListener("hashchange", route);
route();
