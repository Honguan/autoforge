# AutoForge 架構設計 v0.2

> 讓 AI 自己分析需求、選方法、訓練、評估、部署 AI 模型的可安裝平台。
> 本文件依 2026-10-15 需求確認結果撰寫。

## 0. 已確認的需求

| # | 項目 | 決定 |
|---|---|---|
| 1 | 自動化目標 | 兩者都要：(a) 訓練流程自動化，(b) 訓練出的模型用來自動化日常工作 |
| 2 | 支援情境 | 市面常見情境全部支援，依波次推出（見 §6） |
| 3 | AI 大腦 | 訂閱制 AI（透過 MCP）、API 呼叫、本地模型，三種都支援 |
| 4 | 硬體 | 只支援 NVIDIA（CUDA），沒有 GPU 時退回 CPU |
| 5 | 權限 | AI 和人對訓練過程都有完整權限，包含 AI 自己寫訓練程式 |
| 6 | 方法包 | 放在 GitHub，用 AI 好呼叫的方式管理（見 §4） |
| 7 | 介面 | 先把「AI 好呼叫」做好（MCP + JSON CLI + REST），GUI 後做 |
| 8 | 隱私 | 敏感資料只能在本地模型指揮時使用（程式強制執行，見 §5） |
| 9 | 開發 | 一人開發 |
| 10 | 原始碼 | GitHub |

---

## 1. 總體架構

```
           ┌──────────── AI 大腦（三選一）────────────┐
           │ A. 訂閱制 AI   B. API 模式    C. 本地模型   │
           │ (Claude/ChatGPT (Anthropic/   (Ollama/     │
           │  Desktop/Code)  OpenAI/Gemini) vLLM)       │
           └──────┬───────────────┬──────────────┬─────┘
                  │ MCP           │ 內建 Agent 迴圈 │
┌─────────────────▼───────────────▼──────────────▼──────────────┐
│ 介面層：MCP Server │ CLI (--json) │ REST API │ GUI（之後）      │
│        ↑ 全部共用同一套 Core API 和 JSON Schema                │
├────────────────────────────────────────────────────────────────┤
│ 隱私守衛 PrivacyGuard：每次讀資料前檢查「大腦類型 × 資料敏感度」│
├────────────────────────────────────────────────────────────────┤
│ Core API                                                        │
│  hardware │ data │ packs │ plan │ runs │ eval │ registry │ serve │
├────────────────────────────────────────────────────────────────┤
│ 執行引擎：每個 Run 是一個獨立的工作區 + 子行程                    │
│  後端：sklearn/XGBoost │ PyTorch │ HF Transformers+PEFT/TRL │    │
│        Ultralytics │ SB3 │ diffusers │ 自訂程式（AI 撰寫）       │
├────────────────────────────────────────────────────────────────┤
│ 控制通道：pause / resume / stop / 修改參數（人和 AI 都能下指令）  │
│ 稽核日誌：所有動作記錄是誰做的（human / ai:<provider>）          │
└────────────────────────────────────────────────────────────────┘
```

### 1.1 三種大腦模式

| 模式 | 誰在思考 | 怎麼連接 | 適合 |
|---|---|---|---|
| **A. 訂閱制** | 使用者已訂閱的 AI（Claude Desktop、Claude Code、其他支援 MCP 的客戶端） | AutoForge 啟動 MCP Server，AI 把它當工具使用 | 不想另外付 API 費用；互動式工作 |
| **B. API** | AutoForge 內建 Agent 呼叫雲端 API | API Key 設在設定檔或環境變數 | 全自動、排程、批次 |
| **C. 本地** | Ollama / vLLM / llama.cpp 上的本地模型 | OpenAI 相容端點 | **敏感資料**、離線環境 |

模式 A 不需要 AutoForge 內建 LLM，所以最先做（P4a），模式 B、C 共用同一個 Agent 迴圈（P4b）。

### 1.2 模組

| 模組 | 路徑 | 職責 |
|---|---|---|
| hardware | `autoforge/hardware` | 偵測 CPU/RAM/NVIDIA GPU/CUDA/驅動，估算可跑的方法 |
| policy | `autoforge/policy` | 隱私守衛、權限、稽核日誌 |
| schemas | `autoforge/schemas` | 所有資料結構（Pydantic）→ 自動匯出 JSON Schema 給 AI |
| packs | `autoforge/packs` | 方法包搜尋、安裝、驗證、版本鎖定 |
| data | `autoforge/data` | 資料讀取、剖析（只回傳統計，不回傳原始資料）、切分 |
| agents | `autoforge/agents` | 模式 B/C 的內建 Agent：任務分析、規劃、診斷 |
| engine | `autoforge/engine` | Run 生命週期、子行程執行、checkpoint、控制通道 |
| eval | `autoforge/eval` | 評估、錯誤分析、報告 |
| registry | `autoforge/registry` | 模型版本、匯出（safetensors/ONNX/GGUF） |
| serve | `autoforge/serve` | MCP Server、REST API、推論服務 |

---

## 2. 核心資料結構

所有 AI 和人的互動都經過這些結構，並以 JSON Schema 公開（`autoforge schema <name>`）。

- **HardwareProfile**：CPU、RAM、GPU 列表（名稱、VRAM、算力、驅動）、CUDA 版本、torch 是否可用。
- **DatasetProfile**：路徑、筆數、欄位與型別、類別分佈、缺值、`sensitivity`（public / internal / sensitive）。
- **TaskSpec**：任務類型、輸入/輸出模態、目標欄位、評估指標、限制（時間、模型大小、延遲）。
- **TrainingPlan**：選用的 pack + recipe（或自訂程式）、超參數、裝置、預估時間/VRAM、理由。
- **Run**：run_id、狀態（pending/running/paused/succeeded/failed/stopped）、指標歷史、工作區路徑、稽核紀錄。

---

## 3. 一次訓練的生命週期

```
profile_hardware ─┐
profile_dataset ──┼─▶ TaskSpec ─▶ pack_search ─▶ TrainingPlan ─▶ plan_validate
                  │                                               (資源/隱私檢查)
                  │                                                    │
                  │       ┌──────────── 回饋迴圈 ─────────────┐        ▼
                  │       │ run_metrics → 診斷 → run_adjust   │◀── run_start
                  │       └──────────────────────────────────┘        │
                  │                                                    ▼
                  └──────────────────────────────────── evaluate ─▶ register ─▶ deploy
                                                                         (API / MCP 工具)
```

`plan_validate` 是確定性檢查，不靠 AI：VRAM 夠不夠、依賴有沒有裝、隱私規則有沒有違反。AI 規劃錯了也會在這裡被擋下。

---

## 4. 方法包（Pack）與 GitHub

### 4.1 採用方案：「GitHub Repo + 索引分支 + 版本鎖定」

```
GitHub
├── Honguan/autoforge
│   ├── main 分支：packs/tabular-classic/ …        ← 官方方法包放在子目錄
│   └── packs-index 分支（獨立歷史）                ← 中央索引
│       └── index.json   [{name, source, version, commit_sha, tasks, recipes, summary}]
└── <任何人>/autoforge-pack-xxx                    ← 第三方方法包可以是獨立 repo
```

索引放在獨立歷史的分支，好處是不用另外建 repo，又不會和程式碼的 commit 混在一起。
讀取網址：`https://raw.githubusercontent.com/Honguan/autoforge/packs-index/index.json`

為什麼這樣對 AI 最好呼叫：

1. **AI 不用爬 GitHub。** `pack_search(task="object-detection", vram_gb=8)` 直接查 `index.json`，一次呼叫就拿到結構化結果，省 token 又穩定。
2. **安裝時鎖定版本。** `pack_install("github:owner/repo@v1.2.0")` 會記錄 commit SHA，結果可重現，被竄改也能發現。
3. **每個包都有 `KNOWLEDGE.md`。** 給 AI 讀的教學，固定段落：何時用、何時不要用、參數怎麼調、常見失敗與對策。
4. **AI 也能產生新包。** `pack_generate_from_repo("github:huggingface/xxx")`：AI 分析任意 GitHub 訓練專案，轉成 pack 格式，通過 `pack validate` 和內附測試才能安裝（P7）。

### 4.2 Pack 結構

```
autoforge-pack-text-classification/
├── pack.yaml          # 清單（schema 見 autoforge schema PackManifest）
├── KNOWLEDGE.md       # 給 AI 的教學
├── recipes/*.yaml     # 每個方法：需求、預設參數、可調參數範圍
├── trainers/*.py      # 訓練進入點（統一介面 train(plan, ctx) -> RunResult）
├── evaluators/*.py
└── tests/             # 用迷你資料集的煙霧測試，CPU 30 秒內跑完
```

---

## 5. 隱私守衛（需求 #8，程式強制）

| 資料敏感度 | 本地大腦 (C) | API 大腦 (B) | 訂閱制大腦 (A) |
|---|---|---|---|
| public | ✅ | ✅ | ✅ |
| internal | ✅ | ✅（只傳統計摘要） | ✅（只傳統計摘要） |
| sensitive | ✅ | ❌ 拒絕 | ❌ 拒絕 |

- 資料集預設為 `internal`。使用者標成 `public` 才會把原始樣本傳給雲端 AI。
- 檢查放在 Core API 層，CLI、MCP、REST、GUI 都繞不過。
- 拒絕時回傳明確錯誤碼 `PRIVACY_DENIED` 和原因，AI 可以據此提示使用者改用本地模型。

> 訓練本身一定在本機跑，敏感資料不會因為訓練而外傳。這條規則管的是「AI 大腦能不能看到資料」。

---

## 6. 權限與安全（需求 #5）

AI 和人都有完整權限，包含 AI 撰寫並執行自訂訓練程式。為了在完整權限下仍然可以追蹤、可以救回，加上以下機制（都不會限制權限）：

- **稽核日誌**：每個動作記錄 actor（`human` / `ai:anthropic` / `ai:ollama` …）、時間、參數。
- **Run 工作區 git 化**：AI 寫的程式、修改的參數都會 commit，可以 diff 和回滾。
- **子行程執行**：訓練跑在獨立行程，當掉不會拖垮主程式；可以被 stop 指令終止。
- **雙向控制**：人隨時可以 pause、stop、改參數；AI 的指令和人的指令走同一個控制通道，以最後的指令為準。

---

## 7. 情境支援計畫（需求 #2，依波次推出）

| 波次 | 情境 | 主要後端 | 最低硬體 |
|---|---|---|---|
| W1 | 表格分類/回歸、文字分類 | sklearn、XGBoost、Transformers | CPU |
| W2 | LLM 微調（SFT LoRA/QLoRA）、Embedding 微調 | PEFT、TRL、sentence-transformers | 8GB VRAM |
| W3 | 圖片分類、物件偵測、影像分割 | timm、Ultralytics | 4GB VRAM |
| W4 | 時間序列預測、異常偵測、推薦系統 | statsforecast、PyOD、implicit | CPU |
| W5 | 語音辨識微調、LLM 偏好對齊（DPO） | Whisper + PEFT、TRL | 12GB VRAM |
| W6 | 圖像生成 LoRA、強化學習 | diffusers、Stable-Baselines3 | 8GB VRAM / CPU |

---

## 8. AI 好呼叫的介面設計規範

1. **MCP 是一級公民**：每個 Core API 都對應一個 MCP 工具，描述精簡，參數用 JSON Schema。
2. **CLI 全部支援 `--json`**：輸出穩定的 JSON，錯誤也是 JSON：`{"ok": false, "error": {"code": "...", "message": "...", "hint": "..."}}`。
3. **長時間任務非同步**：`run_start` 立刻回傳 `run_id`，之後用 `run_status` / `run_metrics` 輪詢，不會卡住 AI 的工具呼叫。
4. **回傳摘要而不是原始資料**：日誌和指標預設回傳摘要（最近 N 筆 + 趨勢），需要時才分頁取全部。
5. **專案根目錄放 `AGENTS.md`**：AI 讀這一份就知道怎麼使用整個系統。
6. **GUI 之後只是另一個客戶端**：呼叫同一套 REST API，不另外寫邏輯。
