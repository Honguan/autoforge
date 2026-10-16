# 變更紀錄

## 1.0.2 — 2026-10

- **PyPI 套件名稱改為 `autoforge-trainer`**：`autoforge-ml` 在我們發佈前被別人註冊了。指令和 `import autoforge` 都不變。1.0.1 沒有上傳到 PyPI。

## 1.0.1 — 2026-10

- **PyPI 套件名稱改為 `autoforge-ml`**：`autoforge-ai` 和 PyPI 上既有的 `autoforgeai` 太相似，無法註冊。指令和 `import autoforge` 都不變。
- 修正：`run adjust` 的參數修改偶爾沒有寫進訓練資料夾的 git 紀錄（稽核紀錄不受影響）。原因是 `git add -A` 剛好碰到訓練程式寫檔時暫時產生的檔案；現在只加入需要保存版本的檔案。這個問題是 CI 發現的。

## 1.0.0 — 2026-10

第一個正式版本。PyPI 套件名稱為 `autoforge-ai`，指令仍是 `autoforge`。

### 功能（依開發階段）
- **P1 骨架**：硬體偵測（NVIDIA GPU / CUDA），所有指令都能輸出 JSON，資料結構可匯出 JSON Schema。
- **P2 方法包**：`pack.yaml` 格式、驗證（不執行程式碼）；從本地資料夾或 GitHub 安裝並鎖定 commit；索引放在 `packs-index` 分支。
- **P3 執行引擎**：每次訓練在獨立行程執行；可以暫停、繼續、停止，也能從 checkpoint 重跑；稽核紀錄寫明是人還是 AI 操作；計畫和程式碼用 git 保存版本。
- **P4a MCP**：`autoforge mcp`，讓訂閱制 AI（Claude Desktop、Claude Code）直接操作；隱私守衛：敏感資料只給本地模型使用。
- **P4b 內建 AI 代理**：`autoforge train "需求"`，支援 Claude（官方 SDK）、OpenAI 相容 API、Ollama 和本地模型；開始訓練前會請人確認。
- **P5 改進模型**：診斷、自動調參（Optuna，第 0 次為基準）、訓練中途調整參數、比較多次訓練。
- **P6 部署**：模型倉庫（模型會連同訓練它的程式碼一起保存）、REST API、每個模型變成一個 MCP 工具、匯出 bundle 或 ONNX。
- **P7 方法包**：tabular-classic、text-classification、llm-finetune（LoRA/QLoRA）、image-classification、timeseries-forecast、anomaly-detection；另有 `pack new`、`pack test`、MCP `pack_create`。
- **P8 網頁介面**：`autoforge gui`，只在本機開放，用一次性 token 保護。
- **P9 發佈**：`autoforge quickstart`、PyPI 打包、Docker 映像（CPU 和 CUDA）、文件網站、自動發佈流程。

### 已知限制
- GPU 方法（Transformer、LLM LoRA/QLoRA、timm）只用極小模型在 CPU 上驗證過程式能跑通，尚未在真的 NVIDIA GPU 上實測。
- 自動調參的最佳分數是用同一份驗證資料挑出來的，會稍微偏樂觀。
- 內建 AI 代理選方法的品質，還沒用真的 Claude API 做系統性評估。
