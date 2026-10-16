# 變更紀錄

## 1.3.0 — 2026-10

- **評估 AI 選方法的準確度**：`autoforge eval agent --brain <大腦>`。13 個案例，每個是一句需求加一份當場產生的小資料，涵蓋表格、文字、圖片、物件偵測、時間序列、異常偵測、推薦、文字向量、LLM 微調和敏感資料。AI 要開始訓練時會被攔下並記錄計畫，不會真的訓練；結果分成 pass / partial / fail，報告存在 `~/.autoforge/evals/`。
- 資料分析：資料夾會列出最上層的檔案（例如 `annotations.csv`），JSONL 會列出欄位名稱和筆數（不含內容），讓 AI 分得出物件偵測資料、對話資料和配對資料。

## 1.2.0 — 2026-10

- **網頁介面的 AI 助手**：在 `autoforge gui` 裡和內建 AI 代理對話。可以選 AI 大腦（Claude、OpenAI、本機 Ollama），用一句話說需求；AI 開始訓練、重新訓練或自動調參前會在頁面上請你同意，不同意時可以寫意見讓它修改；訓練完可以接著問。AI 用自己的身分操作，隱私規則和指令列相同。
- AI 代理改成可以多輪對話（`AgentSession`），`autoforge train` 的行為不變。
- 修正幾個還寫著舊套件名稱 `autoforge[...]` 的安裝提示。

## 1.1.0 — 2026-10

### 新方法包（P7 第二批）
- `object-detection`：物件偵測，torchvision Faster R-CNN 微調，支援 CSV 或 COCO 標註，評估 mAP@0.5。
- `image-segmentation`：影像分割，LR-ASPP / DeepLabV3，評估 mIoU，可匯出 ONNX。
- `embedding-finetune`：文字向量微調（搜尋、RAG），對比學習，報告微調前後的 recall@1，可匯出 ONNX。
- `speech-recognition`：Whisper 語音辨識微調，報告微調前後的 WER / CER。
- `llm-preference`：LLM 偏好對齊（DPO + LoRA / QLoRA），可從 `llm-finetune` 的結果接著訓練。
- `diffusion-lora`：Stable Diffusion 1.x/2.x 的 UNet LoRA，生成的圖片在網頁介面直接顯示。
- `reinforcement-learning`：stable-baselines3（PPO / A2C / DQN / SAC），環境可用 Gymnasium 名稱或自己寫的 `make_env()`。
- `recommendation`：推薦系統，隱式回饋 ALS 和 Item-KNN，和熱門排行的基準比較。
- `timeseries-forecast` 0.2.0：新增 `global_lag_gbm`，多條序列共用一個模型。

### 核心
- 安裝方法包時檢查 `min_autoforge`，版本太舊會回 `PACK_NEEDS_NEWER_AUTOFORGE`。
- 模型輸入新增 `audio` 類型（REST、MCP、網頁介面）。
- 自動調參能判斷 WER、CER、MAPE、WAPE 是越低越好。
- `ctx.check_updates()` 只在控制檔變動時才讀檔，每一步呼叫也沒有負擔。
- 新的安裝選項 `[rl]`；`[deep]` 加上 diffusers、soundfile。

### 已知限制
- 新的 GPU 方法（偵測、分割、Whisper、DPO、Diffusion LoRA）只用極小模型在 CPU 上驗證過流程，尚未在真的 NVIDIA GPU 上跑過。

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
