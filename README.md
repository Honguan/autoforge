# AutoForge

讓 AI 自己分析需求、選擇方法、訓練、評估、部署 AI 模型的可安裝平台。

- **三種 AI 大腦**：訂閱制 AI（透過 MCP）、雲端 API、本地模型
- **硬體**：NVIDIA GPU（CUDA），沒有 GPU 時自動改用 CPU
- **方法包**：從 GitHub 安裝的「教學包」，AI 讀了就知道怎麼訓練
- **隱私**：敏感資料只允許本地模型指揮

> **1.0.2** · 文件網站：<https://honguan.github.io/autoforge/> · [變更紀錄](CHANGELOG.md)

## 安裝與第一次使用

```bash
pip install autoforge-trainer          # 需要 Python 3.11 以上
autoforge quickstart              # 約 1 分鐘：檢查硬體 → 安裝方法包 → 訓練 → 註冊 → 預測
autoforge gui                     # 網頁介面
```

| 安裝方式 | 包含 |
|---|---|
| `pip install autoforge-trainer` | 表格、文字（TF-IDF）、時間序列、異常偵測、自動調參、網頁介面、MCP、REST |
| `pip install "autoforge-trainer[deep]"` | 再加上 Transformer、LLM 微調（LoRA/QLoRA）、圖片分類（NVIDIA GPU） |
| `pip install "autoforge-trainer[all]"` | 再加上 Claude 代理、ONNX 匯出 |
| `docker run --rm -p 127.0.0.1:8800:8800 -v autoforge:/data ghcr.io/honguan/autoforge:cpu` | CPU 映像，預設啟動網頁介面 |
| `docker run --rm --gpus all -p 127.0.0.1:8800:8800 -v autoforge:/data ghcr.io/honguan/autoforge:cuda` | NVIDIA CUDA 映像，所有方法包 |

PyPI 上的 `autoforge` 是另一個不相關的專案（`autoforge-ai`、`autoforge-ml` 也都無法使用），請安裝 **`autoforge-trainer`**；裝好之後指令一樣是 `autoforge`。

操作方式：[網頁介面](docs/GUI.md) · [一句話訓練](docs/AGENT.md) · [讓訂閱制 AI 透過 MCP 操作](docs/MCP.md) · [改進模型](docs/IMPROVE.md) · [部署](docs/DEPLOY.md) · [架構](docs/ARCHITECTURE.md) · [路線圖](docs/ROADMAP.md)

## 內建方法包

| 方法包 | 用途 | 方法 | 硬體 |
|---|---|---|---|
| `tabular-classic` | 表格分類、回歸 | 邏輯迴歸、隨機森林（可匯出 ONNX） | CPU |
| `text-classification` | 文字分類（情緒、意圖、主題、垃圾訊息） | TF-IDF + 線性模型；Transformer 微調 | CPU / GPU |
| `llm-finetune` | 大型語言模型微調 | LoRA / QLoRA（可匯出合併後的權重） | GPU（8GB+） |
| `image-classification` | 圖片分類 | timm 預訓練模型微調（可匯出 ONNX） | GPU 建議，CPU 可跑小模型 |
| `timeseries-forecast` | 時間序列預測（銷量、用電、流量） | 落後特徵 + 梯度提升，和「明天＝今天」的基準比較 | CPU |
| `anomaly-detection` | 異常偵測（詐欺、故障、離群值） | Isolation Forest | CPU |

每個方法包都附給 AI 讀的 `KNOWLEDGE.md`，以及在 CPU 上 30 秒內跑完的測試。自己做方法包：`autoforge pack new`（見 [docs/PACK_SPEC.md](docs/PACK_SPEC.md)）。

## 從原始碼開發

```bash
git clone https://github.com/Honguan/autoforge && cd autoforge
pip install -e ".[dev]"            # 加上 ,deep 安裝深度學習方法包的依賴
pytest -q
```

## 網頁介面

```bash
autoforge gui      # 打開瀏覽器：上傳資料 → 選任務 → 選方法 → 訓練 → 調參 → 註冊 → 預測
```

## 使用

```bash
autoforge doctor               # 偵測硬體
autoforge doctor --json        # 給 AI 讀的 JSON
autoforge schema               # 列出所有資料結構
autoforge schema TrainingPlan --json

# 方法包
autoforge pack search csv --fit            # 搜尋這台電腦跑得動的方法包
autoforge pack install tabular-classic    # 從索引安裝（鎖定 commit）
autoforge pack install ./packs/tabular-classic   # 從本地資料夾安裝
autoforge pack install github:owner/repo/subdir@v1.0.0
autoforge pack info tabular-classic --json       # 給 AI 讀的知識和方法
autoforge pack list

# 訓練
autoforge plan validate examples/churn_plan.yaml     # 檢查訓練計畫
autoforge run start examples/churn_plan.yaml --wait  # 訓練並即時顯示進度
autoforge run start plan.yaml --json                 # 背景訓練，立即回傳 run_id（給 AI 用）
autoforge run status <run_id>
autoforge run pause|resume|stop <run_id>
autoforge run metrics|logs|audit <run_id>
autoforge run retry <run_id>                         # 同計畫重跑，從 checkpoint 接續
```

每次訓練的所有檔案都在 `~/.autoforge/runs/<run_id>/`：計畫、模型、checkpoint、日誌、指標、稽核紀錄，計畫和 AI 寫的程式會用 git 版本管理。

方法包格式見 [docs/PACK_SPEC.md](docs/PACK_SPEC.md)，索引放在本 repo 的 [`packs-index` 分支](https://github.com/Honguan/autoforge/tree/packs-index)。

## 改進模型

```bash
autoforge run diagnose <run_id>                                       # 找問題、給建議
autoforge hpo start examples/churn_plan.yaml -m f1_macro -n 20 --wait # 自動調參（trial 0 = 基準）
autoforge run adjust <run_id> --set learning_rate=0.0003              # 訓練中途改參數
autoforge run compare <id1> <id2> -m f1_macro                         # 比較
```

## 部署

```bash
autoforge model register <run_id> --name churn      # 註冊成模型（v1、v2…）
autoforge model predict churn --rows '[{...}]'       # 預測
autoforge serve                                      # REST API（127.0.0.1:8000）
autoforge serve-mcp                                  # 每個模型變成 MCP 工具 predict_<名稱>，給其他 AI 呼叫
autoforge model export churn --format onnx           # 匯出
```

## 一句話訓練（內建 AI 代理）

```bash
pip install -e ".[anthropic]"
autoforge train "預測 churned 欄位" --data examples/data/churn.csv                   # Claude（預設）
autoforge train "預測 churned 欄位" --data examples/data/churn.csv --brain ollama:qwen3   # 本地模型
```

詳見 [docs/AGENT.md](docs/AGENT.md)。

## 讓 AI 操作（MCP）

```bash
claude mcp add autoforge -- autoforge mcp --actor ai:claude-code   # Claude Code
```

Claude Desktop 和隱私設定見 [docs/MCP.md](docs/MCP.md)。

```bash
autoforge data mark ~/data/customers -s sensitive   # 敏感資料：雲端 AI 不能使用
```

## 測試

```bash
pytest
```
