# AutoForge

**讓 AI 自己訓練 AI。** AutoForge 會看你的硬體和資料，挑選合適的訓練方法，然後在你自己的電腦上訓練、調參，最後把模型部署成 API 或 MCP 工具。

你可以用四種方式操作它：

| 方式 | 適合 |
|---|---|
| [網頁介面](GUI.md)：`autoforge gui` | 不想打指令 |
| [一句話訓練](AGENT.md)：`autoforge train "預測客戶流失" --data data.csv` | 讓 AI 代理全程處理（Claude、OpenAI、Ollama 本地模型） |
| [MCP](MCP.md)：`autoforge mcp` | 已經在用 Claude Desktop / Claude Code 等訂閱制 AI |
| CLI：`autoforge run start plan.yaml --json` | 寫程式、自動化 |

## 安裝

```bash
pip install autoforge-trainer               # 表格、文字、時間序列、異常偵測（CPU 就能跑）
pip install "autoforge-trainer[deep]"       # 加上 Transformer、LLM 微調與 DPO、影像、語音、圖像生成（NVIDIA GPU）
pip install "autoforge-trainer[all]"        # 再加上強化學習、Claude 代理和 ONNX 匯出
```

需要 Python 3.11 以上。GPU 只支援 NVIDIA（CUDA）。

## 第一次使用

```bash
autoforge quickstart
```

這一個指令會依序完成：檢查硬體 → 安裝方法包 → 訓練客戶流失模型 → 註冊模型 → 預測，大約一分鐘。在全新的電腦上（只有 Python，沒有 git），包含 `pip install` 總共約 63 秒。

## Docker

```bash
docker run --rm -p 127.0.0.1:8800:8800 -v autoforge:/data ghcr.io/honguan/autoforge:cpu             # CPU
docker run --rm --gpus all -p 127.0.0.1:8800:8800 -v autoforge:/data ghcr.io/honguan/autoforge:cuda  # NVIDIA GPU
```

接著打開 <http://127.0.0.1:8800/>。執行其他指令：`docker run --rm -v autoforge:/data ghcr.io/honguan/autoforge:cpu quickstart`

## 內建方法包

| 方法包 | 用途 | 硬體 |
|---|---|---|
| `tabular-classic` | 表格分類、回歸 | CPU |
| `text-classification` | 文字分類（TF-IDF、Transformer 微調） | CPU / GPU |
| `llm-finetune` | LLM 微調（LoRA / QLoRA） | GPU 8GB+ |
| `image-classification` | 圖片分類（timm） | GPU 建議 |
| `timeseries-forecast` | 時間序列預測（單一或多條序列） | CPU |
| `anomaly-detection` | 異常偵測 | CPU |
| `object-detection` | 物件偵測（Faster R-CNN） | GPU 建議 |
| `image-segmentation` | 影像分割（LR-ASPP / DeepLabV3） | GPU 建議 |
| `embedding-finetune` | 文字向量微調（搜尋、RAG） | CPU / GPU |
| `speech-recognition` | 語音辨識（Whisper 微調） | GPU 建議 |
| `llm-preference` | LLM 偏好對齊（DPO） | GPU 8GB+ |
| `diffusion-lora` | 圖像生成 LoRA（Stable Diffusion） | GPU 8GB+ |
| `reinforcement-learning` | 強化學習（stable-baselines3） | CPU / GPU |
| `recommendation` | 推薦系統（ALS、Item-KNN） | CPU |

自己做方法包，或讓 AI 把 GitHub 上的訓練專案轉成方法包，請看[方法包格式](PACK_SPEC.md)。

## 隱私

用 `autoforge data mark <路徑> -s sensitive` 標記的資料，只有本地模型（例如 Ollama）或人可以使用；雲端 AI 對一般資料只看得到統計數字。詳見 [MCP 隱私](MCP.md)。
