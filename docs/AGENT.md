# 內建 AI 代理：一句話完成訓練（P4b）

不需要開任何 AI 客戶端，直接在終端機用一句話描述需求：

```bash
autoforge train "預測 churned 欄位（客戶是否流失）" --data examples/data/churn.csv
```

代理會依序：查看硬體 → 分析資料 → 搜尋並安裝方法包 → 閱讀方法教學 → 寫訓練計畫並驗證 → **詢問你是否開始訓練** → 訓練 → 等待並看結果 → 寫報告。

它和 MCP 使用同一套工具，所以隱私守衛、稽核紀錄、錯誤格式都完全相同。

## 選擇 AI 大腦（`--brain`）

| `--brain` | 說明 | 類型 |
|---|---|---|
| `anthropic:claude-opus-5-5`（預設） | Claude，用官方 SDK。需要 `pip install "autoforge[anthropic]"` 和 `ANTHROPIC_API_KEY`（或 `ant auth login`） | 雲端 |
| `openai:<模型>` | OpenAI，或任何雲端的 OpenAI 相容 API（搭配 `--base-url`、`--api-key-env`） | 雲端 |
| `ollama:<模型>` | 本機的 Ollama，例如 `ollama:qwen3`、`ollama:llama3.1` | **本地** |
| `local:<模型> --base-url http://…/v1` | 其他本地 OpenAI 相容伺服器（vLLM、LM Studio、llama.cpp） | **本地** |

也可以用環境變數 `AUTOFORGE_BRAIN=ollama:qwen3` 設定預設值。

**只有本地大腦能使用標記為 sensitive 的資料**（需求 #8）。雲端大腦對一般資料只看得到統計數字。

### Claude 的設定

- 模型預設 `claude-opus-5-5`，effort 設為 `high`。
- 開啟伺服器端拒答備援（`fallbacks: "default"`）：模型因安全分類器拒答時，API 會自動改用 Anthropic 建議的備援模型。
- 每輪的完整回應會原樣接回對話，對話歷史只增不改，並開啟 prompt caching。

### Ollama 的設定

1. 安裝 Ollama，然後 `ollama pull qwen3`（或其他支援工具呼叫的模型）。
2. **把 context 調大。** Ollama 預設 4096 tokens，裝不下代理的系統提示和 21 個工具說明。請用以下方式啟動：
   ```bash
   OLLAMA_CONTEXT_LENGTH=16384 ollama serve
   ```
3. 執行 `autoforge train "…" --data … --brain ollama:qwen3`。

小模型（例如 1–4B）可以跑完流程，但判斷力有限。用越大的模型，選方法、調參數的品質越好。

## 人工確認

預設在開始訓練（`run_start`、`run_retry`）前會停下來，顯示計畫內容並詢問：

```
The agent wants to start training:
  task:            tabular-classification (target: churned, metric: f1_macro)
  data:            /…/churn.csv
  method:          tabular-classic / logistic_regression
  hyperparameters: {"class_weight": "balanced"}
Start? [y = yes / n = no / or type feedback for the agent]
```

- 輸入 `y` 開始訓練。
- 輸入 `n` 拒絕。
- 直接打字，例如「先用 random_forest」，代理會收到你的意見並修改計畫。

加 `--yes` 會跳過確認，全自動執行。

## 紀錄

- 每次代理執行的完整過程（文字、工具呼叫、結果、你的確認）都存在 `~/.autoforge/agent-sessions/<時間>.jsonl`。
- 代理開始的訓練用 `autoforge run list`、`autoforge run audit <run_id>` 查看。actor 會記成 `ai:anthropic:<模型>` 或 `ai:local:ollama:<模型>`。

## 給程式呼叫

```bash
autoforge train "…" --data … --brain ollama:qwen3 --yes --json
```

回傳 `{"ok": true, "data": {"stop", "steps", "runs", "final_text", "transcript"}}`。`--json` 必須搭配 `--yes`。
