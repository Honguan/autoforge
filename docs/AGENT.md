# 內建 AI 代理：一句話完成訓練（P4b）

不需要開任何 AI 客戶端，直接在終端機用一句話描述需求：

```bash
autoforge train "預測 churned 欄位（客戶是否流失）" --data examples/data/churn.csv
```

代理會依序：查看硬體 → 分析資料 → 搜尋並安裝方法包 → 閱讀方法教學 → 寫訓練計畫並驗證 → **詢問你是否開始訓練** → 訓練 → 等待並看結果 → 寫報告。

它和 MCP 使用同一套工具，所以隱私守衛、稽核紀錄、錯誤格式都完全相同。

不想用終端機的話，`autoforge gui` 的「AI 助手」頁面是同一個代理，可以多輪對話，也在頁面上同意開始訓練（見 [GUI.md](GUI.md)）。

## 選擇 AI 大腦（`--brain`）

| `--brain` | 說明 | 類型 |
|---|---|---|
| `anthropic:claude-opus-5-5`（預設） | Claude，用官方 SDK。需要 `pip install "autoforge-trainer[anthropic]"` 和 `ANTHROPIC_API_KEY`（或 `ant auth login`） | 雲端 |
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

## 評估：AI 選方法選得對不對

```bash
autoforge eval agent --brain anthropic:claude-opus-5-5     # 13 個案例
autoforge eval agent --brain ollama:qwen3 -c churn -c faq-search
autoforge eval agent --list                                 # 看有哪些案例
```

每個案例是一句需求加上一份當場產生的小資料，例如「預測哪些客戶會流失」、「找出可疑交易（沒有標記）」、「在照片裡框出瓶子」、「用客服對話微調語言模型」。AI 照平常的流程檢查硬體、分析資料、找方法包、讀說明、寫計畫；**當它要開始訓練時，評估工具記下計畫並停住，不會真的訓練**，所以每個案例只花規劃的那幾輪。

評分：

| 結果 | 意思 |
|---|---|
| ✅ pass | 任務類型、方法包和方法、要預測的欄位都對 |
| 🟡 partial | 方向對但不是最好的選擇（例如資料很少卻直接用 Transformer、多條序列卻一條一條做），或欄位選錯 |
| ❌ fail | 任務類型錯、方法不適合、沒有提出計畫，或拿雲端模型去用敏感資料 |

幾個案例專門測 AI 有沒有讀方法包的說明：有標記的詐騙資料應該訓練分類器，不是異常偵測；多家店的營收應該用多序列共用模型。另有一個標記為敏感的資料：雲端大腦必須在 `PRIVACY_DENIED` 停下來，本地大腦則照常規劃。

目標是通過率至少 80%（路線圖 P4b 的「10 個案例至少 8 個」）。報告存在 `~/.autoforge/evals/<時間>/report.md`，每個案例的完整對話紀錄也在同一個資料夾。評估在暫時的資料夾裡進行，不會動到你自己的訓練紀錄。每個案例大約要 5–20 次模型呼叫，用雲端 API 會產生費用。

