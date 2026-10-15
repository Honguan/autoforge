# 用訂閱制 AI 操作 AutoForge（MCP）

AutoForge 內建 MCP Server。Claude Desktop、Claude Code 或任何支援 MCP 的客戶端都可以直接呼叫它：分析硬體和資料、挑方法、訓練、監控、控制。用的是你已經訂閱的 AI，不需要另外付 API 費用。

## 1. 安裝

```bash
pip install -e ".[gpu]"     # 或 pip install -e .（只用 CPU）
which autoforge              # 記下完整路徑，Windows 用 where autoforge
```

## 2. 連接客戶端

### Claude Code

```bash
claude mcp add autoforge -- autoforge mcp --actor ai:claude-code
```

### Claude Desktop

編輯 `claude_desktop_config.json`（macOS：`~/Library/Application Support/Claude/`，Windows：`%APPDATA%\Claude\`）：

```json
{
  "mcpServers": {
    "autoforge": {
      "command": "/完整路徑/autoforge",
      "args": ["mcp", "--actor", "ai:claude-desktop"]
    }
  }
}
```

存檔後重新啟動 Claude Desktop。`command` 要填第 1 步記下的完整路徑，虛擬環境裡的 autoforge 也一樣。

### 其他 MCP 客戶端

用 stdio 執行 `autoforge mcp`。

## 3. 開始使用

直接用自然語言下指令，例如：

> 用 /home/me/data/churn.csv 訓練一個預測 churned 欄位的模型，先看看我的硬體和資料，選最適合的方法。

AI 會依序呼叫：`hardware_profile` → `data_profile` → `pack_search` → `pack_install` → `pack_info`（讀方法教學）→ `plan_validate` → `run_start` → `run_wait`。

## 4. 工具清單

| 類別 | 工具 |
|---|---|
| 環境 | `hardware_profile`、`schema_get` |
| 方法包 | `pack_search`、`pack_install`、`pack_info`、`pack_list` |
| 資料 | `data_profile` |
| 自訂程式 | `trainer_write`（AI 自己寫訓練程式） |
| 計畫 | `plan_validate` |
| 訓練 | `run_start`、`run_wait`、`run_status`、`run_list`、`run_metrics`、`run_logs`、`run_audit` |
| 控制 | `run_pause`、`run_resume`、`run_stop`、`run_retry` |

所有工具都回傳 `{"ok": bool, "data": ..., "error": {code, message, hint}}`。

## 5. 只開放「使用模型」：`autoforge serve-mcp`

如果只想讓 AI 呼叫訓練好的模型，不讓它訓練或讀取資料，可以改用 `autoforge serve-mcp`：每個模型會變成一個 `predict_<名稱>` 工具。詳見 [DEPLOY.md](DEPLOY.md)。

## 6. 人和 AI 一起操作

AI 透過 MCP 開始的訓練，你也可以從 CLI 看狀態和控制，反過來也一樣：

```bash
autoforge run list
autoforge run pause <run_id>
autoforge run audit <run_id>     # 看哪些動作是你做的、哪些是 AI 做的
```

`--actor` 設定的名稱會記錄在稽核紀錄裡。

## 7. 隱私（需求 #8）

用 CLI 標記資料的敏感度。只有人能標記，MCP 沒有提供這個工具，所以 AI 不能降低資料的敏感度。

```bash
autoforge data mark ~/data/customers -s sensitive   # 標記資料夾會套用到裡面所有檔案
autoforge data mark ~/data/iris.csv -s public
autoforge data list
```

| 敏感度 | 雲端/訂閱制 AI 透過 MCP | 本地模型（`--actor ai:local:<模型>`） | 人（CLI） |
|---|---|---|---|
| public | 統計 + 樣本列 | 完整 | 完整 |
| internal（預設） | 只有統計，看不到任何一列原始資料 | 完整 | 完整 |
| sensitive | **拒絕**：不能看、不能驗證計畫、不能訓練 | 完整 | 完整 |

- 統計資料裡，只有不同值 ≤ 20 種的欄位會顯示常見值，避免把姓名、ID 這類識別欄位傳給雲端 AI。
- `--actor ai:local:<模型>` 只能用在真的由本地模型驅動的 MCP 客戶端，這個設定由你負責。
