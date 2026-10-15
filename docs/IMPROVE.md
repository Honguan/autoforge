# 改進模型：診斷、自動調參、即時調整、比較（P5）

訓練出第一個能用的模型（基準）之後，用這四個工具改進它。人用 CLI，AI 用同名的 MCP 工具，兩邊功能相同。

## 1. 診斷：`run diagnose`（MCP：`run_diagnose`）

```bash
autoforge run diagnose <run_id>
```

這是確定性的規則檢查，不需要 AI，回傳問題清單和建議做法：

| 代碼 | 意思 | 建議 |
|---|---|---|
| `MISSING_PACKAGE` | 缺少 Python 套件 | `pip install …` |
| `GPU_OOM` / `RAM_OOM` | 顯存或記憶體不足 | 調小 batch_size、換小模型、用 QLoRA |
| `RARE_CLASS` | 某個類別的資料太少，沒辦法切分 | 合併或刪除稀有類別 |
| `MISSING_COLUMN` / `NON_NUMERIC` / `FILE_NOT_FOUND` | 資料問題 | 檢查欄位和路徑 |
| `DIVERGED` | loss 變成 NaN 或無限大 | 學習率除以 10、加梯度裁剪 |
| `LOSS_RISING` / `PLATEAU` | loss 回升，或停滯不動 | 調整學習率、提早停止 |
| `OVERFITTING` | 驗證 loss 回升，但訓練 loss 還在下降 | 提早停止、加強正則化 |
| `POSSIBLE_LEAKAGE` | 分數幾乎完美 | 檢查是否有欄位洩漏了答案 |

結果也會列出這個方法可以調的超參數。

## 2. 自動調參：`hpo`（MCP：`hpo_start` / `hpo_wait` / `hpo_status` / `hpo_stop`）

```bash
autoforge hpo start examples/churn_plan.yaml -m f1_macro -n 20 --wait
```

- 使用 Optuna（`pip install "autoforge[hpo]"`），在背景執行。每個 trial 都是一般的 Run，所以有完整紀錄、日誌，也能暫停和停止。
- **trial 0 一定用原計畫的超參數**，當作基準，最後會報告比基準好了多少。
- 搜尋範圍預設用 recipe 的 `search_space`，也可以用 `--space space.yaml` 自訂。
- 最佳化方向會依指標名稱自動判斷：名稱含 loss、rmse、error 的會取最小值，其他取最大值。也可以用 `--direction` 指定。
- 可以用 `--timeout-min` 限制總時間。

**實測（範例資料 churn.csv，random_forest，20 個 trial）：**

```
trial 0 (baseline): f1_macro=0.679  {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 1}
...
trial 12:           f1_macro=0.7306 {'n_estimators': 145, 'max_depth': 12, 'min_samples_leaf': 8}
best=0.7306 baseline=0.679 (+0.0517 vs baseline)
```

> 注意：目前每個 trial 都用同一份驗證資料打分，再從中挑最好的，所以最佳分數會稍微偏樂觀。要得到公正的估計，需要另外保留測試集或用交叉驗證（列入後續的評估工作）。

## 3. 訓練中途調整：`run adjust`（MCP：`run_adjust`）

```bash
autoforge run adjust <run_id> --set learning_rate=0.0003 --set dropout=0.2
```

- 不用重新開始，直接修改正在跑的訓練的參數。每次修改都會記錄是誰改的，並用 git 保存版本（`adjustments.yaml`）。
- 訓練程式需要在步驟之間呼叫 `ctx.check_updates()`，修改才會生效：

```python
for step in range(...):
    ctx.check_stop()
    changed = ctx.check_updates()        # 例如 {"learning_rate": 0.0003}，已經套用到 ctx.hyperparameters
    if "learning_rate" in changed:
        for g in optimizer.param_groups:
            g["lr"] = ctx.hyperparameters["learning_rate"]
```

## 4. 比較：`run compare`（MCP：`run_compare`）

```bash
autoforge run compare <id1> <id2> <id3> -m f1_macro
```

並排列出各個 Run 的指標和超參數，標出哪些超參數不同，並用 ★ 標出最好的那個。

## AI 代理怎麼用這些工具

`autoforge train` 的代理和 MCP 客戶端都會被告知以下流程：先訓練基準 → 用 `run_diagnose` 找問題 → 用 `hpo_start` 調參 → 報告比基準好了多少。

開始自動調參前，跟開始訓練一樣，會先請你確認。
