# 部署：註冊、預測、服務、匯出（P6）

訓練好的模型有四種用法：

1. 直接用 CLI 預測
2. 開 REST API 給程式呼叫
3. 變成 MCP 工具給其他 AI 呼叫
4. 匯出到別的地方使用

## 1. 註冊成模型

```bash
autoforge model register <run_id> --name churn --description "客戶流失機率"
autoforge model list
autoforge model info churn            # churn = 最新版本；也可以寫 churn:v1
```

- 每次註冊會產生新版本（v1、v2…）。
- **模型會連同訓練它的程式碼一起保存**：每次訓練開始時，會把方法包的程式碼複製進這次訓練的資料夾，註冊時再一併複製。之後就算方法包升級，舊模型的預測結果也不會改變。
- **模型卡**（`card.json`）記錄：
  - 來源訓練、方法、參數、指標
  - 輸入欄位和型別，類別欄位會附上訓練時出現過的值
  - 分類模型會附上可能的預測結果（labels）
- 訓練資料標記為 sensitive 時，模型卡**不會**記錄類別欄位出現過的值和 labels，避免資料內容外流。
- 只有成功的訓練能註冊。方法有提供預測程式（predictor）時，模型才能用來預測。

## 2. 預測

```bash
autoforge model predict churn --rows '[{"tenure_months": 2, "monthly_fee": 110, "contract": "month-to-month", "support_calls": 5}]'
autoforge model predict churn --file new_customers.csv
```

```json
{"model": "churn:v1", "predictions": [{"prediction": 1, "probabilities": {"0": 0.025, "1": 0.975}}], "warnings": []}
```

送進來的每一列都會先檢查：

| 情況 | 處理方式 |
|---|---|
| 缺少欄位 | 回傳 `INVALID_INPUT` 錯誤，並列出所有需要的欄位 |
| 數字欄位不是數字 | 回傳 `INVALID_INPUT` 錯誤 |
| 類別值在訓練時沒出現過 | 照樣預測，但附上警告 |
| 多送了模型不認得的欄位 | 照樣預測，但附上警告 |

## 3. REST API

```bash
autoforge serve                                  # 所有模型，http://127.0.0.1:8000
autoforge serve -m churn -p 9000                 # 只開放 churn
MY_KEY=… autoforge serve --host 0.0.0.0 --api-key-env MY_KEY   # 開放給其他機器，需要 API key
```

| 方法 | 路徑 | 說明 |
|---|---|---|
| GET | `/health` | 健康檢查 |
| GET | `/v1/models` | 模型清單 |
| GET | `/v1/models/{name}` | 模型卡 |
| POST | `/v1/models/{name}/predict` | 傳入 `{"rows": [...]}`，回傳 `{"ok": true, "data": {"predictions": [...]}}` |

```bash
curl -s localhost:8000/v1/models/churn/predict -H 'content-type: application/json' \
  -d '{"rows":[{"tenure_months":2,"monthly_fee":110,"contract":"month-to-month","support_calls":5}]}'
```

- 預設只綁定 127.0.0.1，同一台電腦以外連不進來。
- 綁定其他網路介面又沒設 API key 時，啟動會顯示警告。
- 錯誤格式和 CLI、MCP 一樣，HTTP 狀態碼：404（找不到模型）、422（輸入錯誤）、401（API key 錯誤）。

## 4. 讓其他 AI 呼叫模型（MCP）

```bash
autoforge serve-mcp                  # 所有可預測的模型
autoforge serve-mcp -m churn         # 只開放 churn
```

- 每個模型會變成一個工具 `predict_<模型名稱>`。
- 工具的輸入格式由模型卡自動產生：欄位名稱、型別、類別值。
- 工具說明會寫明模型預測什麼、可能的結果，以及驗證分數。
- **這個伺服器只能預測**，沒有訓練、讀資料或安裝方法包的功能，適合交給只需要「使用」模型的 AI。

Claude Desktop 的設定：

```json
{
  "mcpServers": {
    "my-models": { "command": "/完整路徑/autoforge", "args": ["serve-mcp"] }
  }
}
```

Claude Code 的設定：

```bash
claude mcp add my-models -- autoforge serve-mcp
```

之後直接問 AI，例如「這位客戶會不會流失：簽約 2 個月、月費 110、月繳、客服來電 5 次」，AI 就會呼叫 `predict_churn`。

主要的 `autoforge mcp` 伺服器也有 `model_register`、`model_predict`、`model_info`、`model_list`、`model_export`，所以 AI 訓練完可以直接註冊模型並試著預測。

## 5. 匯出與匯入

```bash
autoforge model export churn                       # bundle：zip（模型 + 程式碼 + 模型卡）
autoforge model export churn --format onnx         # ONNX（需要 pip install "autoforge[onnx]"）
autoforge model import churn-v1.zip --name churn   # 在另一台電腦匯入，成為新版本
```

- 匯出結果會附上 SHA-256，可以用來確認檔案完整。
- 匯入時會檢查 zip 裡的路徑，拒絕會寫到目標資料夾外面的檔案（zip slip）。
- `tabular-classic` 0.3.0 可以匯出 ONNX。測試中確認過，ONNX 的預測結果和 sklearn 完全一致。

## 方法包作者：預測與匯出介面

在 recipe 裡宣告：

```yaml
predictor: trainers/my_trainer.py           # 必須定義 load_model(model_dir) 和 predict(model, rows)
exporters:
  onnx: trainers/my_trainer.py:export_onnx  # export_onnx(model_dir, out_path, features) -> path
```

- `predict` 收到 `[{欄位: 值}]`，回傳每一列的 `{"prediction": …, "probabilities": {…}}`，數值要轉成 JSON 可序列化的型別。
- AI 自己寫的訓練程式只要也定義了 `load_model` 和 `predict`，註冊後一樣能用來預測。
