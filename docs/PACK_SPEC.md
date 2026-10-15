# 方法包（Pack）格式規格 v1

方法包是 AI 用來學會「怎麼訓練某一類模型」的單位，內容包含給 AI 讀的知識和確定性的訓練程式。

## 1. 目錄結構

```
<pack-name>/
├── pack.yaml           必要：清單
├── KNOWLEDGE.md        必要：給 AI 讀的教學（至少 50 字）
├── recipes/<id>.yaml   必要：每個方法一個檔案
├── trainers/*.py       必要：訓練進入點
└── tests/              建議：煙霧測試，CPU 30 秒內跑完
```

完整欄位定義可用 `autoforge schema PackManifest --json` 和 `autoforge schema RecipeSpec --json` 取得。

## 2. pack.yaml

| 欄位 | 必要 | 說明 |
|---|---|---|
| `name` | ✅ | 小寫 kebab-case，例如 `text-classification` |
| `version` | ✅ | semver，例如 `1.0.0` |
| `summary` | ✅ | 一句話說明 |
| `tasks` | ✅ | 支援的任務類型 |
| `recipes` | ✅ | 方法 id 列表，對應 `recipes/<id>.yaml` |
| `dependencies` | | pip 套件需求 |
| `authors`, `license`, `min_autoforge` | | |

## 3. recipes/&lt;id&gt;.yaml

| 欄位 | 說明 |
|---|---|
| `id` | 小寫英數和 `_`，必須和檔名相同 |
| `task_types` | 必須是 pack.yaml `tasks` 的子集 |
| `entrypoint` | `trainers/file.py:function`，不能指到包外面 |
| `requires` | `device`（cpu / cuda / any）、`min_vram_gb`、`min_ram_gb` |
| `good_for` / `avoid_when` | 給 AI 選方法用的一句話 |
| `defaults` | 預設超參數 |
| `search_space` | 自動調參範圍：`{type, low, high, log}` 或 `{choices}` |

## 4. KNOWLEDGE.md 建議段落

`When to use`、`When NOT to use`、`Choosing a recipe`、`Data requirements`、`Metrics`、`Tuning guide`、`Common failures`

寫法重點：用表格和條列，寫具體數字（例如「minority < 20% 就用 class_weight: balanced」），不要寫空泛的建議。

## 5. 訓練程式介面（暫定，P3 定案）

```python
def train(ctx: dict) -> dict:
    # ctx: data_path, target, task_type, hyperparameters, output_dir, seed
    return {"metrics": {...}, "model_path": "..."}
```

## 6. 驗證規則（`autoforge pack validate`）

驗證時不會 import 或執行包裡的程式碼，entrypoint 是用 `ast` 解析原始碼確認函式存在。檢查項目：

- 必要檔案存在，YAML 格式正確
- name 和 version 格式正確，recipe id 不重複且和檔名一致
- task_types 都有在 pack.yaml 宣告
- entrypoint 檔案和函式都存在，而且沒有指到包外
- `device: cpu` 的方法不能要求 VRAM

## 7. 發佈到索引

1. 把包放到 GitHub（獨立 repo 或某個 repo 的子目錄）
2. 取得要發佈的 commit SHA
3. 在索引 repo 執行：
   ```bash
   autoforge index add index.json ./<pack> --source github:<owner>/<repo>[/<subdir>] --commit <sha> --tag <keyword>
   autoforge index validate index.json
   ```
4. 發 PR 到索引 repo

安裝時，AutoForge 會確認抓下來的 commit 和索引記錄的 SHA 一致，不一致就拒絕安裝（`SHA_MISMATCH`）。
