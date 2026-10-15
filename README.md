# AutoForge 方法包索引

[AutoForge](https://github.com/Honguan/autoforge) 的中央方法包索引，放在 `packs-index` 分支（和程式碼分開的獨立歷史）。AI 只要讀一個 `index.json`，就能找到適合任務和硬體的方法包，不必爬整個 GitHub。

## 使用

```bash
autoforge pack search csv --fit --json      # 搜尋
autoforge pack install tabular-classic      # 安裝索引中鎖定的 commit
```

AutoForge 預設讀取：
`https://raw.githubusercontent.com/Honguan/autoforge/packs-index/index.json`
可用環境變數 `AUTOFORGE_INDEX_URL` 改成其他網址或本地檔案。

## 目前收錄

| 方法包 | 任務 | 硬體 |
|---|---|---|
| `tabular-classic` 0.2.0 | 表格分類、回歸 | CPU |

## 新增或更新方法包

1. 方法包必須通過 `autoforge pack validate <path>`（格式見 [PACK_SPEC](https://github.com/Honguan/autoforge/blob/main/docs/PACK_SPEC.md)）。
2. 取得要發佈版本的 **完整 40 字元 commit SHA**。
3. 更新索引：
   ```bash
   autoforge index add index.json <本地包路徑> \
     --source github:<owner>/<repo>[/<subdir>] \
     --commit <sha> --tag <關鍵字>
   autoforge index validate index.json
   ```
4. 發 PR 到 `packs-index` 分支。CI 會自動驗證 `index.json`。

## 安全

- 每個條目都鎖定 commit SHA，安裝時會比對，不一致就拒絕（`SHA_MISMATCH`）。
- 更新版本等於換新的 SHA，必須經過 PR 審核。
