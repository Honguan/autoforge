# AutoForge 開發路線圖（一人開發版）

每個階段的固定流程：**寫規格 → 寫測試 → 實作 → 用真實資料跑一次 → 更新文件 → 打 tag**。
一個階段的驗收標準全部通過，才進入下一階段。

| 階段 | 內容 | 交付物 | 驗收標準 | 預估 |
|---|---|---|---|---|
| **P1 骨架** ✅ | 專案結構、CLI、Schema、硬體偵測、隱私守衛 | `autoforge doctor --json`、`autoforge schema` | 正確偵測 NVIDIA GPU/CUDA；無 GPU 時回報 CPU；測試全過 | 1 週 |
| **P2 方法包系統** ✅ | pack.yaml 格式、本地/GitHub 安裝、索引搜尋、驗證 | `autoforge pack search/install/list/validate` | 從 GitHub 安裝並鎖定 SHA；格式錯誤的包被拒絕 | 1–2 週 |
| **P3 執行引擎** ✅ | Run 生命週期、子行程、checkpoint、控制通道、稽核 | W1 兩個方法包（表格、文字分類） | 手寫 Plan 就能跑完訓練→評估→存模型；pause/resume/stop 正常 | 2–3 週 |
| **P4a MCP Server** | 把 Core API 包成 MCP 工具 | `autoforge mcp` | Claude Desktop/Code 透過訂閱完成一次端到端訓練 | 1 週 |
| **P4b 內建 Agent** | API 與本地模型的 Agent 迴圈 | `autoforge train "需求" --data x --brain ollama:qwen` | 10 個測試案例中 ≥8 個選出合理方法；敏感資料只允許本地 | 2 週 |
| **P5 回饋迴圈** | 指標監控、診斷、Optuna、自訂程式撰寫 | `run_adjust`、`hpo` | 自動調參後指標優於初次訓練 | 2 週 |
| **P6 部署** | 模型倉庫、匯出、推論 API、模型變 MCP 工具 | `autoforge serve <model>` | 訓練好的模型能被其他 AI 透過 MCP 呼叫 | 1–2 週 |
| **P7 擴充** | W2–W6 方法包、`pack_generate_from_repo` | 更多方法包 | 每個包都附 KNOWLEDGE.md 和煙霧測試 | 持續 |
| **P8 GUI** | 網頁介面（呼叫 REST API） | `autoforge gui` | 不碰指令列也能完成一次訓練 | 3 週 |
| **P9 發佈** | PyPI、Docker（CUDA image）、文件站 | v1.0 | 乾淨機器上 `pip install` 後 5 分鐘內跑完範例 | 1 週 |

## 一人開發的建議

- 每個階段先完成「最窄但能端到端跑通」的版本，再加寬功能。
- P3 完成前不寫任何 LLM 相關程式。先確定引擎不靠 AI 也正確，之後 AI 出錯時才分得清是誰的問題。
- 方法包的煙霧測試必須能在 CPU 上 30 秒內跑完，CI（GitHub Actions）才跑得動。

## 實作備註

- **P3**：訓練中途改參數（「控制通道：修改參數」）需要訓練程式配合讀取新參數，和自動調參一起放到 P5。
