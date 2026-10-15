# AutoForge

讓 AI 自己分析需求、選擇方法、訓練、評估、部署 AI 模型的可安裝平台。

- **三種 AI 大腦**：訂閱制 AI（透過 MCP）、雲端 API、本地模型
- **硬體**：NVIDIA GPU（CUDA），沒有 GPU 時自動改用 CPU
- **方法包**：從 GitHub 安裝的「教學包」，AI 讀了就知道怎麼訓練
- **隱私**：敏感資料只允許本地模型指揮

> 目前狀態：**P1 骨架**。完整設計見 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，進度見 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 安裝

```bash
pip install -e ".[dev]"        # CPU
pip install -e ".[dev,gpu]"    # NVIDIA GPU
```

## 使用

```bash
autoforge doctor               # 偵測硬體
autoforge doctor --json        # 給 AI 讀的 JSON
autoforge schema               # 列出所有資料結構
autoforge schema TrainingPlan --json
```

## 測試

```bash
pytest
```
