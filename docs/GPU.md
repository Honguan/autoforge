# 在 NVIDIA GPU 上驗證方法包

GPU 方法包（Transformer 文字分類、LLM 微調與 DPO、圖片分類、物件偵測、影像分割、文字向量、Whisper、Diffusion LoRA）在 CI 裡只用極小的模型在 CPU 上跑過。`autoforge eval gpu` 在你的 GPU 上真的訓練一次，確認 CUDA 這條路沒問題。

```bash
pip install -U "autoforge-trainer[all]" bitsandbytes pyarrow   # bitsandbytes：QLoRA 4-bit；pyarrow：Whisper 用真的語音測
autoforge doctor                                         # 先確認偵測得到 GPU 和 CUDA
autoforge eval gpu                                       # quick：小型真模型，每項幾分鐘
autoforge eval gpu --full                                # full：方法包預設的正式大小
autoforge eval gpu -c llm-qlora -c diffusion-lora        # 只跑其中幾項
autoforge eval gpu --list                                # 看有哪些檢查
```

## 每一項檢查做什麼

1. 從索引安裝方法包（開發時可以用 `--packs-dir packs` 改用本地的版本）。
2. 產生一小份「正確的程式很容易學會」的資料，例如三種顏色的圖片、方塊和圓盤的偵測框、簡單的算術對話。
3. 用 `device: cuda` 走正常的訓練流程：計畫檢查、背景行程、checkpoint 都和平常一樣。
4. 訓練時每 0.25 秒用 NVML 記錄 GPU 記憶體，得到這次訓練的**峰值 VRAM**。
5. 檢查指標合不合理（例如圖片分類準確率 ≥ 0.9、分割 mIoU ≥ 0.5、偵測 mAP@0.5 ≥ 0.3），再註冊成模型並預測一次。

| 檢查 | 方法包 / 方法 | quick | full |
|---|---|---|---|
| image-classification | image-classification / timm_finetune | ResNet18，128 px | ConvNeXt-Tiny，224 px |
| text-transformer | text-classification / transformer_finetune | DistilBERT 多語 | 同左，3 epochs |
| llm-qlora | llm-finetune / sft_lora | Qwen2.5-0.5B，4-bit QLoRA | Qwen2.5-1.5B，4-bit QLoRA |
| llm-dpo | llm-preference / dpo_lora | Qwen2.5-0.5B，4-bit | Qwen2.5-1.5B，4-bit |
| object-detection | object-detection / torchvision_detector | Faster R-CNN MobileNetV3 | Faster R-CNN ResNet50 v2 |
| image-segmentation | image-segmentation / torchvision_segmenter | LR-ASPP MobileNetV3 | DeepLabV3 ResNet50 |
| embedding | embedding-finetune / contrastive_pairs | all-MiniLM-L6-v2 | 同左，3 epochs |
| whisper | speech-recognition / whisper_finetune | whisper-tiny | whisper-small |
| diffusion-lora | diffusion-lora / sd_lora | 極小的 SD 管線（測混合精度） | SD 1.5，512 px，100 步 |
| rl-ppo | reinforcement-learning / sb3 | CartPole PPO | 同左，10 萬步 |

第一次執行會下載預訓練模型：quick 大約 5 GB，full 大約 15 GB。Whisper 用 transformers 測試用的 LibriSpeech 片段（真的英文語音，約 9 MB，需要 `pyarrow`），要求微調後 WER ≤ 0.5；沒裝 `pyarrow` 時改用合成音調，只驗證 GPU 訓練流程。

## 看報告

報告在 `~/.autoforge/evals/gpu-<時間>/report.md`：

| 欄位 | 意思 |
|---|---|
| result | ✅ 通過；❌ 訓練失敗、指標不合理，或沒有用到 GPU；⚠️ 程式錯誤；⏭️ 少了套件（例如 bitsandbytes） |
| peak VRAM | 訓練時 GPU 記憶體的峰值（整張卡的增量） |
| declared min | 方法包宣告的最低 VRAM。峰值如果比宣告還高很多，宣告就要修正 |

失敗的項目會附上訓練日誌的最後 30 行（`report.json` 的 `log_tail`）。把 `report.md` 和 `report.json` 給維護者，就能修正方法包或調整 VRAM 宣告。

## 實測結果

**RTX 4060 Laptop（8 GB）、Windows 11、PyTorch 2.11 + CUDA 12.8，quick 等級（1.4.0）：** 10 項中 9 項通過。

| 檢查 | 時間 | VRAM 峰值 | 結果 |
|---|---|---|---|
| image-classification | 25 s | 未量到（單獨跑時 1.48 GB） | 準確率 1.0。當時誤判為失敗，1.4.1 已修正 |
| text-transformer | 60 s | 2.75 GB | 準確率 1.0 |
| llm-qlora（Qwen2.5-0.5B 4-bit） | 58 s | 1.44 GB | 正常，預測「3 plus 4 is 7.」 |
| llm-dpo（Qwen2.5-0.5B 4-bit） | 34 s | 1.64 GB | 偏好判斷正確率 1.0 |
| object-detection | 21 s | 0.51 GB | mAP@0.5 1.0 |
| image-segmentation | 12 s | 0.42 GB | mIoU 0.94 |
| embedding | 21 s | 0.54 GB | 資料設計有問題，1.4.1 已修正 |
| whisper | 34 s | 1.79 GB | 訓練流程正常；合成音調測不出辨識品質，1.4.1 改用真的語音 |
| diffusion-lora（極小管線，bf16） | 61 s | 0.19 GB | 正常 |
| rl-ppo | 99 s | 0.41 GB | 平均獎勵 492 |

這次實測後的修正（1.4.1）：
- VRAM 改成「峰值減最低讀數」，量不到時只提出警告。在 Windows 上，NVML 偶爾會漏量很短的訓練，而訓練實際有沒有在 GPU 上跑，看引擎記錄的 device 才準。
- LLM 微調和 DPO 的最低 VRAM 宣告從 8 GB 改成 6 GB：0.5B 的 QLoRA 實際只用了 1.4–1.6 GB，而方法包說明本來就寫 6–8 GB 的卡可以訓練 0.5–1.5B 的模型。
- 拿掉訓練日誌裡兩個 PyTorch 警告：`train_loss` 改用 `loss.item()`；gradient checkpointing 改成 `use_reentrant=False`。

