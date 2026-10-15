# image-classification — knowledge for AI planners

## When to use
- One label per image: product category, defect / no defect, plant disease, document type, scene.
- Data layout (folder per class):
  ```
  data/
    ok/      img001.jpg ...
    defect/  img101.jpg ...
  ```
  or `data/train/<class>/...` + `data/val/<class>/...` to fix the validation split yourself.

## When NOT to use
- Need where the object is (boxes) → object detection; pixel masks → segmentation (planned packs).
- Several labels per image (multi-label) — not supported yet.

## Choosing a model (`model_name`, all pretrained on ImageNet)
| Situation | Model |
|---|---|
| CPU only, or quick baseline | `resnet18` (default) |
| GPU, best accuracy/speed balance | `efficientnet_b0`, `convnext_tiny` |
| GPU ≥ 8 GB, large dataset | `resnet50`, `convnext_tiny`, `vit_base_patch16_224` |
Any timm model name works; keep `pretrained: true` unless images look nothing like photos.

## Data requirements
- ≥ 50 images per class for decent results, ≥ 300 for strong ones. Balance classes or expect bias.
- Images of any size/format (jpg/png/webp…); they are resized to `image_size`.
- Make sure near-duplicate photos are not split across train and val, or accuracy is inflated.

## Metrics
- `accuracy` on the validation split (best epoch kept), plus `train_loss`, `val_loss`, `val_accuracy` per epoch.

## Tuning guide
- `learning_rate` 1e-4–1e-3 (AdamW). `freeze_backbone_epochs: 1–2` trains only the new head first — stabler on small data.
- Small dataset (< 200 per class): more epochs (10–15), smaller model, keep pretrained.
- `val_loss` rising while `train_loss` falls → overfitting: fewer epochs, more data, smaller model.
- CPU: `resnet18`, `image_size: 160`, `batch_size: 16` keeps runs to minutes.

## Common failures
| Symptom | Cause | Fix |
|---|---|---|
| `no images found in class folders` | images not inside class sub-folders | move them into `data/<class>/` |
| CUDA out of memory | batch/model too large | `batch_size: 16` or 8, smaller model, lower `image_size` |
| accuracy = share of largest class | model predicts one class | check labels/folders, raise epochs, balance classes |
| download error for weights | no internet to huggingface.co | pre-download, or `pretrained: false` (much worse) |

## Serving
- Predict with `{"image": "<path on the server>"}` or a base64 / data-URL string. `model_export(format="onnx")` gives an ONNX graph (input `image` [N,3,H,W]); needs `pip install onnx onnxscript`.
