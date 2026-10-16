# image-segmentation — knowledge for AI planners

## When to use
- Label **every pixel** with a class (semantic segmentation): defect areas, crop vs weed, road/sky/building,
  water/land in aerial images, tissue regions. You need the area or outline of regions.
- You have masks for ≥ 50 images (≥ 300 for strong results); thin structures need more.

## When NOT to use
- Boxes or counts are enough → `object-detection` (cheaper labels).
- One label per image → `image-classification`.
- Touching objects of the same class must be separated (instance segmentation) — not supported.

## Data format
```
data/images/a.jpg  b.png ...        RGB images, any common format
data/masks/a.png   b.png ...        same file stem, single-channel PNG (mode L or P)
data/classes.txt                    optional: background\nscratch\nrust  (line i = class i)
```
Mask pixel value = class index, 0 = background; `ignore_index` (default 255) marks unlabelled pixels.
Without `classes.txt` classes are `class_0..class_<max value>`. RGB colour masks are rejected: convert colours
to indices first. Optional `data/train/{images,masks}` + `data/val/{images,masks}`; otherwise split by
`val_fraction`. Images and masks are resized to `image_size`×`image_size` (masks nearest). `task.target` unused.

## Recipe
`torchvision_segmenter`: AdamW, cross-entropy, random horizontal flips (`hflip`); `pretrained: true` loads
COCO (VOC classes) weights and replaces the final classifier conv for your classes.
| `model_name` | When |
|---|---|
| `lraspp_mobilenet_v3_large` (default) | CPU, fast serving, coarse regions |
| `deeplabv3_mobilenet_v3_large` | better boundaries, still light |
| `deeplabv3_resnet50` | GPU ≥ 8 GB, best accuracy |

## Metrics
- `miou`: mean IoU over classes present in val masks or predictions — **maximise for HPO**.
- `pixel_accuracy` (inflated by large background), `iou_<class>` per epoch, `train_loss`, `val_loss`.
  The epoch with the best `miou` is kept; per-class IoU of that epoch is in `model.json`.

## Tuning guide
- `learning_rate` 1e-4–3e-3 (AdamW); 5e-4 for pretrained, 1e-3–3e-3 from scratch.
- `epochs` 15–50; small datasets need more epochs.
- `image_size` 256–512: larger keeps thin/small regions, costs memory quadratically.
- Rare classes with IoU ≈ 0: more examples of them, larger `image_size`, more epochs.
- CPU: lraspp, `image_size: 256`, `batch_size: 4–8`.

## Common failures
| Symptom | Cause | Fix |
|---|---|---|
| `no mask for image` | stems differ (a.jpg vs a_mask.png) | rename masks to the image stem |
| `contains value 38 but there are only 3 classes` | mask saved with colours / anti-aliasing, or classes.txt too short | save index PNGs without smoothing; extend classes.txt; or use ignore_index |
| `masks must be single-channel` | RGB mask | map colours to class indices |
| miou high only for background | foreground tiny / rare | larger `image_size`, more epochs, more data |
| CUDA out of memory | batch/image too large | `batch_size: 4`, lower `image_size` |
| download error for weights | no internet to download.pytorch.org | pre-download, or `pretrained: false` |

## Serving
Input row: `{"image": "<path on the server>" | "<base64 or data URL>"}`. Output:
`{"prediction": "data:image/png;base64,...", "class_fractions": {"background": 0.81, "rust": 0.19}}` —
the PNG is a single-channel class-index mask at the ORIGINAL image size.
`model_export(format="onnx")`: input `image` float [N,3,S,S] RGB in 0..1 (normalisation inside the graph),
output `logits` [N,classes,S,S], S = `image_size`; needs `pip install onnx onnxscript`.
