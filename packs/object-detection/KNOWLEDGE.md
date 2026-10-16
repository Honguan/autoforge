# object-detection — knowledge for AI planners

## When to use
- Find **where** objects are and draw boxes: count products on a shelf, locate defects, vehicles, people,
  animals, logos. Several objects and classes per image are fine.
- Labelled boxes exist (≥ 100 boxes per class for decent results, ≥ 500 for strong ones).

## When NOT to use
- One label per whole image is enough → `image-classification` (far less labelling).
- Exact outlines / areas of regions are needed → `image-segmentation`.
- Very small objects (< 10 px after resizing) → raise `min_size`, or crop/tile images upstream.

## Data format
`data_path` is a folder with the images and ONE annotation file (image paths relative to the folder):
```
data/annotations.csv              image,xmin,ymin,xmax,ymax,label
                                  img1.jpg,34,50,120,140,scratch
                                  img1.jpg,200,10,260,70,dent
                                  img2.jpg,,,,,              <- image with no objects
```
Coordinates are **pixels** of the original image (not 0–1). Or `data/annotations.json` in COCO format
(`images[id,file_name]`, `annotations[image_id,bbox=[x,y,w,h],category_id]`, `categories[id,name]`;
`iscrowd` boxes skipped). Optional `data/train/` + `data/val/`, each with its own annotation file; otherwise
images are split randomly by `val_fraction`. `task.target` is not used.

## Recipe
`torchvision_detector`: Faster R-CNN with FPN, COCO-pretrained (`pretrained: true`), box head replaced for
your classes. SGD momentum 0.9, linear warm-up over the first epoch, random horizontal flips (`hflip`).
| `model_name` | When |
|---|---|
| `fasterrcnn_mobilenet_v3_large_fpn` (default) | CPU, quick baseline, real-time-ish serving |
| `fasterrcnn_resnet50_fpn_v2` | GPU ≥ 8 GB, best accuracy, small objects |

## Metrics
- `map50`: mean over classes of VOC average precision at IoU 0.5 on the val split — **maximise for HPO**.
- `precision` / `recall`: boxes with score ≥ `score_threshold` matched at IoU 0.5. `train_loss` = sum of
  the four Faster R-CNN losses (per step). The epoch with the best `map50` is kept.

## Tuning guide
- `learning_rate` 0.002–0.02 (SGD); scale with `batch_size` (0.005 for 4 images, 0.01 for 8).
- `epochs` 10–30 for small datasets; stop earlier if `map50` stops rising.
- `min_size`/`max_size`: shorter/longer image side after resizing. 512/853 default; 800/1333 for small
  objects on GPU; 320/533 for fast CPU runs.
- `score_threshold` only affects precision/recall and serving (0.3–0.7), not training.
- CPU: mobilenet model, `min_size: 384`, `batch_size: 2–4`.

## Common failures
| Symptom | Cause | Fix |
|---|---|---|
| `need xmax > xmin and ymax > ymin` | swapped or (x, y, w, h) values in the CSV | convert to corner pixels |
| `training loss became nan` | learning rate too high | halve `learning_rate` |
| map50 ≈ 0 after many epochs | boxes normalised 0–1 instead of pixels | multiply by image width/height |
| `image listed in annotations not found` | paths not relative to the data folder | fix the `image` column / `file_name` |
| CUDA out of memory | images/batch too large | `batch_size: 2`, lower `min_size`/`max_size` |
| download error for weights | no internet to download.pytorch.org | pre-download, or `pretrained: false` (much worse) |

## Serving
Input row: `{"image": "<path on the server>" | "<base64 or data URL>", "score_threshold": 0.5 (optional)}`.
Output: `{"prediction": [{"label": "dent", "score": 0.93, "box": [xmin, ymin, xmax, ymax]}, ...], "count": n}`
with boxes in pixels of the input image. No exporter (ONNX of Faster R-CNN is not supported here).
