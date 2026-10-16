# diffusion-lora — knowledge for AI planners

## When to use
- Teach Stable Diffusion 1.x/2.x a **subject** (a person, pet, product, character) from 5–50 photos, DreamBooth
  style: set `instance_prompt` to "a photo of sks <class>" with a rare token (`sks`) and use that phrase at inference.
- Teach a **style or domain look** (illustration style, product shots, textures) from 100+ captioned images.
- Result is a small LoRA file (a few MB) on top of the base model; base weights are not changed.

## When NOT to use
- Classify or detect things in images → `image-classification`.
- Need SDXL/Flux-level quality, legible text in images, or exact layouts → this pack only supports SD 1.x/2.x.
- You do not have the rights to the images, or the subject is a real person who has not consented.

## Data format
A folder of images (.jpg .jpeg .png .webp .bmp; any size, they are resized and cropped to `resolution`). Captions,
in priority order: `metadata.csv` (columns `file_name,text`, paths relative to the folder) → a same-stem `.txt`
file (`dog1.jpg` + `dog1.txt`) → `instance_prompt` for every image without a caption.
```
data/metadata.csv:   file_name,text
                     dog1.jpg,a photo of sks dog on the beach
                     dog2.jpg,a photo of sks dog sleeping
```

## Recipe
`sd_lora` — LoRA (rank `lora_r`) on the UNet attention projections `to_q,to_k,to_v,to_out.0`; VAE and text
encoder frozen; noise-prediction MSE (epsilon or v_prediction from the model's scheduler config) with DDPM noise.
On CUDA the frozen weights run in bf16/fp16 (`mixed_precision: auto|bf16|fp16|no`), LoRA weights in fp32.
SD 1.5 at 512 px, `batch_size: 1`, `grad_accum: 4` fits 8 GB; add `gradient_checkpointing: true` for 6 GB or 768 px
(SD 2.x). `max_steps` (> 0) caps optimizer steps across epochs. Default base:
`stable-diffusion-v1-5/stable-diffusion-v1-5` (any diffusers-format SD 1.x/2.x repo or local folder works).

## Metrics
- `val_loss` (**minimize for HPO**): denoising MSE on held-out images (`val_fraction`) with fixed noise and fixed
  timesteps, so epochs are comparable. With fewer than 5 images the training images are reused (logged).
  The epoch with the lowest `val_loss` is kept.
- `train_loss` per optimizer step (noisy by nature — random timesteps).
- Diffusion loss only weakly tracks visual quality: always look at `output/samples/` (set `validation_prompt`).

## Tuning guide
- `learning_rate` 5e-5–3e-4 (default 1e-4). Subject looks "burnt"/oversaturated or ignores prompts → lower it.
- Subject: 5–20 images, ~100–200 optimizer steps per image is plenty (e.g. 10 images → `max_steps: 400–800`
  with `grad_accum: 1`). Style: 1–3 passes over hundreds of images.
- `lora_r` 4–8 for one subject, 16–32 for broad styles; keep `lora_alpha` = `lora_r`.
- `resolution` = the base model's native size (512 for SD 1.x / 2.x-base, 768 for SD 2.x 768-v).
- `random_flip: false` when orientation matters (text, logos, asymmetric faces).

## Common failures
| Symptom | Cause | Fix |
|---|---|---|
| CUDA out of memory | resolution/batch too big | `batch_size: 1`, `gradient_checkpointing: true`, lower `resolution` |
| `no caption` error | images without metadata.csv/.txt | add captions or set `instance_prompt` |
| `needs columns file_name,text` | wrong metadata.csv header | rename columns |
| every output looks like the training photos | overfitting | fewer steps, lower lr, more varied images |
| subject does not appear | trigger phrase missing or too few steps | use the exact `instance_prompt` words, train longer |
| fp16 NaN loss | fp16 overflow on older GPUs | `mixed_precision: no` or a bf16-capable GPU |

## Serving
Row `{"prompt": "a photo of sks dog in snow", "negative_prompt": "blurry" (optional), "num_inference_steps": 30
(optional), "seed": 42 (optional)}` → `{"prediction": "data:image/png;base64,...", "seed": 42}` at `resolution` px.
The model folder holds `pytorch_lora_weights.safetensors` (diffusers LoRA format, loadable elsewhere with
`StableDiffusionPipeline.load_lora_weights`) and `model.json` (base model name). The base model's safety checker is kept if it has one.
