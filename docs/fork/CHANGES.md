# Fork Changes

A log of additions made on top of upstream `ostris/ai-toolkit`. Everything
here is contained within the **Datasets** page of the UI plus a handful of
new Python scripts; no training-loop code is modified.

Last upstream commit at branching: `853ffaf — Add light mode support.`

---

## TL;DR

The Datasets page has been turned into a **dataset-prep workbench**. From a
single page you can now:

- **Caption** images with BLIP-large (with Style preset, conditional prompt,
  repetition-penalty)
- **Auto-Crop** images around a detected face / upper-body (head + torso) /
  torso (no head, with **tight / medium / wide** tightness levels) /
  full person — square cropping that returns real image content, not
  padded bars
- **Resize** images (longest-side or exact W×H, with format conversion)
- **Remove backgrounds** (rembg with u2net / isnet / BiRefNet variants;
  transparent or flat-color output)
- **Upscale** images with Real-ESRGAN (×2 / ×4 / ×4plus via spandrel)
- **Bulk delete** selected images and their captions
- **Show Metadata** overlay (filename, dimensions, file size)
- **Refresh** the gallery and force-rebust image caches
- **Tune the grid** from 1 to 6 columns per row (persisted per browser)

On the **New Training Job** page, a **hardware-aware preset picker** appears
under the Model card. It reads the selected GPU's VRAM from the existing
`/api/gpu` endpoint and suggests Memory / Balanced / Quality presets per
model architecture, marking one "Recommended" based on what fits. Clicking
a preset fills in the relevant model + train + dataset fields as a starting
point.

All long-running ops stream progress events (model-download + per-image
progress) to a **floating progress modal** in the corner of the page.

---

## New Python scripts

All five emit the same newline-delimited JSON event protocol so the UI can
stream a uniform progress bar for any operation. They share the
`spawnImageOpStream` helper in `ui/src/server/imageOps.ts`.

| Script | What it does | Key dep |
|---|---|---|
| [`scripts/caption_dataset.py`](../../scripts/caption_dataset.py) | BLIP-large image captioning with Style/Prompt/RepetitionPenalty | `transformers` (BLIP) |
| [`scripts/resize_images.py`](../../scripts/resize_images.py) | Pillow resize with fit/cover/pad/stretch, format conversion | `Pillow` |
| [`scripts/remove_background.py`](../../scripts/remove_background.py) | rembg with selectable model + output background | `rembg`, `onnxruntime` |
| [`scripts/upscale_images.py`](../../scripts/upscale_images.py) | Real-ESRGAN via spandrel; auto-detects scale from .pth | `spandrel`, `torch` |
| [`scripts/auto_crop.py`](../../scripts/auto_crop.py) | Square-default face/body crop; insightface (face) + YOLOv8n (person) | `insightface`, `ultralytics` |

### Event protocol (stdout, one JSON object per line)

```
{"type": "model_download_start", "totalBytes": 1234}
{"type": "model_download", "downloadedBytes": N, "totalBytes": M}
{"type": "model_loading"}
{"type": "progress", "current": N, "total": M, "image": "/abs/path/img.jpg"}
{"type": "result", "image": "...", "newPath": "...", "width": W, "height": H}
{"type": "caption", "image": "...", "caption": "..."}    # caption only
{"type": "warning", "image": "...", "message": "..."}    # non-fatal per-image
{"type": "done", "processed": N, "total": M}
{"type": "error", "message": "..."}                      # fatal
```

### Model download progress strategy

All five scripts measure download progress by **polling the on-disk cache
directory size** from a background thread rather than hooking the
downloader's tqdm. Reason: `hf_transfer` (which is enabled in this toolkit)
uses a Rust HTTP client that bypasses tqdm entirely. Disk polling is
backend-agnostic and verified to produce smooth byte-level progress under
`hf_transfer`. The captioning script also disables `hf_transfer` because
its Rust HTTP client bypasses the `truststore` SSL patch (see SSL notes
below) and would fail under SSL-intercepting antivirus.

---

## New API routes (Next.js)

All five are streaming SSE endpoints built on the shared `spawnImageOpStream`
helper.

| Route | Method | Purpose |
|---|---|---|
| [`/api/datasets/caption`](../../ui/src/app/api/datasets/caption/route.ts) | POST | Caption (full-dataset or selection) |
| [`/api/datasets/resize`](../../ui/src/app/api/datasets/resize/route.ts) | POST | Resize selected images |
| [`/api/datasets/removeBackground`](../../ui/src/app/api/datasets/removeBackground/route.ts) | POST | Background removal |
| [`/api/datasets/upscale`](../../ui/src/app/api/datasets/upscale/route.ts) | POST | Upscale |
| [`/api/datasets/autoCrop`](../../ui/src/app/api/datasets/autoCrop/route.ts) | POST | Auto-crop |
| [`/api/img/bulkDelete`](../../ui/src/app/api/img/bulkDelete/route.ts) | POST | Delete N images + captions |

The streaming helper [`ui/src/server/imageOps.ts`](../../ui/src/server/imageOps.ts)
handles: temp image-list file lifecycle (and cleanup on abort), Python
process lifecycle, stderr forwarding to the dev console, error
propagation (script-emitted errors take precedence over generic
exit-code fallbacks), and client-disconnect → process-kill.

[`/api/datasets/listImages`](../../ui/src/app/api/datasets/listImages/route.ts)
was modified to also return `width`, `height`, and `size` for each image
(via the new `image-size` npm dep), powering the Show Metadata feature.

---

## UI changes

### Datasets page ([`ui/src/app/datasets/[datasetName]/page.tsx`](../../ui/src/app/datasets/[datasetName]/page.tsx))

The single most-modified file. Now includes:

- **Caption row** with trigger word input, overwrite checkbox, and
  expandable **Advanced** panel exposing Style (short/standard/detailed),
  conditional prompt, and repetition-penalty slider
- **Bulk action bar** below it: selection counter, Select-all / Clear,
  Delete, Resize, Remove BG, Upscale, Auto-Crop, columns slider (1–6),
  Show Metadata toggle, Refresh
- **Floating progress modal** in the bottom-right corner; uniform for all
  ops; shows download progress + per-image progress in one place; doesn't
  block the gallery
- Four modals (Resize, Remove BG, Upscale, Auto-Crop) for op options
- Dynamic grid columns from 1 to 6 (persisted to `localStorage` under
  `AI_TOOLKIT_DATASET_COLS`)
- Cache-bust on `/api/img/<path>` via `?v=${size}-${reloadSignal}` query
  param so refreshing after an op (or clicking Refresh) actually shows
  the new file bytes (the route serves with `Cache-Control: max-age=86400`)

### Card ([`ui/src/components/DatasetImageCard.tsx`](../../ui/src/components/DatasetImageCard.tsx))

- Per-card trash button **removed** in favor of always-selectable cards
- Optional metadata overlay (filename, dimensions, KB/MB)
- `reloadSignal` prop forces caption + image refetch after ops complete

---

## Dependencies added

### Python (installed in `python_embeded`)

```
truststore        # OS cert-store SSL trust (Avast workaround)
rembg             # background removal
onnxruntime       # rembg backend (CPU)
spandrel          # PyTorch super-res model loader
insightface       # RetinaFace face detector
ultralytics       # YOLOv8 person detector
```

NumPy is pinned to `<2` (some of these initially pulled 2.x; downgraded
back to 1.26 to avoid breaking the rest of the toolkit's training stack).

### Node (in `ui/`)

```
image-size        # quick image dimension reads from file headers
```

---

## Models the new features download

All cached under `~/.cache/huggingface/hub` unless noted.

| Feature | Model | Size | Source | Cache location |
|---|---|---|---|---|
| Captioning | `Salesforce/blip-image-captioning-large` | ~1.88 GB | HF Hub | HF cache |
| Remove BG (u2net) | u2net.onnx | 170 MB | rembg | `~/.u2net/` |
| Remove BG (u2netp) | u2netp.onnx | 5 MB | rembg | `~/.u2net/` |
| Remove BG (isnet) | isnet-general-use.onnx | 170 MB | rembg | `~/.u2net/` |
| Remove BG (BiRefNet) | birefnet-general | 885 MB | rembg | `~/.u2net/` |
| Upscale ×2 | `ai-forever/Real-ESRGAN` / `RealESRGAN_x2.pth` | 67 MB | HF Hub | HF cache |
| Upscale ×4 | same / `RealESRGAN_x4.pth` | 67 MB | HF Hub | HF cache |
| Upscale ×4plus | `lllyasviel/Annotators` / `RealESRGAN_x4plus.pth` | 67 MB | HF Hub | HF cache |
| Auto-Crop (face) | insightface `buffalo_l` pack | 282 MB | GitHub releases | `~/.insightface/models/buffalo_l/` |
| Auto-Crop (person) | YOLOv8n | 6 MB | ultralytics releases | `~/.ultralytics/` |

---

## Engineering notes worth preserving

### SSL with Avast HTTPS scanning

Avast (and similar AV products) intercept HTTPS, presenting their own
root cert. That root is in the Windows certificate store but not in
`certifi`'s bundle — so Python downloads fail with `SSL: CERTIFICATE_VERIFY_FAILED`.
The fix used here: call `truststore.inject_into_ssl()` at module load,
which makes Python verify against the OS cert store (which has the Avast
root). All scripts that hit the network do this.

The corollary: `hf_transfer` (Rust HTTP client with rustls) bypasses
`truststore`. The captioning script disables `hf_transfer`
(`os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"`) before any HF import.

### Florence-2 won't load on current `transformers`

Original captioning attempt used `multimodalart/Florence-2-large-no-flash-attn`.
That repo's custom modeling code predates transformers 4.50; loading it
hit a cascade of `_supports_sdpa`, `GenerationMixin`, `generation_config`,
and `prepare_inputs_for_generation` incompatibilities. **Switched to
BLIP-large** as a first-class-supported alternative. Decision recorded
in the captioning script comment.

### Real-ESRGAN HEAD-call timeouts

The first auto-crop / upscale download under Avast SSL interception can
exceed HF's default 10-second ETag-HEAD timeout. The upscale script wraps
`hf_hub_download` in a retry loop with `etag_timeout=30` and surfaces the
underlying SSL/network exception (instead of HF's generic "cannot find
file on Hub" wrapper) for diagnosability.

### Numpy 2.x ABI break

`rembg 2.0.76` requested NumPy 2.x at install time, which broke
`dctorch`. Pinned NumPy back to `<2`. Both rembg and onnxruntime run
fine on 1.26 in practice.

### Gallery image caching

`/api/img/<path>` sends `Cache-Control: public, max-age=86400` to keep
scroll-fetches fast. After ops rewrite an image, the URL is unchanged,
so the browser shows the stale cached version. Solution: card image
URLs include `?v=${size}-${reloadSignal}` — `size` (from listImages)
changes whenever the file bytes change, and `reloadSignal` is the manual
override bumped by the Refresh button and any op completion.

### Auto-crop is square by default

Faces / persons are taller than wide; the natural detected bbox would
produce rectangular crops. Instead, after the user's padding expansion,
`expand_to_aspect()` extends the bbox along its shorter axis (in source
pixel space) so the cropped region itself is square — real image
content, not padded bars. When near an image edge it shifts inward
rather than clipping, preserving the exact requested aspect ratio.
"Exact W×H" mode respects the user's explicit non-square aspect.

### Torso tightness levels

The `torso` target derives its bbox from face detection (same approach
as `upper_body`) but excludes the head. Three preset tightness levels
let the user choose how much of the body below the chin to include —
configured via `TORSO_LEVELS` in [auto_crop.py](../../scripts/auto_crop.py):

| Level | Horizontal pad (× face width) | Vertical extent below chin (× face height) | Typical framing |
|---|---|---|---|
| `tight` | 1.0× | 1.5 | Shoulders + upper chest |
| `medium` (default) | 1.3× | 2.5 | Chest down to mid-torso |
| `wide` | 1.5× | 3.5 | Down to hip area |

Adding additional levels is just another entry in the `TORSO_LEVELS`
dict; no other code changes needed.

### Hardware-aware preset matrix

Lives in [`ui/src/app/jobs/new/configPresets.ts`](../../ui/src/app/jobs/new/configPresets.ts).
Three tiers (`memory` / `balanced` / `quality`) for each of:

- `flux` — FLUX.1
- `qwen_image` — Qwen-Image (with torchao uint3 + ARA at the Balanced tier)
- `sdxl` — SDXL
- `hidream` — HiDream
- `wan21` — Wan 2.1
- `wan22_14b_i2v` — Wan 2.2 14B I2V (with torchao uint4 + ARA at the Balanced tier)
- `wan22_5b` — Wan 2.2 5B TI2V
- `ltx2` — LTX-2

Each preset is a flat `{ 'config.process[0].model.quantize': true, ... }`
override map, applied by iterating and calling the existing
`setJobConfig(value, path)` setter. This matches the pattern already used
by `modelArch.defaults` and avoids needing a separate deep-merge helper.

`recommendPreset(presets, detectedVramGB)` picks the highest preset whose
`approxVramGB` ≤ detected VRAM. The `approxVramGB` numbers are written
as "target GPU class" (e.g. the Qwen-Image Balanced preset = 24 because
it's designed for a 24 GB card), so a literal `≤` comparison is the
right rule. The "Recommended" badge is rendered by the
[`PresetPicker`](../../ui/src/app/jobs/new/PresetPicker.tsx) component.

VRAM estimates are explicitly approximate (called out in the UI). Adding
a new architecture is an isolated edit — add an entry to
`CONFIG_PRESETS` and it appears automatically when that arch is
selected. Archs without an entry simply don't show the picker.

### Auto-crop sidecars never overwrite

In sidecar mode the first run produces `<name>.crop.<ext>`. Re-running
auto-crop on the same source image (e.g. with different padding or a
different target) appends an auto-incremented number rather than
overwriting: `<name>.crop2.<ext>`, `<name>.crop3.<ext>`, etc. The
`next_available_sidecar()` helper picks the smallest free number on
disk. **Replace mode** (`--output-mode replace`) still overwrites the
original, by design — that's the explicit destructive opt-in.

---

## Files changed / added

### Added

```
scripts/auto_crop.py
scripts/caption_dataset.py
scripts/remove_background.py
scripts/resize_images.py
scripts/upscale_images.py
ui/src/server/imageOps.ts
ui/src/app/api/datasets/caption/route.ts        # NEW endpoint, replacing inline impl
ui/src/app/api/datasets/resize/route.ts
ui/src/app/api/datasets/removeBackground/route.ts
ui/src/app/api/datasets/upscale/route.ts
ui/src/app/api/datasets/autoCrop/route.ts
ui/src/app/api/img/bulkDelete/route.ts
ui/src/app/jobs/new/configPresets.ts            # hardware-aware preset matrix
ui/src/app/jobs/new/PresetPicker.tsx            # preset card UI
docs/fork/CHANGES.md                            # this file
docs/fork/TRAINING_OPTIMIZATIONS.md             # proposed work
docs/fork/PUBLISHING_TO_GITHUB.md               # publish guide
```

### Modified

```
ui/src/app/api/datasets/listImages/route.ts     # adds width/height/size to response
ui/src/app/datasets/[datasetName]/page.tsx      # bulk action bar, modals, progress UI
ui/src/components/DatasetImageCard.tsx          # selection, metadata overlay, cache-bust
ui/src/app/jobs/new/SimpleJob.tsx               # mounts PresetPicker + GPU detection
ui/package.json                                 # +image-size
ui/package-lock.json
```

No training-loop code (`jobs/`, `toolkit/network/`, `toolkit/optimizer.py`,
loss / scheduler code) is touched.
