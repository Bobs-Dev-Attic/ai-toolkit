"""Upscale a list of images using Real-ESRGAN via spandrel.

Emits newline-delimited JSON events compatible with the streaming UI.
"""

import argparse
import json
import os
import sys
import threading
from pathlib import Path

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import torch
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Repo, filename, approximate size (bytes) for the progress UI.
MODEL_REGISTRY = {
    "x2": ("ai-forever/Real-ESRGAN", "RealESRGAN_x2.pth", 67_000_000),
    "x4": ("ai-forever/Real-ESRGAN", "RealESRGAN_x4.pth", 67_000_000),
    "x4plus": ("lllyasviel/Annotators", "RealESRGAN_x4plus.pth", 67_000_000),
}

_emit_lock = threading.Lock()


def emit(event: dict) -> None:
    with _emit_lock:
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()


def load_image_list(list_path: Path) -> list[Path]:
    images: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        candidate = Path(line)
        if candidate.suffix.lower() in IMAGE_EXTENSIONS and candidate.is_file():
            images.append(candidate)
    return images


def _directory_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _hf_download_with_retry(repo_id: str, filename: str, attempts: int = 3) -> str:
    """Wrap hf_hub_download with retry + longer etag timeout.

    On networks with SSL-intercepting antivirus (e.g. Avast), the HEAD call HF
    makes for ETag resolution can exceed the default 10s. We bump it to 30s
    and retry to ride out transient hiccups.
    """
    import time

    from huggingface_hub import hf_hub_download

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return hf_hub_download(repo_id, filename, etag_timeout=30)
        except Exception as exc:
            last_error = exc
            # Try to surface the underlying cause (e.g. SSLError) instead of
            # HF's wrapper message which omits the root reason.
            root = exc.__cause__ or exc.__context__ or exc
            sys.stderr.write(
                f"[upscale] hf_hub_download attempt {attempt + 1}/{attempts} failed: "
                f"{type(root).__name__}: {root}\n"
            )
            sys.stderr.flush()
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    root = last_error.__cause__ or last_error.__context__ or last_error
    raise RuntimeError(
        f"Could not download {filename} from {repo_id}: {type(root).__name__}: {root}"
    ) from last_error


def ensure_model_downloaded(repo_id: str, filename: str, approx_size: int) -> str:
    """Download the model weights if needed and return the local path."""
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.constants import HF_HUB_CACHE

    cached = try_to_load_from_cache(repo_id, filename)
    if isinstance(cached, str):
        return cached

    emit({"type": "model_download_start", "totalBytes": approx_size})

    cache_folder = os.path.join(HF_HUB_CACHE, "models--" + repo_id.replace("/", "--"))
    baseline = _directory_size(cache_folder)
    stop_event = threading.Event()
    result_holder: dict = {}

    def report_downloaded() -> int:
        downloaded = max(0, _directory_size(cache_folder) - baseline)
        return min(downloaded, approx_size) if approx_size else downloaded

    def poll_progress() -> None:
        while not stop_event.is_set():
            emit(
                {
                    "type": "model_download",
                    "downloadedBytes": report_downloaded(),
                    "totalBytes": approx_size,
                }
            )
            stop_event.wait(0.5)

    def worker() -> None:
        try:
            result_holder["path"] = _hf_download_with_retry(repo_id, filename)
        except Exception as exc:
            result_holder["error"] = exc

    poller = threading.Thread(target=poll_progress, daemon=True)
    poller.start()
    fetcher = threading.Thread(target=worker, daemon=True)
    fetcher.start()
    fetcher.join()
    stop_event.set()
    poller.join(timeout=2)

    if "error" in result_holder:
        raise result_holder["error"]

    emit(
        {
            "type": "model_download",
            "downloadedBytes": approx_size or report_downloaded(),
            "totalBytes": approx_size,
        }
    )
    return result_holder["path"]


def downscale_to_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    if w >= h:
        new_w = max_side
        new_h = max(1, round(h * max_side / w))
    else:
        new_h = max_side
        new_w = max(1, round(w * max_side / h))
    return image.resize((new_w, new_h), Image.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upscale images with Real-ESRGAN via spandrel.")
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="x4")
    parser.add_argument("--max-side", type=int, default=2048, help="Clamp the final longest side to this many pixels (0 disables).")
    parser.add_argument("--output-mode", choices=["replace", "sidecar"], default="sidecar")
    args = parser.parse_args()

    images = load_image_list(Path(args.image_list))
    if not images:
        emit({"type": "done", "processed": 0, "total": 0})
        return

    repo_id, filename, approx_size = MODEL_REGISTRY[args.model]
    weights_path = ensure_model_downloaded(repo_id, filename, approx_size)

    emit({"type": "model_loading"})

    import spandrel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_desc = spandrel.ModelLoader().load_from_file(weights_path)
    model = model_desc.model.eval().to(device)
    if device.type == "cuda":
        model = model.half()
    scale = model_desc.scale

    total = len(images)
    processed = 0

    for index, image_path in enumerate(images):
        emit({"type": "progress", "current": index + 1, "total": total, "image": str(image_path)})
        try:
            with Image.open(image_path) as opened:
                src = opened.convert("RGB")

            import numpy as np

            arr = np.array(src).astype("float32") / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            if device.type == "cuda":
                tensor = tensor.half()

            with torch.no_grad():
                out = model(tensor)
            out = out.clamp(0, 1).float().squeeze(0).permute(1, 2, 0).cpu().numpy()
            upscaled = Image.fromarray((out * 255.0).round().astype("uint8"))

            if args.max_side and args.max_side > 0:
                upscaled = downscale_to_max_side(upscaled, args.max_side)

            if args.output_mode == "sidecar":
                dest = image_path.with_name(f"{image_path.stem}.upscaled{scale}x.png")
            else:
                dest = image_path.with_suffix(".png")

            upscaled.save(dest, "PNG", optimize=True)

            if args.output_mode == "replace" and dest != image_path and image_path.exists():
                try:
                    os.remove(image_path)
                except OSError:
                    pass

            processed += 1
            emit(
                {
                    "type": "result",
                    "image": str(image_path),
                    "newPath": str(dest),
                    "width": upscaled.width,
                    "height": upscaled.height,
                }
            )
        except Exception as exc:
            emit({"type": "warning", "image": str(image_path), "message": str(exc)})

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    emit({"type": "done", "processed": processed, "total": total})


if __name__ == "__main__":
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    try:
        main()
    except Exception as exc:
        import traceback as _tb

        emit({"type": "error", "message": str(exc) or _tb.format_exc().strip().splitlines()[-1]})
        _tb.print_exc()
        sys.exit(1)
