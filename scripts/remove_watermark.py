"""Remove text, watermarks and logos from images (AI inpainting).

Two-stage, fully-local pipeline:
  1. Detection - Grounding DINO (native in transformers) locates the regions to
     erase from open-vocabulary text prompts. Each enabled target becomes a
     phrase ("text", "watermark", "logo") and any extra user phrases are added
     to the same query, so one pass finds everything.
  2. Inpainting - LaMa (simple-lama-inpainting) fills the masked regions so the
     background is reconstructed rather than blanked out.

Emits newline-delimited JSON events compatible with the streaming UI:
    {"type": "model_download_start", "totalBytes": N}
    {"type": "model_download", "downloadedBytes": N, "totalBytes": M}
    {"type": "model_loading"}
    {"type": "progress", "current": N, "total": M, "image": "..."}
    {"type": "result", "image": "...", "newPath": "..."}
    {"type": "warning", "image": "...", "message": "..."}
    {"type": "done", "processed": N, "total": M}
    {"type": "error", "message": "..."}
"""

import argparse
import fnmatch
import json
import os
import sys
import threading
from pathlib import Path

# Make Python trust the OS (Windows) certificate store so the HuggingFace hub
# (Grounding DINO) and GitHub (LaMa checkpoint) downloads go through cleanly.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import numpy as np
import cv2
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Grounding DINO detector choices exposed to the UI.
DETECTOR_MODELS = {
    "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
    "grounding-dino-base": "IDEA-Research/grounding-dino-base",
}

# Approx size of the LaMa torchscript checkpoint (big-lama.pt) for the bar.
LAMA_APPROX_BYTES = 205_000_000

# HF weight formats we never need to pull.
DOWNLOAD_IGNORE_PATTERNS = ["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.pth"]

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


# ---------------------------------------------------------------------------
# Model download helpers (progress via on-disk polling, backend-agnostic)
# ---------------------------------------------------------------------------


def ensure_hf_model_downloaded(model_id: str) -> None:
    from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
    from huggingface_hub.constants import HF_HUB_CACHE

    def is_cached(filename: str) -> bool:
        return isinstance(try_to_load_from_cache(model_id, filename), str)

    total_bytes = 0
    listing_succeeded = False
    allow_patterns: list[str] | None = None
    try:
        info = HfApi().model_info(model_id, files_metadata=True)
        weight_files = [s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors")]
        if not weight_files:
            weight_files = [s.rfilename for s in info.siblings if s.rfilename.endswith(".bin")]
        non_weight_files = [
            s.rfilename
            for s in info.siblings
            if not s.rfilename.endswith((".safetensors", ".bin", ".msgpack", ".h5", ".ot", ".onnx", ".pth"))
            and not any(fnmatch.fnmatch(s.rfilename, p) for p in DOWNLOAD_IGNORE_PATTERNS)
        ]
        allow_patterns = non_weight_files + weight_files
        size_by_name = {s.rfilename: s.size or 0 for s in info.siblings}
        for name in allow_patterns:
            if not is_cached(name):
                total_bytes += size_by_name.get(name, 0)
        listing_succeeded = True
    except Exception:
        total_bytes = 0

    if listing_succeeded and total_bytes == 0:
        return  # everything already cached

    emit({"type": "model_download_start", "totalBytes": total_bytes})

    cache_folder = os.path.join(HF_HUB_CACHE, "models--" + model_id.replace("/", "--"))
    baseline = _directory_size(cache_folder)
    stop_event = threading.Event()

    def report_downloaded() -> int:
        downloaded = max(0, _directory_size(cache_folder) - baseline)
        return min(downloaded, total_bytes) if total_bytes else downloaded

    def poll_progress() -> None:
        while not stop_event.is_set():
            emit({"type": "model_download", "downloadedBytes": report_downloaded(), "totalBytes": total_bytes})
            stop_event.wait(0.5)

    poller = threading.Thread(target=poll_progress, daemon=True)
    poller.start()
    try:
        if allow_patterns:
            snapshot_download(model_id, allow_patterns=allow_patterns)
        else:
            snapshot_download(model_id, ignore_patterns=DOWNLOAD_IGNORE_PATTERNS)
    finally:
        stop_event.set()
        poller.join(timeout=2)

    emit({"type": "model_download", "downloadedBytes": total_bytes or report_downloaded(), "totalBytes": total_bytes})


def lama_checkpoint_path() -> str:
    from torch.hub import get_dir

    return os.path.join(get_dir(), "checkpoints", "big-lama.pt")


def ensure_lama_downloaded() -> None:
    """Pre-warm the LaMa torchscript checkpoint with a progress bar."""
    ckpt = lama_checkpoint_path()
    if os.path.exists(ckpt) and os.path.getsize(ckpt) > 1_000_000:
        return

    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    total_bytes = LAMA_APPROX_BYTES
    emit({"type": "model_download_start", "totalBytes": total_bytes})

    stop_event = threading.Event()
    result_holder: dict = {}

    def report_downloaded() -> int:
        try:
            size = os.path.getsize(ckpt) if os.path.exists(ckpt) else 0
        except OSError:
            size = 0
        return min(size, total_bytes)

    def poll_progress() -> None:
        while not stop_event.is_set():
            emit({"type": "model_download", "downloadedBytes": report_downloaded(), "totalBytes": total_bytes})
            stop_event.wait(0.5)

    def worker() -> None:
        try:
            from simple_lama_inpainting.utils.util import download_model
            from simple_lama_inpainting.models.model import LAMA_MODEL_URL

            download_model(LAMA_MODEL_URL)
        except Exception as exc:  # noqa: BLE001
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

    emit({"type": "model_download", "downloadedBytes": total_bytes, "totalBytes": total_bytes})


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class Detector:
    def __init__(self, model_id: str, threshold: float):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.torch = torch
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Keep the detector in float32 - it is small, and mixing fp16 weights
        # with fp32 image inputs trips Grounding DINO's conv layers.
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            dtype=torch.float32,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def detect(self, image: Image.Image, phrases: list[str]) -> list[tuple[int, int, int, int]]:
        """Detect all phrases in one pass, returning [x1, y1, x2, y2] boxes.

        Grounding DINO expects a lowercase, period-separated caption.
        """
        torch = self.torch
        caption = ". ".join(p.strip().lower() for p in phrases if p.strip())
        if not caption:
            return []
        caption = caption + "."
        inputs = self.processor(images=image, text=caption, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.threshold,
            text_threshold=0.2,
            target_sizes=[(image.height, image.width)],
        )[0]

        boxes: list[tuple[int, int, int, int]] = []
        for box in results["boxes"].tolist():
            x1, y1, x2, y2 = box
            boxes.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
        return boxes


# ---------------------------------------------------------------------------
# Mask building + inpaint
# ---------------------------------------------------------------------------


def build_mask(
    detector: Detector,
    image: Image.Image,
    targets: set[str],
    extra_phrases: list[str],
    dilate_px: int,
) -> np.ndarray:
    h, w = image.height, image.width
    mask = np.zeros((h, w), dtype=np.uint8)

    phrases: list[str] = []
    if "text" in targets:
        phrases.append("text")
    if "watermark" in targets:
        phrases.append("watermark")
    if "logo" in targets:
        phrases.append("logo")
    phrases.extend(p for p in extra_phrases if p)

    for (x1, y1, x2, y2) in detector.detect(image, phrases):
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c > x1c and y2c > y1c:
            cv2.rectangle(mask, (x1c, y1c), (x2c, y2c), 255, thickness=-1)

    if dilate_px > 0 and mask.any():
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove text/watermarks/logos via detection + LaMa inpainting.")
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--model", default="grounding-dino-tiny", choices=list(DETECTOR_MODELS.keys()))
    parser.add_argument(
        "--targets",
        default="text,watermark,logo",
        help="Comma list of what to erase: text, watermark, logo.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Extra comma/period separated phrases to also detect and erase (e.g. 'timestamp, signature').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Detection confidence threshold (0-1). Lower catches more but risks false positives.",
    )
    parser.add_argument("--dilate", type=int, default=6, help="Pixels to grow each detected region before inpainting.")
    parser.add_argument(
        "--output-mode",
        default="sidecar",
        choices=["replace", "sidecar"],
        help="'replace' overwrites the original; 'sidecar' writes <name>.clean.png next to it.",
    )
    args = parser.parse_args()

    images = load_image_list(Path(args.image_list))
    if not images:
        emit({"type": "done", "processed": 0, "total": 0})
        return

    targets = {t.strip().lower() for t in args.targets.split(",") if t.strip()}
    targets &= {"text", "watermark", "logo"}
    if not targets:
        targets = {"text", "watermark", "logo"}

    extra_phrases = [p.strip() for chunk in args.prompt.split(",") for p in chunk.split(".") if p.strip()]

    model_id = DETECTOR_MODELS[args.model]
    threshold = min(max(args.threshold, 0.05), 0.95)

    # Stage 1 model: Grounding DINO detector.
    ensure_hf_model_downloaded(model_id)
    # Stage 2 model: LaMa inpainter.
    ensure_lama_downloaded()

    emit({"type": "model_loading"})

    detector = Detector(model_id, threshold)

    from simple_lama_inpainting import SimpleLama

    lama = SimpleLama()

    total = len(images)
    processed = 0

    for index, image_path in enumerate(images):
        emit({"type": "progress", "current": index + 1, "total": total, "image": str(image_path)})
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")

            mask = build_mask(detector, image, targets, extra_phrases, args.dilate)

            if not mask.any():
                # Nothing detected - skip so we don't rewrite an unchanged image.
                emit({"type": "warning", "image": str(image_path), "message": "No matching regions detected."})
                continue

            result = lama(image, Image.fromarray(mask))
            result = result.convert("RGB")

            if args.output_mode == "sidecar":
                dest = image_path.with_name(image_path.stem + ".clean.png")
                result.save(dest, "PNG", optimize=True)
            else:
                dest = image_path
                suffix = image_path.suffix.lower()
                if suffix in (".jpg", ".jpeg"):
                    result.save(dest, "JPEG", quality=95)
                elif suffix == ".webp":
                    result.save(dest, "WEBP", quality=95)
                else:
                    result.save(dest, "PNG", optimize=True)

            processed += 1
            emit({"type": "result", "image": str(image_path), "newPath": str(dest)})
        except Exception as exc:  # noqa: BLE001
            emit({"type": "warning", "image": str(image_path), "message": str(exc)})

    emit({"type": "done", "processed": processed, "total": total})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        import traceback as _tb

        emit({"type": "error", "message": str(exc) or _tb.format_exc().strip().splitlines()[-1]})
        _tb.print_exc()
        sys.exit(1)
