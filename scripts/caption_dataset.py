import argparse
import fnmatch
import json
import os
import sys
import threading
from pathlib import Path

# Make Python trust the OS (Windows) certificate store in addition to certifi's
# bundle. Needed when antivirus/proxy HTTPS scanning (e.g. Avast Web Shield)
# re-signs traffic with a root CA that only lives in the Windows store.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# BLIP-large is officially supported by current transformers, ~990MB, and
# produces solid single-sentence captions for dataset prep without needing
# trust_remote_code.
DEFAULT_MODEL = "Salesforce/blip-image-captioning-large"

# Repo files that from_pretrained never needs - skip them so the download
# progress total reflects only the weights/config/tokenizer that matter.
DOWNLOAD_IGNORE_PATTERNS = ["*.ipynb", "*.md", ".gitattributes", "LICENSE", "*.msgpack", "*.h5", "*.ot"]

# Serialize stdout writes so progress events emitted from download worker
# threads never interleave into a corrupt JSON line.
_emit_lock = threading.Lock()


def emit(event: dict) -> None:
    """Write a single newline-delimited JSON event to stdout."""
    with _emit_lock:
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()


def find_images(dataset_dir: Path) -> list[Path]:
    images: list[Path] = []
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith("."):
            images.append(path)
    return sorted(images)


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


def caption_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def build_caption(raw_caption: str, trigger_word: str) -> str:
    caption = raw_caption.strip()
    # BLIP often prefixes a generic framing phrase - trim the common ones.
    for prefix in ("a picture of ", "an image of ", "a photo of ", "the image shows "):
        if caption.lower().startswith(prefix):
            caption = caption[len(prefix):].strip()
            break
    if trigger_word and trigger_word not in caption:
        caption = f"{caption} {trigger_word}".strip()
    return caption


def _directory_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _select_weight_file(siblings) -> list[str]:
    """Pick exactly one weights file to download.

    Repos often publish the same weights in multiple formats (safetensors,
    pytorch, msgpack, etc.). Preferring safetensors keeps the download to the
    minimum required.
    """
    names = [s.rfilename for s in siblings]
    preferred = ["model.safetensors", "pytorch_model.bin"]
    for name in preferred:
        if name in names:
            return [name]
    # Fall back to any single-file weights match.
    for s in siblings:
        if s.rfilename.endswith((".safetensors", ".bin")):
            return [s.rfilename]
    return []


def ensure_model_downloaded(model_id: str) -> None:
    """Download the model if it isn't cached yet, emitting byte progress.

    Progress is measured by polling the on-disk cache size rather than hooking
    tqdm, because hf_transfer (when enabled) downloads outside of tqdm and
    never drives a progress-bar callback. Disk polling is backend-agnostic.
    """
    from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
    from huggingface_hub.constants import HF_HUB_CACHE

    def is_cached(filename: str) -> bool:
        return isinstance(try_to_load_from_cache(model_id, filename), str)

    # Figure out which files we actually need and how big they are. Skipping
    # duplicate weight formats (e.g. msgpack, h5) keeps the download tight.
    total_bytes = 0
    listing_succeeded = False
    allow_patterns: list[str] | None = None
    try:
        info = HfApi().model_info(model_id, files_metadata=True)
        weight_file = _select_weight_file(info.siblings)
        non_weight_files = [
            s.rfilename
            for s in info.siblings
            if not s.rfilename.endswith((".safetensors", ".bin", ".msgpack", ".h5", ".ot"))
            and not any(fnmatch.fnmatch(s.rfilename, p) for p in DOWNLOAD_IGNORE_PATTERNS)
        ]
        allow_patterns = non_weight_files + weight_file
        size_by_name = {s.rfilename: s.size or 0 for s in info.siblings}
        for name in allow_patterns:
            if not is_cached(name):
                total_bytes += size_by_name.get(name, 0)
        listing_succeeded = True
    except Exception:
        total_bytes = 0

    if listing_succeeded and total_bytes == 0:
        # All required files already on disk - no download needed.
        return

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

    emit(
        {
            "type": "model_download",
            "downloadedBytes": total_bytes or report_downloaded(),
            "totalBytes": total_bytes,
        }
    )


STYLE_PRESETS = {
    "short": {"max_new_tokens": 30, "num_beams": 3, "min_length": 0},
    "standard": {"max_new_tokens": 80, "num_beams": 5, "min_length": 0},
    "detailed": {"max_new_tokens": 140, "num_beams": 5, "min_length": 20},
}


def main():
    parser = argparse.ArgumentParser(description="Generate txt captions for an AI Toolkit dataset.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trigger-word", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--image-list",
        default="",
        help="Path to a UTF-8 file of newline-separated image paths to caption (replaces existing captions).",
    )
    parser.add_argument("--style", choices=list(STYLE_PRESETS.keys()), default="standard")
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional conditional prompt that BLIP continues from (e.g. 'a photograph of').",
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    if args.image_list:
        # Explicit selection: caption exactly these images, replacing any
        # existing caption regardless of the --overwrite flag.
        images = load_image_list(Path(args.image_list))
    else:
        images = find_images(dataset_dir)
        if not args.overwrite:
            images = [image_path for image_path in images if not caption_path_for(image_path).exists()]
    if args.limit > 0:
        images = images[: args.limit]

    if not images:
        emit({"type": "done", "processed": 0, "total": 0})
        return

    ensure_model_downloaded(args.model)

    emit({"type": "model_loading"})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = BlipForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        local_files_only=True,
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)

    style = STYLE_PRESETS[args.style]
    conditional_prompt = args.prompt.strip()

    total = len(images)
    processed = 0
    for index, image_path in enumerate(images):
        emit({"type": "progress", "current": index + 1, "total": total, "image": str(image_path)})

        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")

        # When a conditional prompt is supplied BLIP continues from it; the
        # prompt text leaks into the decoded output so we strip it back off.
        processor_kwargs = {"images": image, "return_tensors": "pt"}
        if conditional_prompt:
            processor_kwargs["text"] = conditional_prompt
        inputs = processor(**processor_kwargs).to(device, torch_dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=style["max_new_tokens"],
                num_beams=style["num_beams"],
                min_length=style["min_length"],
                repetition_penalty=args.repetition_penalty,
            )
        raw_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if conditional_prompt and raw_caption.lower().startswith(conditional_prompt.lower()):
            raw_caption = raw_caption[len(conditional_prompt):].lstrip()
        caption = build_caption(raw_caption, args.trigger_word)
        caption_path_for(image_path).write_text(caption, encoding="utf-8")
        processed += 1
        emit({"type": "caption", "image": str(image_path), "caption": caption})

    del model
    del processor
    if device == "cuda":
        torch.cuda.empty_cache()

    emit({"type": "done", "processed": processed, "total": total})


if __name__ == "__main__":
    # Force the pure-Python download path. hf_transfer uses a Rust HTTP client
    # (rustls) that bypasses the truststore SSL patch above, so corporate /
    # antivirus HTTPS interception (e.g. Avast Web Shield re-signing traffic)
    # fails the handshake on hosts that aren't covered by Mozilla's CA bundle.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    try:
        main()
    except Exception as exc:  # surface a structured error to the streaming UI
        import traceback as _tb

        emit({"type": "error", "message": str(exc) or _tb.format_exc().strip().splitlines()[-1]})
        _tb.print_exc()
        sys.exit(1)
