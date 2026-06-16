"""Remove the background from a list of images using rembg.

Emits newline-delimited JSON events compatible with the streaming UI:
    {"type": "model_download_start", "totalBytes": N}
    {"type": "model_download", "downloadedBytes": N, "totalBytes": M}
    {"type": "model_loading"}
    {"type": "progress", "current": N, "total": M, "image": "..."}
    {"type": "result", "image": "...", "newPath": "..."}
    {"type": "done", "processed": N, "total": M}
    {"type": "error", "message": "..."}
"""

import argparse
import json
import os
import sys
import threading
from pathlib import Path

# Make Python trust the OS (Windows) certificate store - rembg downloads its
# ONNX models from huggingface.co / github via requests, which goes through
# Python's SSL stack.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Approximate model file sizes (bytes) used to render a determinate progress
# bar before the first chunk lands on disk. Pulled from each model's release.
APPROX_MODEL_SIZES = {
    "u2net": 176_000_000,
    "u2netp": 4_700_000,
    "isnet-general-use": 178_000_000,
    "birefnet-general": 885_000_000,
    "birefnet-general-lite": 220_000_000,
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


def rembg_cache_dir() -> str:
    # Mirrors rembg's own resolution order
    env_home = os.environ.get("U2NET_HOME")
    if env_home:
        return env_home
    return os.path.join(os.path.expanduser("~"), ".u2net")


def parse_bg(text: str) -> tuple[str, tuple[int, int, int, int] | None]:
    t = text.strip().lower()
    if t == "transparent":
        return "transparent", None
    if t in {"white", "#ffffff"}:
        return "color", (255, 255, 255, 255)
    if t == "black" or t == "#000000":
        return "color", (0, 0, 0, 255)
    if t.startswith("#") and len(t) == 7:
        return "color", (int(t[1:3], 16), int(t[3:5], 16), int(t[5:7], 16), 255)
    return "transparent", None


def ensure_model_downloaded(model_name: str) -> None:
    """Trigger rembg's model download with progress polling.

    rembg lazily downloads the ONNX file the first time new_session() is
    called. We pre-warm it here so the user sees a model_download phase
    in the UI before any per-image progress events appear.
    """
    from rembg import new_session

    cache = rembg_cache_dir()
    model_file_guess = os.path.join(cache, f"{model_name}.onnx")
    if os.path.exists(model_file_guess) and os.path.getsize(model_file_guess) > 1_000_000:
        return  # already on disk

    os.makedirs(cache, exist_ok=True)
    total_bytes = APPROX_MODEL_SIZES.get(model_name, 0)
    emit({"type": "model_download_start", "totalBytes": total_bytes})

    baseline = _directory_size(cache)
    stop_event = threading.Event()
    result_holder: dict = {}

    def report_downloaded() -> int:
        downloaded = max(0, _directory_size(cache) - baseline)
        return min(downloaded, total_bytes) if total_bytes else downloaded

    def poll_progress() -> None:
        while not stop_event.is_set():
            emit({"type": "model_download", "downloadedBytes": report_downloaded(), "totalBytes": total_bytes})
            stop_event.wait(0.5)

    def worker() -> None:
        try:
            result_holder["session"] = new_session(model_name)
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
            "downloadedBytes": total_bytes or report_downloaded(),
            "totalBytes": total_bytes,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove image backgrounds with rembg.")
    parser.add_argument("--image-list", required=True)
    parser.add_argument(
        "--model",
        default="u2net",
        choices=list(APPROX_MODEL_SIZES.keys()),
    )
    parser.add_argument(
        "--bg",
        default="transparent",
        help="Output background: 'transparent', 'white', 'black', or a #RRGGBB hex color.",
    )
    parser.add_argument(
        "--output-mode",
        default="replace",
        choices=["replace", "sidecar"],
        help="'replace' overwrites the original (forces PNG); 'sidecar' writes <name>.nobg.png next to it.",
    )
    args = parser.parse_args()

    images = load_image_list(Path(args.image_list))
    if not images:
        emit({"type": "done", "processed": 0, "total": 0})
        return

    bg_mode, bg_color = parse_bg(args.bg)

    ensure_model_downloaded(args.model)

    emit({"type": "model_loading"})

    from rembg import new_session, remove

    session = new_session(args.model)

    total = len(images)
    processed = 0

    for index, image_path in enumerate(images):
        emit({"type": "progress", "current": index + 1, "total": total, "image": str(image_path)})
        try:
            with Image.open(image_path) as opened:
                src = opened.convert("RGBA")
            cutout = remove(src, session=session)  # RGBA with alpha matte

            if bg_mode == "color" and bg_color is not None:
                flat = Image.new("RGBA", cutout.size, bg_color)
                flat.alpha_composite(cutout)
                final = flat.convert("RGB")
            else:
                final = cutout  # keep transparency

            if args.output_mode == "sidecar":
                dest = image_path.with_name(image_path.stem + ".nobg.png")
            else:
                dest = image_path.with_suffix(".png")

            final.save(dest, "PNG", optimize=True)

            # If we changed the extension on replace (e.g. .jpg -> .png), delete
            # the now-stale original so the dataset has exactly one file per item.
            if args.output_mode == "replace" and dest != image_path and image_path.exists():
                try:
                    os.remove(image_path)
                except OSError:
                    pass

            processed += 1
            emit({"type": "result", "image": str(image_path), "newPath": str(dest)})
        except Exception as exc:
            emit({"type": "warning", "image": str(image_path), "message": str(exc)})

    emit({"type": "done", "processed": processed, "total": total})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback as _tb

        emit({"type": "error", "message": str(exc) or _tb.format_exc().strip().splitlines()[-1]})
        _tb.print_exc()
        sys.exit(1)
