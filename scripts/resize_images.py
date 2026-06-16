"""Resize a list of images in-place using Pillow.

Emits newline-delimited JSON events compatible with the streaming UI:
    {"type": "progress", "current": N, "total": M, "image": "..."}
    {"type": "result", "image": "...", "newPath": "...", "width": W, "height": H}
    {"type": "done", "processed": N, "total": M}
    {"type": "error", "message": "..."}
"""

import argparse
import json
import os
import sys
import threading
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

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


def output_path_for(image_path: Path, fmt: str) -> Path:
    suffix_by_fmt = {"keep": image_path.suffix, "jpg": ".jpg", "png": ".png", "webp": ".webp"}
    return image_path.with_suffix(suffix_by_fmt.get(fmt, image_path.suffix))


def compute_target_size(image: Image.Image, mode: str, width: int, height: int, fit_mode: str) -> tuple[int, int]:
    w, h = image.size
    if mode == "longest_side":
        target = width or height
        if w >= h:
            new_w = target
            new_h = max(1, round(h * target / w))
        else:
            new_h = target
            new_w = max(1, round(w * target / h))
        return new_w, new_h
    # exact mode
    if fit_mode == "fit":
        # preserve aspect, fit within w x h
        ratio = min(width / w, height / h)
        return max(1, round(w * ratio)), max(1, round(h * ratio))
    return width, height  # cover/exact: we'll handle padding/cropping below


def resize_image(
    image: Image.Image,
    mode: str,
    width: int,
    height: int,
    fit_mode: str,
    background_color: tuple[int, int, int],
) -> Image.Image:
    if mode == "longest_side":
        new_w, new_h = compute_target_size(image, mode, width, height, fit_mode)
        return image.resize((new_w, new_h), Image.LANCZOS)
    # exact
    if fit_mode == "fit":
        new_w, new_h = compute_target_size(image, mode, width, height, fit_mode)
        return image.resize((new_w, new_h), Image.LANCZOS)
    if fit_mode == "cover":
        # scale to cover, then center-crop
        ratio = max(width / image.width, height / image.height)
        scaled = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), Image.LANCZOS)
        left = (scaled.width - width) // 2
        top = (scaled.height - height) // 2
        return scaled.crop((left, top, left + width, top + height))
    if fit_mode == "pad":
        # scale to fit, then pad to exact w x h
        ratio = min(width / image.width, height / image.height)
        scaled = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), Image.LANCZOS
        )
        canvas = Image.new("RGB", (width, height), background_color)
        canvas.paste(scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2))
        return canvas
    # exact stretch
    return image.resize((width, height), Image.LANCZOS)


def save_image(image: Image.Image, dest: Path, fmt: str, quality: int) -> None:
    if fmt == "jpg":
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(dest, "JPEG", quality=quality, optimize=True)
    elif fmt == "png":
        image.save(dest, "PNG", optimize=True)
    elif fmt == "webp":
        image.save(dest, "WEBP", quality=quality, method=6)
    else:
        # keep: derive from extension
        ext = dest.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(dest, "JPEG", quality=quality, optimize=True)
        elif ext == ".png":
            image.save(dest, "PNG", optimize=True)
        elif ext == ".webp":
            image.save(dest, "WEBP", quality=quality, method=6)
        else:
            image.save(dest)


def parse_color(text: str) -> tuple[int, int, int]:
    t = text.strip().lower()
    if t in {"white", "#ffffff"}:
        return (255, 255, 255)
    if t in {"black", "#000000"}:
        return (0, 0, 0)
    if t.startswith("#") and len(t) == 7:
        return (int(t[1:3], 16), int(t[3:5], 16), int(t[5:7], 16))
    return (0, 0, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize a list of images in-place.")
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--mode", choices=["longest_side", "exact"], default="longest_side")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--fit", choices=["fit", "cover", "pad", "stretch"], default="fit")
    parser.add_argument("--pad-color", default="black")
    parser.add_argument("--format", choices=["keep", "jpg", "png", "webp"], default="keep")
    parser.add_argument("--quality", type=int, default=92)
    args = parser.parse_args()

    images = load_image_list(Path(args.image_list))
    if not images:
        emit({"type": "done", "processed": 0, "total": 0})
        return

    pad_color = parse_color(args.pad_color)
    total = len(images)
    processed = 0

    for index, image_path in enumerate(images):
        emit({"type": "progress", "current": index + 1, "total": total, "image": str(image_path)})
        try:
            with Image.open(image_path) as opened:
                src = opened.convert("RGBA") if args.format == "png" else opened.convert("RGB")
                resized = resize_image(src, args.mode, args.width, args.height, args.fit, pad_color)

            dest = output_path_for(image_path, args.format)
            save_image(resized, dest, args.format, args.quality)

            # If the format changed, drop the original (and its caption stays
            # attached to whatever stem we wrote to since stems match).
            if dest != image_path and image_path.exists():
                try:
                    os.remove(image_path)
                except OSError:
                    pass

            processed += 1
            emit({
                "type": "result",
                "image": str(image_path),
                "newPath": str(dest),
                "width": resized.width,
                "height": resized.height,
            })
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
