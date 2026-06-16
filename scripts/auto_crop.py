"""Auto-crop images around a detected target (face, person, upper body).

Face detection uses insightface (RetinaFace/SCRFD via buffalo_l, ~250 MB).
Person detection uses Ultralytics YOLOv8n (~6 MB).
Upper body is derived from face detection by extending the face bbox.

Emits newline-delimited JSON events compatible with the streaming UI.
"""

import argparse
import contextlib
import io
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

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Approx download sizes for the progress UI.
FACE_MODEL_PACK = "buffalo_l"
FACE_MODEL_APPROX_BYTES = 290_000_000  # buffalo_l.zip ≈ 280 MB
YOLO_MODEL_FILE = "yolov8n.pt"
YOLO_MODEL_APPROX_BYTES = 6_500_000

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


def insightface_home() -> str:
    return os.environ.get("INSIGHTFACE_HOME") or os.path.expanduser("~/.insightface")


def yolo_weights_dir() -> str:
    # ultralytics by default caches model weights in the current working dir
    # the first time you call YOLO('yolov8n.pt'). We pin it under the user's
    # cache so subsequent runs find it.
    home = os.path.expanduser("~/.ultralytics")
    os.makedirs(home, exist_ok=True)
    return home


def _poll_dir_progress(cache_folder: str, total_bytes: int, stop_event: threading.Event) -> None:
    baseline = _directory_size(cache_folder)
    while not stop_event.is_set():
        downloaded = max(0, _directory_size(cache_folder) - baseline)
        if total_bytes:
            downloaded = min(downloaded, total_bytes)
        emit({"type": "model_download", "downloadedBytes": downloaded, "totalBytes": total_bytes})
        stop_event.wait(0.5)


def ensure_face_model_downloaded() -> None:
    """Trigger insightface's model download with progress polling."""
    cache_root = os.path.join(insightface_home(), "models")
    extracted = os.path.join(cache_root, FACE_MODEL_PACK, "det_10g.onnx")
    if os.path.exists(extracted) and os.path.getsize(extracted) > 1_000_000:
        return

    os.makedirs(cache_root, exist_ok=True)
    emit({"type": "model_download_start", "totalBytes": FACE_MODEL_APPROX_BYTES})

    stop_event = threading.Event()
    result_holder: dict = {}

    def worker() -> None:
        try:
            # insightface prints progress + provider info to stdout, which would
            # pollute our JSON event stream. Capture and discard it.
            with contextlib.redirect_stdout(io.StringIO()):
                from insightface.app import FaceAnalysis

                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                app = FaceAnalysis(name=FACE_MODEL_PACK, allowed_modules=["detection"], providers=providers)
                app.prepare(ctx_id=0, det_size=(640, 640))
            result_holder["app"] = app
        except Exception as exc:
            result_holder["error"] = exc

    poller = threading.Thread(target=_poll_dir_progress, args=(cache_root, FACE_MODEL_APPROX_BYTES, stop_event), daemon=True)
    poller.start()
    fetcher = threading.Thread(target=worker, daemon=True)
    fetcher.start()
    fetcher.join()
    stop_event.set()
    poller.join(timeout=2)

    if "error" in result_holder:
        raise result_holder["error"]

    emit({"type": "model_download", "downloadedBytes": FACE_MODEL_APPROX_BYTES, "totalBytes": FACE_MODEL_APPROX_BYTES})


def ensure_yolo_model_downloaded() -> str:
    """Download yolov8n.pt to a stable cache location and return its path."""
    weights_dir = yolo_weights_dir()
    weights_path = os.path.join(weights_dir, YOLO_MODEL_FILE)
    if os.path.exists(weights_path) and os.path.getsize(weights_path) > 1_000_000:
        return weights_path

    emit({"type": "model_download_start", "totalBytes": YOLO_MODEL_APPROX_BYTES})

    stop_event = threading.Event()
    result_holder: dict = {}

    def worker() -> None:
        try:
            import urllib.request

            url = f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{YOLO_MODEL_FILE}"
            with urllib.request.urlopen(url, timeout=60) as response, open(weights_path, "wb") as out:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
        except Exception as exc:
            result_holder["error"] = exc

    poller = threading.Thread(target=_poll_dir_progress, args=(weights_dir, YOLO_MODEL_APPROX_BYTES, stop_event), daemon=True)
    poller.start()
    fetcher = threading.Thread(target=worker, daemon=True)
    fetcher.start()
    fetcher.join()
    stop_event.set()
    poller.join(timeout=2)

    if "error" in result_holder:
        raise result_holder["error"]

    emit({"type": "model_download", "downloadedBytes": YOLO_MODEL_APPROX_BYTES, "totalBytes": YOLO_MODEL_APPROX_BYTES})
    return weights_path


# --- detection ----------------------------------------------------------------


def detect_faces(face_app, image: Image.Image) -> list[tuple[float, float, float, float, float]]:
    """Return list of (x1, y1, x2, y2, score) face boxes."""
    import numpy as np

    bgr = np.array(image.convert("RGB"))[:, :, ::-1]  # PIL RGB -> cv2 BGR
    faces = face_app.get(bgr)
    out = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.tolist()
        out.append((float(x1), float(y1), float(x2), float(y2), float(getattr(f, "det_score", 0.0))))
    return out


def detect_people(yolo_model, image: Image.Image) -> list[tuple[float, float, float, float, float]]:
    """Return list of (x1, y1, x2, y2, score) person boxes."""
    import numpy as np

    rgb = np.array(image.convert("RGB"))
    # classes=[0] limits YOLO to the 'person' class
    results = yolo_model(rgb, classes=[0], verbose=False)
    out: list[tuple[float, float, float, float, float]] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0].item()) if box.conf is not None else 0.0
            out.append((float(x1), float(y1), float(x2), float(y2), score))
    return out


def derive_upper_body_box(
    face_box: tuple[float, float, float, float, float], image_w: int, image_h: int
) -> tuple[float, float, float, float, float]:
    """Extend a face bbox to roughly cover the head + shoulders + chest."""
    x1, y1, x2, y2, score = face_box
    fw = x2 - x1
    fh = y2 - y1
    nx1 = x1 - fw * 1.0
    ny1 = y1 - fh * 0.4
    nx2 = x2 + fw * 1.0
    ny2 = y2 + fh * 3.0
    nx1 = max(0.0, nx1)
    ny1 = max(0.0, ny1)
    nx2 = min(float(image_w), nx2)
    ny2 = min(float(image_h), ny2)
    return nx1, ny1, nx2, ny2, score


# Torso tightness presets - each entry is (x_pad_factor, top_offset, bottom_extent)
# expressed in units of face width / face height. All start just below the chin.
TORSO_LEVELS = {
    # Shoulders + upper chest only. Best for headshot-style portrait crops
    # where you want the subject's clothing context but nothing below.
    "tight":  {"xpad": 1.0, "top": -0.1, "bottom": 1.5},
    # Chest down to mid-torso. The most common framing.
    "medium": {"xpad": 1.3, "top": -0.1, "bottom": 2.5},
    # Full torso down to hip area. Roughly the original upper_body geometry
    # but without the head.
    "wide":   {"xpad": 1.5, "top": -0.1, "bottom": 3.5},
}


def derive_torso_box(
    face_box: tuple[float, float, float, float, float],
    image_w: int,
    image_h: int,
    tightness: str = "medium",
) -> tuple[float, float, float, float, float]:
    """Derive a torso bbox from a face bbox, excluding the head.

    `tightness` picks one of TORSO_LEVELS. All levels start just below the
    chin (small overlap to keep the neckline) and extend downward; the
    horizontal extent always centers on the face and grows with tightness.
    """
    x1, y1, x2, y2, score = face_box
    fw = x2 - x1
    fh = y2 - y1
    preset = TORSO_LEVELS.get(tightness, TORSO_LEVELS["medium"])
    nx1 = x1 - fw * preset["xpad"]
    ny1 = y2 + fh * preset["top"]
    nx2 = x2 + fw * preset["xpad"]
    ny2 = y2 + fh * preset["bottom"]
    nx1 = max(0.0, nx1)
    ny1 = max(0.0, ny1)
    nx2 = min(float(image_w), nx2)
    ny2 = min(float(image_h), ny2)
    return nx1, ny1, nx2, ny2, score


def expand_box(
    box: tuple[float, float, float, float], padding: float, image_w: int, image_h: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    px = padding * bw
    py = padding * bh
    return (
        max(0, int(round(x1 - px))),
        max(0, int(round(y1 - py))),
        min(image_w, int(round(x2 + px))),
        min(image_h, int(round(y2 + py))),
    )


def pick_largest(boxes: list[tuple[float, float, float, float, float]]) -> tuple[float, float, float, float, float] | None:
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def expand_to_aspect(
    box: tuple[float, float, float, float],
    target_aspect: float,
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int]:
    """Expand a bbox to match a target width/height aspect ratio.

    The box is extended along the shorter axis (in source pixels) so the crop
    contains actual image content, not pad bars. When the image is too small or
    the desired box would run off an edge, the box is uniformly shrunk to fit
    and then shifted inward so the subject stays inside the frame.
    """
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return (int(x1), int(y1), int(x2), int(y2))

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    current = bw / bh
    if current > target_aspect:
        # Box is wider than we want -> grow height
        new_w = bw
        new_h = bw / target_aspect
    else:
        # Box is taller than we want -> grow width
        new_h = bh
        new_w = bh * target_aspect

    # If the desired box doesn't fit in the image, scale it down preserving
    # the target aspect ratio.
    max_w = float(image_w)
    max_h = float(image_h)
    scale = min(1.0, max_w / new_w, max_h / new_h)
    new_w *= scale
    new_h *= scale

    nx1 = cx - new_w / 2.0
    ny1 = cy - new_h / 2.0
    nx2 = cx + new_w / 2.0
    ny2 = cy + new_h / 2.0

    # Shift inward (rather than clipping) so the requested aspect ratio is
    # preserved exactly.
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0.0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0.0
    if nx2 > max_w:
        nx1 -= nx2 - max_w
        nx2 = max_w
    if ny2 > max_h:
        ny1 -= ny2 - max_h
        ny2 = max_h

    return (
        max(0, int(round(nx1))),
        max(0, int(round(ny1))),
        min(image_w, int(round(nx2))),
        min(image_h, int(round(ny2))),
    )


# --- output sizing ------------------------------------------------------------


def parse_color(text: str) -> tuple[int, int, int]:
    t = text.strip().lower()
    if t in {"white", "#ffffff"}:
        return (255, 255, 255)
    if t in {"black", "#000000"}:
        return (0, 0, 0)
    if t.startswith("#") and len(t) == 7:
        return (int(t[1:3], 16), int(t[3:5], 16), int(t[5:7], 16))
    return (0, 0, 0)


def to_square(image: Image.Image, pad_color: tuple[int, int, int]) -> Image.Image:
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), pad_color)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def resize_to_exact(
    image: Image.Image, width: int, height: int, fit: str, pad_color: tuple[int, int, int]
) -> Image.Image:
    w, h = image.size
    if fit == "fit":
        ratio = min(width / w, height / h)
        return image.resize((max(1, round(w * ratio)), max(1, round(h * ratio))), Image.LANCZOS)
    if fit == "cover":
        ratio = max(width / w, height / h)
        scaled = image.resize((max(1, round(w * ratio)), max(1, round(h * ratio))), Image.LANCZOS)
        left = (scaled.width - width) // 2
        top = (scaled.height - height) // 2
        return scaled.crop((left, top, left + width, top + height))
    if fit == "pad":
        ratio = min(width / w, height / h)
        scaled = image.resize((max(1, round(w * ratio)), max(1, round(h * ratio))), Image.LANCZOS)
        canvas = Image.new("RGB", (width, height), pad_color)
        canvas.paste(scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2))
        return canvas
    return image.resize((width, height), Image.LANCZOS)


def next_available_sidecar(base: Path) -> Path:
    """Return `base` if no file exists there, otherwise the same stem with the
    smallest integer suffix that doesn't collide.

    `test.crop.png` -> `test.crop.png` (if free)
                    -> `test.crop2.png` (if `test.crop.png` exists)
                    -> `test.crop3.png` (if both exist), etc.
    """
    if not base.exists():
        return base
    parent = base.parent
    stem = base.stem  # e.g. "test.crop"
    suffix = base.suffix  # e.g. ".png"
    n = 2
    while True:
        candidate = parent / f"{stem}{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def save_image(image: Image.Image, dest: Path, quality: int) -> None:
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


# --- main ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-crop images around a detected target.")
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--target", choices=["face", "upper_body", "torso", "person"], default="face")
    parser.add_argument("--padding", type=float, default=0.2, help="Fractional padding around the detection box (0.2 = 20%%).")
    parser.add_argument(
        "--output-size",
        default="none",
        help="'none' | 'square:NNN' | 'exact:WxH'",
    )
    parser.add_argument("--fit", choices=["fit", "cover", "pad", "stretch"], default="cover")
    parser.add_argument("--pad-color", default="black")
    parser.add_argument("--output-mode", choices=["replace", "sidecar"], default="sidecar")
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument(
        "--torso-tightness",
        choices=list(TORSO_LEVELS.keys()),
        default="medium",
        help="Only used when --target=torso. Controls how much of the torso below the head is included.",
    )
    args = parser.parse_args()

    images = load_image_list(Path(args.image_list))
    if not images:
        emit({"type": "done", "processed": 0, "total": 0})
        return

    pad_color = parse_color(args.pad_color)

    # Decide which models we need.
    needs_face = args.target in ("face", "upper_body", "torso")
    needs_yolo = args.target == "person"

    face_app = None
    yolo_model = None

    if needs_face:
        ensure_face_model_downloaded()
    if needs_yolo:
        weights_path = ensure_yolo_model_downloaded()

    emit({"type": "model_loading"})

    if needs_face:
        with contextlib.redirect_stdout(io.StringIO()):
            from insightface.app import FaceAnalysis

            face_app = FaceAnalysis(
                name=FACE_MODEL_PACK,
                allowed_modules=["detection"],
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            face_app.prepare(ctx_id=0, det_size=(640, 640))

    if needs_yolo:
        from ultralytics import YOLO

        yolo_model = YOLO(weights_path)

    # Parse the output-size spec.
    output_mode: str  # "none" | "square" | "exact"
    sq_size = 0
    ex_w = ex_h = 0
    spec = args.output_size.strip().lower()
    if spec == "none" or spec == "":
        output_mode = "none"
    elif spec.startswith("square:"):
        output_mode = "square"
        try:
            sq_size = max(1, int(spec.split(":", 1)[1]))
        except ValueError:
            sq_size = 512
    elif spec.startswith("exact:"):
        output_mode = "exact"
        try:
            wh = spec.split(":", 1)[1]
            w_str, h_str = wh.lower().split("x")
            ex_w = max(1, int(w_str))
            ex_h = max(1, int(h_str))
        except ValueError:
            ex_w = ex_h = 512
    else:
        output_mode = "none"

    total = len(images)
    processed = 0

    for index, image_path in enumerate(images):
        emit({"type": "progress", "current": index + 1, "total": total, "image": str(image_path)})
        try:
            with Image.open(image_path) as opened:
                src = opened.convert("RGB")
            img_w, img_h = src.size

            box5: tuple[float, float, float, float, float] | None
            if args.target == "face":
                faces = detect_faces(face_app, src)
                faces = [b for b in faces if b[4] >= args.min_confidence]
                box5 = pick_largest(faces)
            elif args.target == "upper_body":
                faces = detect_faces(face_app, src)
                faces = [b for b in faces if b[4] >= args.min_confidence]
                largest = pick_largest(faces)
                box5 = derive_upper_body_box(largest, img_w, img_h) if largest else None
            elif args.target == "torso":
                faces = detect_faces(face_app, src)
                faces = [b for b in faces if b[4] >= args.min_confidence]
                largest = pick_largest(faces)
                box5 = (
                    derive_torso_box(largest, img_w, img_h, tightness=args.torso_tightness)
                    if largest
                    else None
                )
            else:  # person
                people = detect_people(yolo_model, src)
                people = [b for b in people if b[4] >= args.min_confidence]
                box5 = pick_largest(people)

            if box5 is None:
                emit({"type": "warning", "image": str(image_path), "message": f"No {args.target} detected"})
                continue

            x1, y1, x2, y2, _score = box5
            # First expand by user-requested padding, then reshape to the
            # requested aspect ratio. Square is the default (1:1); 'exact' mode
            # respects the user's W:H choice; 'none' still crops square because
            # the user asked for square outputs.
            xi1, yi1, xi2, yi2 = expand_box((x1, y1, x2, y2), max(0.0, args.padding), img_w, img_h)
            if output_mode == "exact":
                target_aspect = ex_w / float(ex_h) if ex_h > 0 else 1.0
            else:
                target_aspect = 1.0
            xi1, yi1, xi2, yi2 = expand_to_aspect((xi1, yi1, xi2, yi2), target_aspect, img_w, img_h)
            if xi2 - xi1 < 4 or yi2 - yi1 < 4:
                emit({"type": "warning", "image": str(image_path), "message": "Detection too small"})
                continue

            cropped = src.crop((xi1, yi1, xi2, yi2))

            if output_mode == "square":
                # Crop is already square; just resize to the requested edge.
                final = cropped.resize((sq_size, sq_size), Image.LANCZOS)
            elif output_mode == "exact":
                # Crop already matches the target aspect; a plain resize is
                # lossless of framing. (fit/cover/pad are unused now.)
                final = cropped.resize((ex_w, ex_h), Image.LANCZOS)
            else:
                final = cropped

            # Sidecar uses ".crop" suffix; replace overwrites original.
            # Sidecars never overwrite an existing crop - auto-increment instead
            # so multiple runs accumulate (test.crop.png, test.crop2.png, ...).
            if args.output_mode == "sidecar":
                dest = next_available_sidecar(
                    image_path.with_name(f"{image_path.stem}.crop{image_path.suffix}")
                )
            else:
                dest = image_path

            save_image(final, dest, args.quality)

            processed += 1
            emit(
                {
                    "type": "result",
                    "image": str(image_path),
                    "newPath": str(dest),
                    "width": final.width,
                    "height": final.height,
                }
            )
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
