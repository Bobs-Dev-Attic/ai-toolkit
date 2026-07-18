"""
AI-Toolkit training diagnostic.

Read-only. Does not install, uninstall, or upgrade anything. Checks:

  1. Which Python interpreter is running this script and where it lives.
  2. Whether the bundled python_embeded interpreter is reachable.
  3. Every package pinned in requirements.txt / requirements_base.txt:
       - Is it importable?
       - Is the installed version what the file pins?
       - Where on disk is it loaded from?
  4. Torch / CUDA / GPU sanity (driver visible, build version, device list).
  5. The exact import chain that the extension loader walks at job start —
     so missing modules and wrong-version modules surface BEFORE you run a
     training job and discover them in the log.
  6. A quick check of the Next.js spawner's interpreter — same logic the
     UI uses (`ui/cron/pythonPath.ts`) — to confirm jobs will spawn against
     the bundled Python rather than a global one on PATH.

Run from anywhere. Nothing is written. Exit code is 0 if all checks pass,
1 if anything looks wrong. Pipe to a file with `> diagnose.txt 2>&1` if
you want to share the output.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# `packaging` ships with pip, so it's present in any modern Python env. We
# need it for full PEP 508 specifier handling (==, >=, <=, ~=, !=, multi-spec
# lines like `numpy<3.0.0,>=2.3.0`, environment markers, extras).
try:
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version
    _HAVE_PACKAGING = True
except ImportError:
    _HAVE_PACKAGING = False

# ---- ANSI colors (Windows 10+ terminals support them, also works in PowerShell) ----
RESET = "\033[0m"
DIM = "\033[90m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"

# Disable colors if we're not on a tty (e.g. piped to a file) and ANSI doesn't help.
if not sys.stdout.isatty():
    RESET = DIM = RED = GREEN = YELLOW = BLUE = BOLD = ""


def header(title: str) -> None:
    print()
    print(f"{BOLD}{BLUE}=== {title} ==={RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}OK{RESET}    {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}WARN{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")


def info(msg: str) -> None:
    print(f"  {DIM}info{RESET}  {msg}")


PROBLEMS: list[str] = []


def record_fail(msg: str) -> None:
    fail(msg)
    PROBLEMS.append(msg)


# ---------- locate the toolkit & launcher ----------

def find_toolkit_root() -> Optional[Path]:
    """The folder containing run.py + requirements.txt."""
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, *here.parents]:
        if (candidate / "run.py").exists() and (candidate / "requirements.txt").exists():
            return candidate
    return None


def find_easy_install_root(toolkit_root: Optional[Path]) -> Optional[Path]:
    """The folder that contains python_embeded + Start-AI-Toolkit.bat."""
    if not toolkit_root:
        return None
    for candidate in [toolkit_root.parent, toolkit_root.parent.parent, toolkit_root.parent.parent.parent]:
        if (candidate / "python_embeded" / "python.exe").exists() or (candidate / "Start-AI-Toolkit.bat").exists():
            return candidate
    return None


# ---------- requirements parsing ----------

def parse_requirements(toolkit_root: Path) -> list[dict]:
    """
    Returns a list of dicts. Common keys: raw, name, source, kind.
    `kind` is one of:
      - "spec"      → has `req` (a packaging.Requirement). Uses SpecifierSet for verdict.
      - "unpinned"  → bare package name, no constraints.
      - "git"       → git+https install. Has `git_ref` (commit/branch/tag string).
    """
    seen: list[dict] = []

    def parse_file(path: Path) -> None:
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("-r "):
                included = path.parent / raw[3:].strip()
                parse_file(included)
                continue
            if raw.startswith("git+"):
                # e.g. git+https://github.com/huggingface/diffusers.git@<sha>
                m = re.search(r"/([^/]+?)\.git(?:@(.+))?$", raw)
                name = m.group(1) if m else raw
                ref = (m.group(2) if (m and m.group(2)) else "")
                seen.append({"raw": raw, "name": name, "git_ref": ref,
                             "kind": "git", "source": path.name})
                continue
            if _HAVE_PACKAGING:
                try:
                    req = Requirement(raw)
                    if req.specifier:
                        seen.append({"raw": raw, "name": req.name, "req": req,
                                     "kind": "spec", "source": path.name})
                    else:
                        seen.append({"raw": raw, "name": req.name, "req": req,
                                     "kind": "unpinned", "source": path.name})
                    continue
                except Exception as e:
                    seen.append({"raw": raw, "name": raw, "parse_error": str(e),
                                 "kind": "unparseable", "source": path.name})
                    continue
            # Fallback when packaging isn't importable for some reason.
            seen.append({"raw": raw, "name": raw.split("=")[0].split("<")[0].split(">")[0].strip(),
                         "kind": "unpinned", "source": path.name})

    parse_file(toolkit_root / "requirements.txt")
    return seen


def normalize_pkg_name(name: str) -> str:
    return name.lower().replace("_", "-")


def installed_version(name: str) -> Optional[str]:
    candidates = [name, name.replace("-", "_"), name.replace("_", "-")]
    for candidate in candidates:
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def installed_location(name: str) -> Optional[str]:
    candidates = [name, name.replace("-", "_"), name.replace("_", "-")]
    for candidate in candidates:
        try:
            dist = importlib.metadata.distribution(candidate)
            return str(dist.locate_file(""))
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


# ---------- import probes ----------

# Modules the AI-Toolkit extension loader walks during get_all_extensions().
# If ANY of these fails, every training job fails before it starts — exactly
# the failure mode the user has been hitting.
EXTENSION_IMPORT_PROBES: list[tuple[str, str]] = [
    # (display name, importable module path)
    ("transformers", "transformers"),
    ("transformers.utils.generic.merge_with_config_defaults",
     "transformers.utils.generic:merge_with_config_defaults"),
    ("diffusers", "diffusers"),
    ("diffusers.ErnieImagePipeline", "diffusers:ErnieImagePipeline"),
    ("diffusers.AutoencoderKLFlux2", "diffusers:AutoencoderKLFlux2"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchao", "torchao"),
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("librosa", "librosa"),
    ("torchcodec", "torchcodec"),
    ("av", "av"),
    ("mutagen", "mutagen"),
    ("peft", "peft"),
    ("bitsandbytes", "bitsandbytes"),
    ("optimum.quanto", "optimum.quanto"),
    ("sentencepiece", "sentencepiece"),
    ("controlnet_aux", "controlnet_aux"),
    ("k_diffusion", "k_diffusion"),
    ("open_clip", "open_clip"),
    ("timm", "timm"),
    ("kornia", "kornia"),
    ("albumentations", "albumentations"),
    ("lycoris", "lycoris"),
    ("lpips", "lpips"),
    ("pytorch_fid", "pytorch_fid"),
    ("pytorch_wavelets", "pytorch_wavelets"),
    ("prodigyopt", "prodigyopt"),
    ("invisible_watermark", "imwatermark"),
    ("hf_transfer", "hf_transfer"),
]


def probe_import(spec: str) -> tuple[bool, str]:
    """spec is either 'module.path' or 'module.path:symbol'."""
    if ":" in spec:
        mod_path, sym = spec.split(":", 1)
    else:
        mod_path, sym = spec, None
    try:
        mod = importlib.import_module(mod_path)
    except BaseException as e:  # ImportError, OSError from DLL load, etc.
        return False, f"{type(e).__name__}: {e}"
    if sym:
        if not hasattr(mod, sym):
            return False, f"module imported but has no attribute '{sym}'"
    return True, getattr(mod, "__version__", "") or ""


# ---------- spawner check ----------

def check_ui_spawner(toolkit_root: Path, easy_root: Optional[Path]) -> None:
    """
    Mirror the logic in ui/cron/pythonPath.ts so we can tell the user
    whether the Next.js UI will spawn jobs against the bundled Python or
    against a global one.
    """
    candidates: list[Path] = []
    if easy_root:
        candidates.append(easy_root / "python_embeded" / "python.exe")
    # The TS file resolves TOOLKIT_ROOT to the `ui/` folder so it also walks
    # up from there. We just check both the easy-install root and one above
    # the toolkit, matching the .ts logic's reach.
    candidates.append(toolkit_root.parent / "python_embeded" / "python.exe")
    candidates.append(toolkit_root.parent.parent / "python_embeded" / "python.exe")
    candidates.append(toolkit_root / ".venv" / "Scripts" / "python.exe")
    candidates.append(toolkit_root / "venv" / "Scripts" / "python.exe")

    for c in candidates:
        if c.exists():
            ok(f"UI spawner will use: {c}")
            return

    path_python = shutil.which("python.exe") or shutil.which("python")
    if path_python:
        warn(f"UI spawner will fall back to PATH python: {path_python}")
        warn("  This is almost certainly NOT the embedded interpreter — training jobs will likely fail to import packages.")
        PROBLEMS.append("UI spawner falling back to PATH python")
    else:
        record_fail("UI spawner has no Python to call and PATH has none either.")


# ---------- GPU / CUDA check ----------

def check_torch_cuda() -> None:
    try:
        import torch  # noqa: WPS433
    except Exception as e:
        record_fail(f"torch failed to import: {type(e).__name__}: {e}")
        return
    info(f"torch {torch.__version__}")
    info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        warn("CUDA not available — training will fall back to CPU (essentially unusable for Z-Image / FLUX).")
        return
    info(f"CUDA build version: {torch.version.cuda}")
    info(f"cuDNN version: {torch.backends.cudnn.version()}")
    n = torch.cuda.device_count()
    info(f"Visible GPUs: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        info(f"  [{i}] {props.name} — {props.total_memory / (1024**3):.1f} GiB")


# ---------- main ----------

def main() -> int:
    header("Interpreter")
    info(f"Running with: {sys.executable}")
    info(f"Version:      {sys.version.split()[0]}")
    info(f"sys.prefix:   {sys.prefix}")
    info(f"cwd:          {os.getcwd()}")

    toolkit_root = find_toolkit_root()
    if not toolkit_root:
        record_fail("Could not find an AI-Toolkit folder (no run.py + requirements.txt above this script).")
        return 1
    info(f"Toolkit root: {toolkit_root}")

    easy_root = find_easy_install_root(toolkit_root)
    if easy_root:
        info(f"Launcher root: {easy_root}")
        embedded = easy_root / "python_embeded" / "python.exe"
        if embedded.exists():
            ok(f"Bundled python_embeded found at {embedded}")
            running_under_embedded = Path(sys.executable).resolve() == embedded.resolve()
            if not running_under_embedded:
                warn(f"You are NOT running this diagnostic with python_embeded — results below describe {sys.executable}, not the bundled environment.")
        else:
            warn("Launcher root has no python_embeded/python.exe — the bundled interpreter may be elsewhere.")
    else:
        warn("Couldn't locate an Easy-Install root (no python_embeded/Start-AI-Toolkit.bat above the toolkit).")

    # ---- requirements check ----
    header("requirements.txt vs installed")
    if not _HAVE_PACKAGING:
        warn("The `packaging` library isn't importable in this Python — strict constraint checking is disabled.")
        warn("Install it with: python -m pip install packaging")
    reqs = parse_requirements(toolkit_root)
    info(f"{len(reqs)} requirement lines parsed.")
    missing = 0
    mismatched = 0
    matched = 0
    skipped = 0
    for r in reqs:
        name = r["name"]
        if r["kind"] == "git":
            ver = installed_version(name)
            loc = installed_location(name)
            ref = r.get("git_ref", "")
            ref_short = (ref[:12] + "…") if len(ref) > 12 else ref
            if not ver:
                record_fail(f"{name}: git pin from {r['source']} — package NOT installed")
                missing += 1
                continue
            extra = f" (git ref expected: {ref_short})" if ref_short else ""
            ok(f"{name} {ver}{extra} at {loc}")
            matched += 1
            continue

        if r["kind"] == "unparseable":
            warn(f"{r['raw']!r} from {r['source']} — could not parse: {r.get('parse_error', '?')}")
            skipped += 1
            continue

        if r["kind"] == "unpinned":
            ver = installed_version(name)
            if not ver:
                record_fail(f"{name}: required by {r['source']} — NOT installed")
                missing += 1
                continue
            ok(f"{name} {ver} (unpinned)")
            matched += 1
            continue

        # kind == "spec" — has a packaging.Requirement with at least one specifier.
        req = r["req"]
        ver = installed_version(name)
        spec_str = str(req.specifier)
        if not ver:
            record_fail(f"{name}{spec_str}: NOT installed (from {r['source']})")
            missing += 1
            continue

        # Honor environment markers (e.g. `; sys_platform == 'win32'`) — if the
        # marker excludes us, the constraint doesn't apply on this machine.
        if req.marker is not None:
            try:
                if not req.marker.evaluate():
                    info(f"{name} {ver} — marker `{req.marker}` excludes this environment, constraint skipped")
                    skipped += 1
                    continue
            except Exception:
                pass  # malformed marker — fall through and check anyway

        # Strict constraint check.
        try:
            satisfied = req.specifier.contains(ver, prereleases=True)
        except InvalidVersion as e:
            warn(f"{name}: installed version {ver!r} isn't a valid PEP 440 version ({e}); skipping constraint check")
            skipped += 1
            continue

        if satisfied:
            # Show what the constraint is so the OK line is informative,
            # but call it out as an exact-match when it's `==`.
            is_exact = all(s.operator == "==" for s in req.specifier)
            label = "" if is_exact else f" (satisfies {spec_str})"
            ok(f"{name} {ver}{label}")
            matched += 1
        else:
            # Diagnose direction so the user knows whether to upgrade or downgrade.
            verdict = "version mismatch"
            try:
                v = Version(ver)
                # Walk specifiers to figure out if installed is below a lower bound
                # or above an upper bound — clearer than just printing the spec.
                hints = []
                for spec in req.specifier:
                    op = spec.operator
                    try:
                        sv = Version(spec.version)
                    except InvalidVersion:
                        continue
                    if op in (">=", ">") and v < sv:
                        hints.append(f"needs UPGRADE to satisfy {op}{spec.version}")
                    elif op in ("<=", "<") and v > sv:
                        hints.append(f"needs DOWNGRADE to satisfy {op}{spec.version}")
                    elif op == "==" and v != sv:
                        hints.append("needs DOWNGRADE" if v > sv else "needs UPGRADE")
                    elif op == "!=" and v == sv:
                        hints.append(f"must NOT be =={spec.version}")
                    elif op == "~=":
                        hints.append(f"needs version compatible with ~={spec.version}")
                if hints:
                    verdict = "; ".join(hints)
            except InvalidVersion:
                pass
            record_fail(f"{name}: installed {ver}, required {spec_str} — {verdict} (from {r['source']})")
            mismatched += 1

    info(f"matched={matched}  mismatched={mismatched}  missing={missing}  skipped={skipped}")

    # ---- import probes (the real test) ----
    header("Extension-loader import chain")
    for label, spec in EXTENSION_IMPORT_PROBES:
        success, detail = probe_import(spec)
        if success:
            if detail:
                ok(f"{label}  ({detail})")
            else:
                ok(label)
        else:
            record_fail(f"{label}  — {detail}")

    # ---- torch / cuda ----
    header("Torch / CUDA")
    check_torch_cuda()

    # ---- UI spawner ----
    header("UI training spawner")
    check_ui_spawner(toolkit_root, easy_root)

    # ---- summary ----
    header("Summary")
    if PROBLEMS:
        fail(f"{len(PROBLEMS)} problem(s) found:")
        for p in PROBLEMS:
            print(f"        - {p}")
        print()
        print("Suggested next steps:")
        print("  - Stop the AI-Toolkit server (close the Start-AI-Toolkit.bat window).")
        print("  - For missing/mismatched packages, run:")
        print(r'      .\python_embeded\python.exe -I -m pip install --no-cache "<name>==<expected>"')
        print(r"  - If the spawner is falling back to PATH python, rebuild the UI so")
        print(r"      ui/cron/pythonPath.ts picks up the bundled interpreter.")
        return 1
    ok("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
