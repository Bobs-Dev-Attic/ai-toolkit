# Publishing this fork to your GitHub

Step-by-step guide for pushing this fork (your custom dataset-prep
workbench on top of upstream `ostris/ai-toolkit`) to your own GitHub
account.

## Prerequisites

- A GitHub account
- `git` installed locally (run `git --version` to check)
- An [SSH key set up](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
  or [GitHub CLI](https://cli.github.com/) installed (`gh --version`).
  Either works; the steps below assume SSH.

## What this directory actually is

Your `AI-Toolkit/` folder is a clone of `https://github.com/ostris/ai-toolkit.git`
with our modifications. The outer `C:\Users\bobch\Documents\ai-toolkit\`
directory (with `python_embeded/`, `Start-AI-Toolkit.bat`) is **not** a
git repo and shouldn't be — those are just your local install
artifacts. Only the `AI-Toolkit/` subdirectory gets pushed.

## Current state

Open a PowerShell or terminal in `AI-Toolkit/` and run:

```powershell
cd C:\Users\bobch\Documents\ai-toolkit\AI-Toolkit
git status
git remote -v
```

You should see:

- Remote `origin` pointing to `ostris/ai-toolkit.git`
- A mix of modified files (`M ui/...`) and untracked new files (`?? scripts/...`)

## Step 1 — Fork `ostris/ai-toolkit` on GitHub

1. Go to https://github.com/ostris/ai-toolkit
2. Click **Fork** in the top-right
3. Choose your account as the destination
4. Leave the fork name as `ai-toolkit` (or rename if you prefer)
5. **Uncheck** "Copy the main branch only" if you want all branches
6. Click **Create fork**

You now own `https://github.com/<your-username>/ai-toolkit`.

## Step 2 — Repoint your local clone to your fork

By default `origin` still points at Ostris's repo. Rename that to
`upstream` (so you can pull future Ostris updates) and add your fork as
the new `origin`:

```powershell
# rename the existing remote to upstream
git remote rename origin upstream

# add your fork as origin (replace <your-username>)
git remote add origin git@github.com:<your-username>/ai-toolkit.git

# verify
git remote -v
```

Expected output:
```
origin    git@github.com:<your-username>/ai-toolkit.git (fetch)
origin    git@github.com:<your-username>/ai-toolkit.git (push)
upstream  https://github.com/ostris/ai-toolkit.git (fetch)
upstream  https://github.com/ostris/ai-toolkit.git (push)
```

> **If you don't use SSH:** use the HTTPS URL instead:
> `https://github.com/<your-username>/ai-toolkit.git`. GitHub will prompt
> for credentials on push (or use a personal-access token).

## Step 3 — Make a feature branch for your work

Don't commit directly to `main` — keep `main` mirroring Ostris's `main`
so you can pull future updates cleanly. Create a feature branch:

```powershell
git checkout -b dataset-prep-workbench
```

## Step 4 — Review what will be committed

Before committing, sanity-check what's about to be tracked. The toolkit's
existing `.gitignore` already excludes the big stuff (`datasets/`,
`output/`, `aitk_db.db`, Python caches, `node_modules`, `.next`), so
this should be only our additions:

```powershell
git status
```

Specifically you should see:
- **Modified**: `ui/package.json`, `ui/package-lock.json`,
  `ui/src/app/api/datasets/listImages/route.ts`,
  `ui/src/app/datasets/[datasetName]/page.tsx`,
  `ui/src/components/DatasetImageCard.tsx`
- **New**: `scripts/auto_crop.py`, `scripts/caption_dataset.py`,
  `scripts/remove_background.py`, `scripts/resize_images.py`,
  `scripts/upscale_images.py`, several new directories under
  `ui/src/app/api/datasets/` and `ui/src/server/`, and the
  `docs/fork/` directory

If you see anything else (e.g. a stray `.env`, sample images, model
weights), do **not** commit them — investigate first.

To inspect specific changes:

```powershell
git diff ui/src/components/DatasetImageCard.tsx
git diff --stat                     # quick overview of every change
```

## Step 5 — Commit

Stage everything we intend to push (specific paths, not `git add -A`):

```powershell
git add scripts/auto_crop.py scripts/caption_dataset.py `
        scripts/remove_background.py scripts/resize_images.py `
        scripts/upscale_images.py `
        ui/src/server/imageOps.ts `
        ui/src/app/api/datasets/caption `
        ui/src/app/api/datasets/resize `
        ui/src/app/api/datasets/removeBackground `
        ui/src/app/api/datasets/upscale `
        ui/src/app/api/datasets/autoCrop `
        ui/src/app/api/img/bulkDelete `
        ui/src/app/api/datasets/listImages/route.ts `
        ui/src/app/datasets/`[datasetName`]/page.tsx `
        ui/src/components/DatasetImageCard.tsx `
        ui/src/app/jobs/new/configPresets.ts `
        ui/src/app/jobs/new/PresetPicker.tsx `
        ui/src/app/jobs/new/SimpleJob.tsx `
        ui/package.json ui/package-lock.json `
        docs/fork
```

> The backticks in PowerShell are line continuations. In bash, replace
> them with backslashes (`\`).

Commit with a clear, multi-line message:

```powershell
git commit -m "Add dataset-prep workbench (caption, crop, resize, bg-remove, upscale)" -m "Adds a streaming SSE op framework and five Python scripts (BLIP captioning, Pillow resize, rembg, Real-ESRGAN via spandrel, insightface auto-crop) wired into the Datasets page. Includes a bulk-action bar, show-metadata overlay, grid-size control, and floating progress modal. See docs/fork/CHANGES.md."
```

## Step 6 — Push to your fork

```powershell
git push -u origin dataset-prep-workbench
```

`-u` sets the upstream so future `git push` from this branch goes to
your fork without arguments.

## Step 7 — (Optional) Open a Pull Request against Ostris's repo

If you want to offer the work upstream:

1. Go to `https://github.com/<your-username>/ai-toolkit`
2. GitHub will show a yellow banner suggesting "Compare & pull request" —
   click it
3. Make sure the base is `ostris/ai-toolkit:main` and the compare is
   `<your-username>/ai-toolkit:dataset-prep-workbench`
4. Write a PR description; reference `docs/fork/CHANGES.md` for detail
5. Submit

Ostris may accept, modify, or decline — that's fine, your fork is yours
to maintain regardless.

## Step 8 — Future: pulling new upstream changes

When Ostris updates `main`, sync your fork:

```powershell
git checkout main
git fetch upstream
git merge upstream/main           # or: git reset --hard upstream/main
git push origin main              # update your fork's main

# rebase your feature branch on top
git checkout dataset-prep-workbench
git rebase main
# resolve any conflicts, then:
git push --force-with-lease origin dataset-prep-workbench
```

`--force-with-lease` is safer than `--force`: it refuses to overwrite
work pushed by someone else (relevant if you collaborate).

## Repo hygiene checklist

Before each push, run these:

- `git status` — verify nothing unintended is staged
- `git diff --cached` — review the final diff
- Confirm no real datasets or model weights are tracked:
  `git ls-files | grep -E "datasets/|output/|\.safetensors|\.pth"`
  → should be empty (or only the `output/.gitkeep` placeholder)
- Confirm no secrets: `git ls-files | grep -E "\.env|credentials|secret"`
  → should be empty

## What stays local (don't commit these)

These are ignored by `.gitignore` but worth being explicit about:

| Path | Why |
|---|---|
| `aitk_db.db` | Your local jobs database |
| `datasets/` | Your training images |
| `output/` | Training artifacts (LoRAs, samples) |
| `python_embeded/` | (outside the repo entirely) Local Python install |
| `node_modules/` | Reinstalled via `npm install` |
| `.next/` | Next.js build artifacts |
| Model caches under `~/.cache/huggingface/`, `~/.u2net/`, `~/.insightface/`, `~/.ultralytics/` | (outside the repo) HF / model downloads |

## If something goes wrong

- **"Permission denied (publickey)"**: SSH key isn't registered with
  GitHub. Run `ssh-keygen -t ed25519 -C "<your-email>"`, then add the
  contents of `~/.ssh/id_ed25519.pub` to GitHub → Settings → SSH and
  GPG keys.
- **"failed to push some refs"**: someone (or you on another machine)
  pushed first. `git pull --rebase origin dataset-prep-workbench`, then
  push again.
- **You accidentally committed a big file**: `git rm --cached <file>`,
  add the path to `.gitignore`, commit, force-push. For truly massive
  files already pushed, you may need `git filter-repo` to scrub
  history.
- **You want to undo everything since last commit**: `git reset --hard HEAD`
  (destructive — uncommitted work is gone).
