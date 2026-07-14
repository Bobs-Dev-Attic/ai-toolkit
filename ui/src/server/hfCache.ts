import fs from 'fs';
import os from 'os';
import path from 'path';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// The setting key that stores the user's chosen HuggingFace hub cache location.
// Empty/unset means "use the default location".
export const HF_CACHE_KEY = 'HF_HUB_CACHE';

// Where HF puts the model hub cache when the user hasn't overridden it. Respect
// any env the server itself was launched with, else the standard default.
export function defaultHfCachePath(): string {
  if (process.env.HF_HUB_CACHE) return process.env.HF_HUB_CACHE;
  if (process.env.HF_HOME) return path.join(process.env.HF_HOME, 'hub');
  return path.join(os.homedir(), '.cache', 'huggingface', 'hub');
}

export async function getStoredHfCache(): Promise<string> {
  try {
    const row = await prisma.settings.findFirst({ where: { key: HF_CACHE_KEY } });
    return row?.value?.trim() || '';
  } catch {
    return '';
  }
}

export async function resolveEffectiveHfCache(): Promise<{ path: string; isDefault: boolean }> {
  const stored = await getStoredHfCache();
  if (stored) return { path: stored, isDefault: false };
  return { path: defaultHfCachePath(), isDefault: true };
}

// Remember whatever HF_HUB_CACHE the UI server was launched with, so that
// clearing the setting reverts to it (or to the default) rather than sticking.
let _launchHfCache: string | undefined;
let _launchCaptured = false;

// Apply the configured cache location to THIS process's environment. Python we
// spawn from the UI server (captioning, upscaling, ui_scripts) inherits it via
// `{ ...process.env }`, so those downloads land on the chosen drive too. Cheap
// enough to call per user-triggered op; the training worker reads the DB
// directly (see cron/actions/startJob.ts).
export async function applyHfCacheEnv(): Promise<void> {
  if (!_launchCaptured) {
    _launchHfCache = process.env.HF_HUB_CACHE;
    _launchCaptured = true;
  }
  const stored = await getStoredHfCache();
  if (stored) {
    process.env.HF_HUB_CACHE = stored;
  } else if (_launchHfCache) {
    process.env.HF_HUB_CACHE = _launchHfCache;
  } else {
    delete process.env.HF_HUB_CACHE;
  }
}

// Normalize a folder the user picked into a proper cache path. We namespace it
// under huggingface/hub so we don't scatter models--* dirs directly into an
// arbitrary folder they selected (e.g. picking "D:\AI" -> "D:\AI\huggingface\hub").
export function formatCachePath(selected: string): string {
  const norm = path.resolve(selected);
  const lower = norm.toLowerCase().replace(/[\\/]+$/, '');
  const hubTail = path.join('huggingface', 'hub').toLowerCase();
  if (lower.endsWith(hubTail)) return norm;
  if (lower.endsWith('huggingface')) return path.join(norm, 'hub');
  return path.join(norm, 'huggingface', 'hub');
}

// Walk up until we find a directory that exists, so we can stat the drive even
// when the chosen cache folder hasn't been created yet.
export function nearestExistingDir(p: string): string {
  let cur = path.resolve(p);
  while (cur && !fs.existsSync(cur)) {
    const parent = path.dirname(cur);
    if (parent === cur) break;
    cur = parent;
  }
  return cur;
}

export async function diskFor(p: string): Promise<{ totalBytes: number; freeBytes: number } | null> {
  try {
    const st: any = await (fs.promises as any).statfs(nearestExistingDir(p));
    return { totalBytes: st.blocks * st.bsize, freeBytes: st.bavail * st.bsize };
  } catch {
    return null;
  }
}

export function isWritable(p: string): boolean {
  try {
    fs.accessSync(nearestExistingDir(p), fs.constants.W_OK);
    return true;
  } catch {
    return false;
  }
}

// Sum sizes of real files under `root`. Symlinks are skipped so HF snapshot->blob
// links aren't double-counted (on symlink-capable systems); on Windows-copy mode
// there are no links and every file is real, so the sum equals actual disk usage
// either way. Capped to avoid pathological walks.
export function dirSizeBytes(root: string, cap = 500000): number {
  if (!fs.existsSync(root)) return 0;
  let total = 0;
  let count = 0;
  const stack = [root];
  while (stack.length) {
    const dir = stack.pop() as string;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const e of entries) {
      if (++count > cap) return total;
      if (e.isSymbolicLink()) continue;
      const full = path.join(dir, e.name);
      if (e.isDirectory()) {
        stack.push(full);
      } else if (e.isFile()) {
        try {
          total += fs.statSync(full).size;
        } catch {
          /* skip unreadable */
        }
      }
    }
  }
  return total;
}
