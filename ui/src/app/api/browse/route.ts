import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { getModelsFolder } from '@/server/settings';

export const runtime = 'nodejs';

// File extensions we flag as model checkpoints so the browser can highlight them.
const MODEL_EXT = new Set(['.safetensors', '.ckpt', '.bin', '.pt', '.pth', '.gguf']);

interface DirEntry {
  name: string;
  path: string;
}
interface FileEntry extends DirEntry {
  size: number;
  isModel: boolean;
}

// Lets the user pick a folder (or model file) from the machine's filesystem.
// This is a local desktop tool, so browsing arbitrary readable directories is
// consistent with how the app already accepts absolute paths anywhere.
export async function GET(request: NextRequest) {
  try {
    const requested = request.nextUrl.searchParams.get('path') || '';

    // Resolve a sensible starting location: the configured Models Folder,
    // falling back to the user's home directory.
    const fallback = async () => {
      const modelsFolder = await getModelsFolder();
      return modelsFolder && fs.existsSync(modelsFolder) ? modelsFolder : os.homedir();
    };

    let resolved = path.resolve(requested || (await fallback()));

    let stat: fs.Stats | null = null;
    try {
      stat = fs.statSync(resolved);
    } catch {
      // The requested path may be an HF repo id or a typo — don't error out,
      // just open a usable default location so the browser is always navigable.
      resolved = path.resolve(await fallback());
      try {
        stat = fs.statSync(resolved);
      } catch {
        return NextResponse.json({ error: 'Path not found' }, { status: 404 });
      }
    }

    // If they handed us a file, browse its containing directory instead.
    const dir = stat.isDirectory() ? resolved : path.dirname(resolved);

    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return NextResponse.json({ error: 'Cannot read directory' }, { status: 403 });
    }

    const dirs: DirEntry[] = [];
    const files: FileEntry[] = [];

    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue; // skip hidden/system dotfiles
      const full = path.join(dir, entry.name);
      try {
        if (entry.isDirectory()) {
          dirs.push({ name: entry.name, path: full });
        } else if (entry.isFile()) {
          const ext = path.extname(entry.name).toLowerCase();
          const st = fs.statSync(full);
          files.push({ name: entry.name, path: full, size: st.size, isModel: MODEL_EXT.has(ext) });
        }
      } catch {
        // skip entries we can't stat (permissions, broken links)
      }
    }

    dirs.sort((a, b) => a.name.localeCompare(b.name));
    files.sort((a, b) => a.name.localeCompare(b.name));

    // parent is null once we reach a filesystem root (e.g. C:\ or /).
    const parent = path.dirname(dir);
    const atRoot = parent === dir;

    return NextResponse.json({
      path: dir,
      parent: atRoot ? null : parent,
      sep: path.sep,
      dirs,
      files,
    });
  } catch (error) {
    console.error('Error browsing filesystem:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
