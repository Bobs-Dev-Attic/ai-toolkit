import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import fs from 'fs';
import path from 'path';

const prisma = new PrismaClient();

const MODEL_EXT = new Set(['.safetensors', '.ckpt', '.bin', '.pt', '.pth', '.gguf']);

interface ModelFile {
  /** Absolute path on disk. */
  path: string;
  /** Filename (no directory). */
  name: string;
  /** Relative path from the configured MODELS_FOLDER. */
  rel: string;
  /** Extension including the dot. */
  ext: string;
  /** Bytes. */
  size: number;
  /** mtime as ms since epoch. */
  modified_at: number;
  /** Best-effort sniffed metadata from the file header (safetensors only). */
  metadata?: {
    /** Number of tensors in the file. */
    tensor_count?: number;
    /** Approximate parameter count summed across tensors. */
    param_count?: number;
    /** Distinct dtypes seen in the header (e.g. ["F16", "BF16"]). */
    dtypes?: string[];
    /** Tensor names that look like a base-architecture giveaway. */
    arch_hints?: string[];
  };
}

function walk(dir: string, root: string, depth: number, max: number, out: ModelFile[]) {
  if (depth > max) return;
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(full, root, depth + 1, max, out);
    } else if (entry.isFile()) {
      const ext = path.extname(entry.name).toLowerCase();
      if (!MODEL_EXT.has(ext)) continue;
      try {
        const stat = fs.statSync(full);
        const item: ModelFile = {
          path: full,
          name: entry.name,
          rel: path.relative(root, full),
          ext,
          size: stat.size,
          modified_at: stat.mtimeMs,
        };
        if (ext === '.safetensors' && stat.size > 16) {
          item.metadata = readSafetensorsHeader(full);
        }
        out.push(item);
      } catch {
        // best-effort; skip
      }
    }
  }
}

/**
 * Read the JSON header of a .safetensors file without loading any tensor data.
 * The format is: 8 bytes little-endian header length, then UTF-8 JSON of that
 * length describing every tensor. We cap at 8 MB to avoid bad files.
 */
function readSafetensorsHeader(filePath: string): ModelFile['metadata'] {
  let fd: number | null = null;
  try {
    fd = fs.openSync(filePath, 'r');
    const lenBuf = Buffer.alloc(8);
    fs.readSync(fd, lenBuf, 0, 8, 0);
    const headerLen = Number(lenBuf.readBigUInt64LE(0));
    if (!Number.isFinite(headerLen) || headerLen <= 0 || headerLen > 8 * 1024 * 1024) {
      return undefined;
    }
    const headerBuf = Buffer.alloc(headerLen);
    fs.readSync(fd, headerBuf, 0, headerLen, 8);
    const json = JSON.parse(headerBuf.toString('utf-8'));
    const dtypes = new Set<string>();
    const archHints = new Set<string>();
    let tensorCount = 0;
    let paramCount = 0;
    for (const [key, val] of Object.entries(json)) {
      if (key === '__metadata__' || typeof val !== 'object' || val === null) continue;
      tensorCount += 1;
      const dtype = (val as any).dtype;
      const shape = (val as any).shape;
      if (typeof dtype === 'string') dtypes.add(dtype);
      if (Array.isArray(shape)) {
        const product = shape.reduce((a: number, b: number) => a * (Number.isFinite(b) ? b : 1), 1);
        if (Number.isFinite(product)) paramCount += product;
      }
      // Heuristic arch hints — prefixes that commonly identify the model family.
      const hintPrefixes = [
        'transformer.', 'text_encoder.', 'text_encoder_2.', 'unet.', 'vae.',
        'model.diffusion_model.', 'lora_unet_', 'lora_te_', 'double_blocks.', 'single_blocks.',
        'time_embed.', 'down_blocks.', 'mid_block.', 'up_blocks.',
      ];
      for (const p of hintPrefixes) {
        if (key.startsWith(p)) {
          archHints.add(p.replace(/\.$/, ''));
          break;
        }
      }
    }
    return {
      tensor_count: tensorCount,
      param_count: paramCount,
      dtypes: Array.from(dtypes).sort(),
      arch_hints: Array.from(archHints).sort(),
    };
  } catch {
    return undefined;
  } finally {
    if (fd !== null) {
      try { fs.closeSync(fd); } catch {}
    }
  }
}

export async function GET() {
  try {
    const row = await prisma.settings.findUnique({ where: { key: 'MODELS_FOLDER' } });
    const folder = row?.value || '';
    if (!folder) {
      return NextResponse.json({ folder: '', models: [], error: 'No MODELS_FOLDER configured.' });
    }
    if (!fs.existsSync(folder)) {
      return NextResponse.json({ folder, models: [], error: 'Folder does not exist.' });
    }
    const out: ModelFile[] = [];
    walk(folder, folder, 0, 5, out);
    out.sort((a, b) => b.modified_at - a.modified_at);
    return NextResponse.json({ folder, models: out });
  } catch (error) {
    console.error('Error listing models:', error);
    return NextResponse.json({ error: 'Failed to list models' }, { status: 500 });
  }
}
