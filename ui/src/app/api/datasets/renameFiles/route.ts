import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

const ALLOWED_EXT = /\.(jpg|jpeg|png|bmp|gif|tiff|webp|mp4|avi|mov|mkv|wmv|m4v|flv|mp3|wav|flac|ogg)$/i;

const DELIM: Record<string, string> = { underscore: '_', hyphen: '-', space: ' ', dot: '.', none: '' };

// Illegal Windows filename characters: < > : " / \ | ? *
const ILLEGAL = /[<>:"/\\|?*]/g;

// Strip only illegal filename characters; keep case and the user's spaces/dashes.
function sanitizePrefix(s: string): string {
  return (s || '').replace(ILLEGAL, '').replace(/^[.\s]+|[.\s]+$/g, '');
}

interface Plan {
  fromImg: string;
  toImg: string;
  fromTxt: string | null;
  toTxt: string | null;
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPaths, prefix, delimiter, padding, start } = body;

    if (!Array.isArray(imgPaths) || imgPaths.length === 0) {
      return NextResponse.json({ error: 'No files provided' }, { status: 400 });
    }
    const cleanPrefix = sanitizePrefix(String(prefix ?? ''));
    if (!cleanPrefix) {
      return NextResponse.json({ error: 'A prefix is required' }, { status: 400 });
    }
    const delim = DELIM[String(delimiter ?? 'underscore')] ?? '_';
    const pad = Math.max(0, Math.min(6, parseInt(String(padding ?? 3), 10) || 0));
    const startN = Number.isFinite(parseInt(String(start), 10)) ? parseInt(String(start), 10) : 1;

    const datasetsPath = await getDatasetsRoot();
    const trainingPath = await getTrainingFolder();

    // Validate every source path up front (inside a dataset root, supported type).
    const sources: string[] = [];
    for (const p of imgPaths) {
      if (typeof p !== 'string') continue;
      if (!p.startsWith(datasetsPath) && !p.startsWith(trainingPath)) {
        return NextResponse.json({ error: `Path is outside the datasets folder: ${p}` }, { status: 400 });
      }
      if (!ALLOWED_EXT.test(p)) {
        return NextResponse.json({ error: `Unsupported file type: ${path.basename(p)}` }, { status: 400 });
      }
      if (!fs.existsSync(p)) {
        return NextResponse.json({ error: `File no longer exists: ${path.basename(p)}` }, { status: 404 });
      }
      sources.push(p);
    }

    // Plan the new names (image + sibling .txt caption), numbered in order.
    const plans: Plan[] = [];
    const targetKeys = new Set<string>();
    let n = startN;
    for (const src of sources) {
      const dir = path.dirname(src);
      const ext = path.extname(src);
      const numStr = pad > 0 ? String(n).padStart(pad, '0') : String(n);
      const baseName = `${cleanPrefix}${delim}${numStr}`;
      const toImg = path.join(dir, baseName + ext);
      const key = toImg.toLowerCase();
      if (targetKeys.has(key)) {
        return NextResponse.json(
          { error: `Two files would receive the same name (${baseName + ext}). Adjust the numbering.` },
          { status: 400 },
        );
      }
      targetKeys.add(key);
      const fromTxt = src.replace(/\.[^./\\]+$/, '.txt');
      const hasTxt = fromTxt !== src && fs.existsSync(fromTxt);
      plans.push({
        fromImg: src,
        toImg,
        fromTxt: hasTxt ? fromTxt : null,
        toTxt: hasTxt ? path.join(dir, baseName + '.txt') : null,
      });
      n += 1;
    }

    // Reject collisions with existing files that are NOT part of this batch
    // (renaming among the selection itself is fine — handled via temp names).
    const srcLower = new Set(sources.map(s => s.toLowerCase()));
    const srcTxtLower = new Set(plans.filter(p => p.fromTxt).map(p => (p.fromTxt as string).toLowerCase()));
    for (const pl of plans) {
      if (fs.existsSync(pl.toImg) && !srcLower.has(pl.toImg.toLowerCase())) {
        return NextResponse.json(
          { error: `A file named ${path.basename(pl.toImg)} already exists. Pick a different prefix or start number.` },
          { status: 409 },
        );
      }
      if (pl.toTxt && fs.existsSync(pl.toTxt) && !srcTxtLower.has(pl.toTxt.toLowerCase())) {
        return NextResponse.json(
          { error: `A caption named ${path.basename(pl.toTxt)} already exists. Pick a different prefix or start number.` },
          { status: 409 },
        );
      }
    }

    // Two-phase rename via unique temp names so in-batch shuffles (e.g. shifting
    // every number by one) never clobber a file that hasn't been moved yet.
    const tmpSuffix = `.rntmp_${process.pid}_`;
    const moves: { tmp: string; to: string }[] = [];
    let i = 0;
    for (const pl of plans) {
      const tmpImg = `${pl.fromImg}${tmpSuffix}${i}`;
      fs.renameSync(pl.fromImg, tmpImg);
      moves.push({ tmp: tmpImg, to: pl.toImg });
      if (pl.fromTxt && pl.toTxt) {
        const tmpTxt = `${pl.fromTxt}${tmpSuffix}${i}`;
        fs.renameSync(pl.fromTxt, tmpTxt);
        moves.push({ tmp: tmpTxt, to: pl.toTxt });
      }
      i += 1;
    }
    for (const m of moves) {
      fs.renameSync(m.tmp, m.to);
    }

    return NextResponse.json({ renamed: plans.length });
  } catch (error) {
    console.error('renameFiles error', error);
    return NextResponse.json({ error: 'Failed to rename files' }, { status: 500 });
  }
}
