import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import {
  resolveEffectiveHfCache,
  dirSizeBytes,
  diskFor,
  formatCachePath,
  isWritable,
} from '@/server/hfCache';

export const runtime = 'nodejs';

// GET /api/hf-cache
//   -> { current: { path, isDefault, sizeBytes, freeBytes, totalBytes } }
// GET /api/hf-cache?candidate=<folder>
//   -> also { candidate: { path (formatted), exists, writable, freeBytes, totalBytes } }
export async function GET(request: NextRequest) {
  try {
    const effective = await resolveEffectiveHfCache();
    const currentDisk = await diskFor(effective.path);
    const current = {
      path: effective.path,
      isDefault: effective.isDefault,
      sizeBytes: dirSizeBytes(effective.path),
      freeBytes: currentDisk?.freeBytes ?? null,
      totalBytes: currentDisk?.totalBytes ?? null,
    };

    const candidateParam = request.nextUrl.searchParams.get('candidate');
    let candidate:
      | { path: string; exists: boolean; writable: boolean; freeBytes: number | null; totalBytes: number | null }
      | undefined;

    if (candidateParam && candidateParam.trim() !== '') {
      const formatted = formatCachePath(candidateParam.trim());
      const disk = await diskFor(formatted);
      candidate = {
        path: formatted,
        exists: fs.existsSync(formatted),
        writable: isWritable(formatted),
        freeBytes: disk?.freeBytes ?? null,
        totalBytes: disk?.totalBytes ?? null,
      };
    }

    return NextResponse.json({ current, candidate });
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to read HF cache info: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 },
    );
  }
}
