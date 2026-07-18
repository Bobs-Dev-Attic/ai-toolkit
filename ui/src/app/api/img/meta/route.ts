import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder, getDataRoot } from '@/server/settings';

export const runtime = 'nodejs';

// Returns lightweight file metadata (size + timestamps) for a sample file.
// Dimensions are read client-side from the loaded media element, so we only
// need to stat the file here — no need to read image bytes.
export async function GET(request: NextRequest) {
  try {
    const raw = request.nextUrl.searchParams.get('path');
    if (!raw) {
      return NextResponse.json({ error: 'path is required' }, { status: 400 });
    }

    const filepath = decodeURIComponent(raw);

    // Same allow-list security check used by the image serving route.
    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();
    const dataRoot = await getDataRoot();
    const allowedDirs = [datasetRoot, trainingRoot, dataRoot];

    const resolved = path.resolve(filepath);
    const isAllowed = allowedDirs.some(
      allowedDir => resolved === allowedDir || resolved.startsWith(allowedDir + path.sep),
    );

    if (!isAllowed) {
      return new NextResponse('Access denied', { status: 403 });
    }

    const stat = await fs.promises.stat(resolved).catch(() => null);
    if (!stat || !stat.isFile()) {
      return new NextResponse('File not found', { status: 404 });
    }

    // birthtime can be 0 / unreliable on some filesystems; fall back to mtime.
    const createdMs = stat.birthtimeMs && stat.birthtimeMs > 0 ? stat.birthtimeMs : stat.mtimeMs;

    return NextResponse.json({
      sizeBytes: stat.size,
      createdMs,
      modifiedMs: stat.mtimeMs,
    });
  } catch (error) {
    console.error('Error reading image meta:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
