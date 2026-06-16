import { NextResponse } from 'next/server';
import {
  filterImagesWithinFolder,
  resolveDatasetFolder,
  spawnImageOpStream,
} from '@/server/imageOps';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const ALLOWED_MODELS = new Set(['x2', 'x4', 'x4plus']);

export async function POST(request: Request) {
  let body: any;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }

  const { datasetName, images, model = 'x4', maxSide = 2048, outputMode = 'sidecar' } = body;

  if (!datasetName || typeof datasetName !== 'string') {
    return NextResponse.json({ error: 'Dataset name is required' }, { status: 400 });
  }

  let datasetFolder: string;
  try {
    datasetFolder = await resolveDatasetFolder(datasetName);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Invalid dataset' },
      { status: 400 },
    );
  }

  const validImages = filterImagesWithinFolder(images, datasetFolder);
  if (validImages.length === 0) {
    return NextResponse.json({ error: 'No valid images in selection' }, { status: 400 });
  }

  const clampedMaxSide = Math.max(0, Math.min(16384, Number(maxSide) || 0));
  const extraArgs = [
    '--model',
    ALLOWED_MODELS.has(model) ? model : 'x4',
    '--max-side',
    String(clampedMaxSide),
    '--output-mode',
    outputMode === 'replace' ? 'replace' : 'sidecar',
  ];

  return spawnImageOpStream({
    request,
    script: 'upscale_images.py',
    images: validImages,
    extraArgs,
  });
}
