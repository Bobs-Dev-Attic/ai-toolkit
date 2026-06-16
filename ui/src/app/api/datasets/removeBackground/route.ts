import { NextResponse } from 'next/server';
import {
  filterImagesWithinFolder,
  resolveDatasetFolder,
  spawnImageOpStream,
} from '@/server/imageOps';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const ALLOWED_MODELS = new Set(['u2net', 'u2netp', 'isnet-general-use', 'birefnet-general', 'birefnet-general-lite']);

export async function POST(request: Request) {
  let body: any;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }

  const { datasetName, images, model = 'u2net', bg = 'transparent', outputMode = 'sidecar' } = body;

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

  const extraArgs = [
    '--model',
    ALLOWED_MODELS.has(model) ? model : 'u2net',
    '--bg',
    String(bg || 'transparent'),
    '--output-mode',
    outputMode === 'replace' ? 'replace' : 'sidecar',
  ];

  return spawnImageOpStream({
    request,
    script: 'remove_background.py',
    images: validImages,
    extraArgs,
  });
}
