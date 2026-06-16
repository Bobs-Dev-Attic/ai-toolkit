import { NextResponse } from 'next/server';
import {
  filterImagesWithinFolder,
  resolveDatasetFolder,
  spawnImageOpStream,
} from '@/server/imageOps';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  let body: any;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }

  const {
    datasetName,
    images,
    mode = 'longest_side',
    width = 1024,
    height = 1024,
    fit = 'fit',
    padColor = 'black',
    format = 'keep',
    quality = 92,
  } = body;

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
    '--mode',
    mode === 'exact' ? 'exact' : 'longest_side',
    '--width',
    String(Math.max(1, Math.min(8192, Number(width) || 1024))),
    '--height',
    String(Math.max(1, Math.min(8192, Number(height) || 1024))),
    '--fit',
    ['fit', 'cover', 'pad', 'stretch'].includes(fit) ? fit : 'fit',
    '--pad-color',
    String(padColor || 'black'),
    '--format',
    ['keep', 'jpg', 'png', 'webp'].includes(format) ? format : 'keep',
    '--quality',
    String(Math.max(1, Math.min(100, Number(quality) || 92))),
  ];

  return spawnImageOpStream({
    request,
    script: 'resize_images.py',
    images: validImages,
    extraArgs,
  });
}
