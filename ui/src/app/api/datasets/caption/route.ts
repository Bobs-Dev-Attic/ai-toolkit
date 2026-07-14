import { NextResponse } from 'next/server';
import {
  filterImagesWithinFolder,
  resolveDatasetFolder,
  spawnImageOpStream,
} from '@/server/imageOps';
import { applyHfCacheEnv } from '@/server/hfCache';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  await applyHfCacheEnv();
  let body: any;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }

  const {
    datasetName,
    overwrite = false,
    triggerWord = '',
    limit = 0,
    images,
    style,
    prompt,
    repetitionPenalty,
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

  const extraArgs: string[] = ['--trigger-word', String(triggerWord || '')];
  if (style === 'short' || style === 'standard' || style === 'detailed') {
    extraArgs.push('--style', style);
  }
  if (typeof prompt === 'string' && prompt.trim()) {
    extraArgs.push('--prompt', prompt.trim());
  }
  const penalty = Number(repetitionPenalty);
  if (Number.isFinite(penalty) && penalty > 0) {
    extraArgs.push('--repetition-penalty', String(penalty));
  }
  if (Number(limit) > 0) extraArgs.push('--limit', String(limit));

  // Caption supports two modes: full-dataset (no images selected) and explicit
  // image-list (selected images, replacing existing captions). Only in
  // full-dataset mode is --overwrite meaningful.
  const useImageList = Array.isArray(images) && images.length > 0;
  if (useImageList) {
    const validImages = filterImagesWithinFolder(images, datasetFolder);
    if (validImages.length === 0) {
      return NextResponse.json({ error: 'No valid images in selection' }, { status: 400 });
    }
    return spawnImageOpStream({
      request,
      script: 'caption_dataset.py',
      datasetFolder,
      images: validImages,
      extraArgs,
    });
  }

  if (overwrite) extraArgs.push('--overwrite');
  return spawnImageOpStream({
    request,
    script: 'caption_dataset.py',
    datasetFolder,
    images: [],
    skipImageList: true,
    extraArgs,
  });
}
