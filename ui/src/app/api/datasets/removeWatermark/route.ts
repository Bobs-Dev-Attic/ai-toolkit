import { NextResponse } from 'next/server';
import {
  filterImagesWithinFolder,
  resolveDatasetFolder,
  spawnImageOpStream,
} from '@/server/imageOps';
import { applyHfCacheEnv } from '@/server/hfCache';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const ALLOWED_MODELS = new Set(['grounding-dino-tiny', 'grounding-dino-base']);
const ALLOWED_TARGETS = new Set(['text', 'watermark', 'logo']);

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
    images,
    model = 'grounding-dino-tiny',
    targets = ['text', 'watermark', 'logo'],
    prompt = '',
    threshold = 0.3,
    dilate = 6,
    outputMode = 'sidecar',
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

  const targetList = (Array.isArray(targets) ? targets : [])
    .filter((t: unknown): t is string => typeof t === 'string' && ALLOWED_TARGETS.has(t));
  const targetsArg = targetList.length ? targetList.join(',') : 'text,watermark,logo';

  const dilatePx = Number.isFinite(dilate) ? Math.min(Math.max(Math.round(dilate), 0), 64) : 6;
  const thresholdVal = Number.isFinite(threshold) ? Math.min(Math.max(threshold, 0.05), 0.95) : 0.3;

  const extraArgs = [
    '--model',
    ALLOWED_MODELS.has(model) ? model : 'grounding-dino-tiny',
    '--targets',
    targetsArg,
    '--threshold',
    String(thresholdVal),
    '--dilate',
    String(dilatePx),
    '--output-mode',
    outputMode === 'replace' ? 'replace' : 'sidecar',
  ];

  if (typeof prompt === 'string' && prompt.trim()) {
    extraArgs.push('--prompt', prompt.trim());
  }

  return spawnImageOpStream({
    request,
    script: 'remove_watermark.py',
    images: validImages,
    extraArgs,
  });
}
