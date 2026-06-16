import { NextResponse } from 'next/server';
import {
  filterImagesWithinFolder,
  resolveDatasetFolder,
  spawnImageOpStream,
} from '@/server/imageOps';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const ALLOWED_TARGETS = new Set(['face', 'upper_body', 'torso', 'person']);
const ALLOWED_TORSO_TIGHTNESS = new Set(['tight', 'medium', 'wide']);

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
    target = 'face',
    padding = 0.2,
    outputSizeMode = 'none', // 'none' | 'square' | 'exact'
    squareSize = 512,
    exactWidth = 512,
    exactHeight = 512,
    fit = 'cover',
    padColor = 'black',
    outputMode = 'sidecar',
    minConfidence = 0.4,
    torsoTightness = 'medium',
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

  let outputSizeArg = 'none';
  if (outputSizeMode === 'square') {
    const n = Math.max(16, Math.min(4096, Number(squareSize) || 512));
    outputSizeArg = `square:${n}`;
  } else if (outputSizeMode === 'exact') {
    const w = Math.max(16, Math.min(8192, Number(exactWidth) || 512));
    const h = Math.max(16, Math.min(8192, Number(exactHeight) || 512));
    outputSizeArg = `exact:${w}x${h}`;
  }

  const extraArgs = [
    '--target',
    ALLOWED_TARGETS.has(target) ? target : 'face',
    '--padding',
    String(Math.max(0, Math.min(2, Number(padding) || 0))),
    '--output-size',
    outputSizeArg,
    '--fit',
    ['fit', 'cover', 'pad', 'stretch'].includes(fit) ? fit : 'cover',
    '--pad-color',
    String(padColor || 'black'),
    '--output-mode',
    outputMode === 'replace' ? 'replace' : 'sidecar',
    '--min-confidence',
    String(Math.max(0, Math.min(1, Number(minConfidence) || 0.4))),
    '--torso-tightness',
    ALLOWED_TORSO_TIGHTNESS.has(torsoTightness) ? torsoTightness : 'medium',
  ];

  return spawnImageOpStream({
    request,
    script: 'auto_crop.py',
    images: validImages,
    extraArgs,
  });
}
