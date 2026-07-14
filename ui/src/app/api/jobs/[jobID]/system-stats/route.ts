import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

export const runtime = 'nodejs';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  // this must be awaited to avoid TS error
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const logPath = path.join(jobFolder, 'system_stats.jsonl');

  if (!fs.existsSync(logPath)) {
    return NextResponse.json({ points: [], keys: [] });
  }

  const url = new URL(request.url);
  // Only return samples strictly newer than this wall-clock timestamp (seconds).
  // Lets the client poll incrementally instead of re-fetching the whole file.
  const sinceParam = url.searchParams.get('since');
  const since = sinceParam != null ? Number(sinceParam) : null;

  let content = '';
  try {
    content = await fs.promises.readFile(logPath, 'utf-8');
  } catch {
    return NextResponse.json({ points: [], keys: [] });
  }

  const points: Record<string, number | null>[] = [];
  const keySet = new Set<string>();

  for (const line of content.split('\n')) {
    const s = line.trim();
    if (!s) continue;
    let obj: Record<string, number | null>;
    try {
      obj = JSON.parse(s);
    } catch {
      // a partially-written trailing line (sampler mid-write) — skip it
      continue;
    }
    if (since != null && typeof obj.t === 'number' && obj.t <= since) continue;
    for (const k of Object.keys(obj)) keySet.add(k);
    points.push(obj);
  }

  return NextResponse.json({ points, keys: Array.from(keySet) });
}
