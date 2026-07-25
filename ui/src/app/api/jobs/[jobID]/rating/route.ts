import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

export const runtime = 'nodejs';

const prisma = new PrismaClient();

// A run's human quality judgement, stored as a sidecar file in the job output
// folder (rating.json). Sidecar rather than a DB column so it lives next to the
// run's own logs/config — a future cross-run aggregator (Stage C) can just walk
// output/*/rating.json alongside system_stats.jsonl, with no schema migration.
export interface RunRating {
  score?: number | null; // 1-5, overall quality
  likeness?: 'good' | 'bad' | null; // for likeness jobs: did it resemble the subject?
  notes?: string;
  rated_at?: string; // ISO timestamp, set server-side on write
}

const RATING_FILE = 'rating.json';

async function jobFolder(jobID: string): Promise<string | null> {
  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return null;
  const trainingFolder = await getTrainingFolder();
  return path.join(trainingFolder, job.name);
}

export async function GET(_request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const folder = await jobFolder(jobID);
  if (!folder) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const file = path.join(folder, RATING_FILE);
  if (!fs.existsSync(file)) return NextResponse.json({ rating: null });
  try {
    const rating = JSON.parse(await fs.promises.readFile(file, 'utf-8'));
    return NextResponse.json({ rating });
  } catch {
    return NextResponse.json({ rating: null });
  }
}

export async function POST(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const folder = await jobFolder(jobID);
  if (!folder) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  if (!fs.existsSync(folder)) {
    return NextResponse.json({ error: 'Job output folder does not exist yet' }, { status: 400 });
  }

  let body: RunRating;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  // Validate/normalize — never trust the client shape.
  const rating: RunRating = { rated_at: new Date().toISOString() };
  if (body.score != null) {
    const s = Number(body.score);
    if (!Number.isFinite(s) || s < 1 || s > 5) {
      return NextResponse.json({ error: 'score must be 1-5' }, { status: 400 });
    }
    rating.score = Math.round(s);
  } else {
    rating.score = null;
  }
  rating.likeness = body.likeness === 'good' || body.likeness === 'bad' ? body.likeness : null;
  if (typeof body.notes === 'string') rating.notes = body.notes.slice(0, 2000);

  try {
    await fs.promises.writeFile(path.join(folder, RATING_FILE), JSON.stringify(rating, null, 2), 'utf-8');
  } catch (e) {
    return NextResponse.json({ error: `Could not save rating: ${String(e)}` }, { status: 500 });
  }
  return NextResponse.json({ rating });
}
